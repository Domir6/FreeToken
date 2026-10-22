# FreeToken KV Swap 补丁文档

> 版本：2026-10-22（已适配 FreeToken 0.2.0；含 idle preload、pinned OOM 修复、QSA index tiers 持久化）
> 基线：本地隔离目录 `C:\Users\X303\Documents\FreetokenP`，FreeToken **0.2.0+g425dc7bf1** 运行时（torch 2.14.1+cu130）
> 目标：把 GPU KV 前缀缓存扩展到 **host RAM + SSD** 两级，停服落盘、重启恢复；升级 FreeToken 版本后可照本文重新打补丁。
>
> **0.2.0 起 `scheduler/scheduler.py` 与 `server/api_server.py` 已变为 `.pyd`（不可改）**，调度器相关补丁改为 `freetoken/kv_swap/runtime.py` 里的运行时 monkeypatch，详见第 16 节。第 7、11.1、11.4~11.7 节保留为 0.1.3 时代的历史参考。

---

## 0. 一句话总结

FreeToken 原生只有 GPU 内的 paged KV + radix 前缀复用。本补丁在 radix 树之外加了两级缓存：

- **host RAM**（`KVHostStore`）：每个可驱逐的 radix 节点存一份 `tokens + K/V + GDN 快照`；
- **SSD**（`SSDStore`）：host 装不下时溢出到磁盘，进程重启后仍能命中；
- **QSA 专属状态**：本模型走 `qsa_sparse` 后端，除 K/V 外每个 token 还有两份状态（`_cmp_k_buffer` 压缩 index key、`_rope_positions` mrope 位置），必须与 K/V 同存同取；缺了它短上下文看不出，长上下文会“忘事”（见第 15 节）。

落盘触发是**停服脚本写标记文件**，调度器在 idle 时消费；恢复是在 prefill 准入时沿 radix 命中点向后逐段 `swap-in`。

---

## 1. 它做什么 / 不做什么

做：

- 停服时把 GPU 里所有 **unlocked** 前缀（含 GDN/mamba 状态）落盘；
- 重启后相同前缀直接从 host/SSD 恢复，不重算 prefill；
- 会话空闲时（等工具返回、等下一个问题）把前缀**预搬运**到 host，后续被驱逐时零成本；
- 共享前缀只存一份，子智能体/多会话分叉互不串扰；
- 落盘格式带 shape/dtype 指纹（`signature`），换模型/换量化不会误加载。

不做：

- 不做 GPU↔host 的**逐 token** 换页（没有 page fault，没有 unified memory）；
- 不改变 decode 的 KV 布局；
- 不缓存**正在生成**的请求（只缓存 `ref_count == 0` 的树节点）；
- 不保证被驱逐后一定命中（`CAPTURE_ON_EVICT` 默认关，驱逐时若没提前预搬运就丢失）；
- 不支持 SWA 模型（`attempt_host_restore` 对 `is_swa` 直接返回）。

---

## 2. 架构与数据流

### 2.1 radix 节点与 KV 段

FreeToken 的 hybrid radix 树按 `page_size`（本模型 64）分裂。**每个节点只持有自己那一段 KV**：

```
root
 └── node A (tokens 0..4096,  kv_start=0,    length=4096, mamba=slot 3)
      └── node B (tokens 4096..8192, kv_start=4096, length=4096, mamba=slot 7)
           └── node C (tokens 8192..8256, kv_start=8192, length=64,  mamba=None)
```

- `node.value` 是这段 KV 在 GPU page pool 里的索引；
- `node.length` 是**段长**，不是前缀长；
- 前缀长 = 祖先链 `length` 之和；
- `HostEntry.kv_start = prefix_len - node.length`，恢复端靠它判断“这一段能不能接在我已有的命中后面”。

### 2.2 dump（停服落盘）

```
stop_server_isolated.ps1
  ├─ 写 <persist_dir>\.dump_request
  ├─ 发一个 1-token 请求（制造 idle 转换）
  ├─ 轮询 flag 消失（≤300s）
  └─ CTRL_C_EVENT / 杀进程树

scheduler.run_when_idle
  ├─ 发现 flag
  ├─ _dump_kv_to_ssd()
  │    ├─ capture_resident(cache, max_bytes=None)   # 沿每个叶子的祖先链根→叶
  │    │    └─ capture_node(node, min_tokens=page_size)
  │    │         ├─ 跳过 ref_count != 0 的节点
  │    │         ├─ gather_kv 分块 D2H 到 CPU（默认 pageable）
  │    │         ├─ gather_index_tiers（QSA：cmp index 行 + mrope 位置，见 15.2）
  │    │         ├─ snapshot_slot 复制 GDN 四件套
  │    │         └─ 插入 host tier（超 16GiB 则 _spill 到 SSD）
  │    └─ flush_ssd(max_bytes=1<<62)
  └─ 删除 flag
```

### 2.3 restore（重启恢复）

```
请求到达 → PrefillAdder._try_allocate_one
  ├─ match_req(req)                      # 只查 GPU radix
  ├─ lock(handle)                        # 先拿锁
  ├─ while True:
  │    extended = attempt_host_restore(input_ids, handle)
  │    if extended is handle: break      # 没有下一段了
  │    handle = extended                 # 锁已迁移到新 handle
  │    if 剩余 KV 不够: break
  ├─ 空间/mamba/SWA 三项检查（不足则 unlock 后 return None）
  └─ 返回 (handle, table_idx, live_slot, ping_pong, mamba_value)
```

`attempt_host_restore` 每次只恢复**一段**（一个 host entry）：

1. `store.lookup(input_ids, min_len=cached_len)`：要求 `entry.kv_start <= cached_len`（条目可以回伸到 GPU 命中点之前，例如两个会话共享同一段模板前缀）且 `entry.length > cached_len`；
2. 计算 `skip = cached_len - entry.kv_start`、`add = entry.length - cached_len`，两者都必须是 `page_size` 的整数倍，否则放弃这段；
3. `_allocate(add // page_size)` 拿新页；
4. `kv_pool.scatter_kv(slots, entry.kv_k[:, skip:], entry.kv_v[:, skip:])` 把 K/V 写回 GPU；
5. QSA：`scatter_index_tiers(slots, entry.cmp_index[:, skip // index_ratio:], entry.rope_pos[skip:])` 写回压缩 index 行和 mrope 位置；旧条目的 `cmp_index`/`rope_pos` 为 `None`，跳过（见 15.4）；
6. hybrid 模型：`linear_state_pool.alloc(1)` + `restore_slot(slot, entry.mamba)`；
7. `prefix_cache.insert(ids, full_slots, mamba_slot)` 把这段接进 radix 树；
8. `unlock(old) + lock(new)` 迁移锁；
9. `store.take(entry)` 从 host 删除（已搬到 GPU）。

### 2.4 idle preload（空闲预搬运）

```
run_when_idle（每 ≥1s）
  └─ store.capture_resident(cache, max_bytes=1GiB)
       └─ 遍历 unlocked 叶子，沿祖先链根→叶，把还没捕获的节点搬进 host
```

效果：A 会话在等工具返回时，KV 已进 host；B 进来需要空间时直接驱逐 A（host 副本已在），A 回来 `swap-in` 秒级恢复。

### 2.5 evict hook（默认关闭）

`FREETOKEN_KV_SWAP_CAPTURE_ON_EVICT=1` 时，驱逐叶子前会同步 `capture_node`。默认关，因为同步 D2H 会卡 decode。

### 2.6 SSD 布局

```
<FREETOKEN_KV_PERSIST_DIR>/
  <sha1(signature)[:16]>/
    index.json                     # {entries: {文件名: {first,length,kv_start,sha1,bytes,atime}}}
    entries/
      <sha1(tokens)[:20]>_<kv_start>_<length>.pt
```

`signature` = `page=<n>|kv=<dtype>|<shape>|conv_states=...|recurrent_states=...`，换模型/量化/页大小都会落到不同目录。

---

## 3. 文件清单

| 文件 | 类型 | 作用 |
|---|---|---|
| `freetoken/kv_swap/__init__.py` | 新增 | 导出 `KVHostStore` / `HostEntry` / `create_host_store` |
| `freetoken/kv_swap/manager.py` | 新增 | host tier、捕获、查找、溢出、idle/dump 入口 |
| `freetoken/kv_swap/ssd.py` | 新增 | SSD tier、index、LRU 淘汰 |
| `freetoken/scheduler/scheduler.py` | 修改 | 创建 host store、idle 消费 flag/预搬运、dump 入口、mamba 恢复 |
| `freetoken/scheduler/cache.py` | 修改 | `match_req` 透传 mamba、`attempt_host_restore`、evict hook、host store 注入 |
| `freetoken/scheduler/prefill.py` | 修改 | 准入时先锁、循环恢复、`mamba_value` 取最终节点 |
| `freetoken/message/backend.py` | 修改 | `CacheRebuildBackendMsg.dump_only` |
| `freetoken/message/tokenizer.py` | 修改 | `CacheRebuildMsg.dump_only` |
| `freetoken/tokenizer/server.py` | 修改 | 透传 `dump_only` |
| `freetoken/server/api_server.py` | 修改 | `dispatch_rebuild(dump_only=)`、`POST /v1/cache/dump` |
| `freetoken/cli.py` | 修改 | 捕获 `KeyboardInterrupt` 返回 130 |
| `scripts/start_server_isolated.ps1` | 脚本 | 环境变量、PYTHONPATH、启动参数 |
| `scripts/stop_server_isolated.ps1` | 脚本 | flag + poke + 轮询 + CTRL_C + 孤儿清理 |

> `freetoken/message/frontend.py` 的 `CacheRebuildReply` 是上游已有的 runtime rebuild 消息，本补丁没有改它（`dump_only` 只走 backend/tokenizer 两跳）。

---

## 4. 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `FREETOKEN_KV_SWAP` | `0` | 总开关；0 时 `create_host_store` 返回 None |
| `FREETOKEN_KV_SWAP_HOST_BYTES` | `16GiB` | host tier 逻辑上限（超过 spill 到 SSD） |
| `FREETOKEN_KV_SWAP_MIN_TOKENS` | `512` | `capture_node` 的默认最小段长（dump/idle 路径会显式传 `page_size`） |
| `FREETOKEN_KV_SWAP_PIN` | `0` | 1 = pinned memory；**生产必须 0**，否则会耗尽 Windows pin 配额 |
| `FREETOKEN_KV_SWAP_COPY_CHUNK` | `8192` | D2H 分块大小（token 数） |
| `FREETOKEN_KV_SWAP_CAPTURE_ON_EVICT` | `0` | 1 = 驱逐时同步捕获（会卡 decode） |
| `FREETOKEN_KV_SWAP_PRELOAD_ON_IDLE` | `1` | 1 = idle 时预搬运 |
| `FREETOKEN_KV_SWAP_IDLE_BYTES` | `1GiB` | 每次 idle 预搬运预算 |
| `FREETOKEN_KV_PERSIST` | `0` | 1 = 启用 SSD tier |
| `FREETOKEN_KV_PERSIST_DIR` | `<cwd>/kv_persist` | SSD 根目录 |
| `FREETOKEN_KV_PERSIST_BYTES` | `64GiB` | SSD 上限（LRU 淘汰） |
| `FREETOKEN_KV_PERSIST_FLUSH_BYTES` | `1GiB` | `flush_ssd` 每次写盘预算 |
| `FREETOKEN_KV_RESTORE` | `1` | 0 = 只落盘不恢复（SSD 仍写入，但重启后照常全量 prefill） |
| `FREETOKEN_KV_DIAG` | `0` | 1 = 追加写 `C:\AI_Models\logs\kv_diag.log`（`free/tree/gap`、restore ok/FAIL、`fin` 行） |
| `FREETOKEN_KV_RESTORE_VERIFY` | `0` | 1 = 恢复后逐元素校验 K/V、index 行、rope 位置、mamba，写 `restore_verify.log` |

推荐生产配置：

```powershell
$env:FREETOKEN_KV_SWAP = "1"
$env:FREETOKEN_KV_SWAP_HOST_BYTES = "17179869184"   # 16 GiB
$env:FREETOKEN_KV_SWAP_MIN_TOKENS = "256"
$env:FREETOKEN_KV_SWAP_PIN = "0"                    # 关键
$env:FREETOKEN_KV_SWAP_COPY_CHUNK = "2048"
$env:FREETOKEN_KV_PERSIST = "1"
$env:FREETOKEN_KV_PERSIST_DIR = "C:\Users\X303\Documents\FreetokenP\kv_persist"
$env:FREETOKEN_KV_PERSIST_BYTES = "68719476736"     # 64 GiB
```

---

## 5. 启动 / 停服 / 落盘

### 5.1 启动

```powershell
$env:PYTHONPATH = "C:\Users\X303\Documents\FreetokenP;$env:PYTHONPATH"
& "C:\Users\X303\AppData\Local\FreeToken\venv\Scripts\ft.exe" serve `
    --model "C:\AI_Models\Qwen3.8-Flash-Next-NVFP4" `
    --port 1919 --moe-strategy offload --image-max-tokens 512 `
    --memory-ratio 0.95 --num-tokens 98304 --moe-cache-size 512 `
    --moe-cpu-layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,34,35,36,37,38,39,40,41,42,43,44,45,46,47 `
    --max-running-requests 2 --cuda-graph-max-bs 1 --moe-cpu-threads 16 `
    --gpu GPU-54e316a5-5b97-c3dd-ebbc-d54e2f80cf22 --host 127.0.0.1
```

关键点：

- `PYTHONPATH` 必须把隔离目录放在最前：`ft.exe` 的 `.pyd` 用的是 venv 里的二进制，但 `.py`（scheduler/kvcache/kv_swap/...）会从隔离目录加载；
- 启动约 95–130s 到 ready（加载 15.5GB 权重 + CUDA graph）。

### 5.2 停服（必须用脚本）

```powershell
& "C:\Users\X303\Documents\FreetokenP\scripts\stop_server_isolated.ps1"
```

脚本流程：

1. 写 `<KvPersistDir>\.dump_request`；
2. 发 1-token 请求制造 idle；
3. 轮询 flag 消失（≤300s）；
4. `CTRL_C_EVENT` 优雅退出；超时则杀进程树；
5. 清扫孤儿 worker。

**不要用 `Stop-Process` / `terminate()` 直接杀**：Windows 上硬杀不会执行 Python 清理，dump 不会发生。

**停服时不能有正在生成的请求**：`dump_only` 在 `prefill_manager.runnable || decode_manager.runnable` 时回复 `busy`，脚本会等 300s 后强杀。

### 5.3 落盘成功的样子

```
host KV dump: status=ok captured=9 persisted=9 | host_kv entries=9 bytes=2.91GiB ...
```

`captured` 是 host tier 捕获的节点数，`persisted` 是写到 SSD 的条目数。

### 5.4 恢复成功的样子

```
host KV swap-in: restored 9152 tokens (prefix 9152), host_kv entries=0 ... ssd_loaded=1
Prefill batch, #new-seq: 1, #new-token: 42, #cached-token: 9152, ...
```

---

## 6. 验证清单

### 6.1 最小 E2E

```powershell
# 1) 起 16384 小池
& scripts\start_server_isolated.ps1 -NumTokens 16384 -KvPersistDir ...\kv_test -Detach
# 2) 发一个 ~9000 token 的请求（内容固定）
# 3) 停服（dump）
& scripts\stop_server_isolated.ps1 -KvPersistDir ...\kv_test
# 4) 重启
& scripts\start_server_isolated.ps1 -NumTokens 16384 -KvPersistDir ...\kv_test -Detach
# 5) 再发同一个请求：日志应出现 swap-in: restored <N>，耗时明显低于首次
```

期望：

- 首次 prefill 12s 左右（9000 tokens）；
- 重启后同一 prompt 3–4s（restored 9152）；
- 答案与首次一致。

### 6.2 共享前缀 / 子智能体

- A、B 共享 960-token 前缀后分叉；
- B 首次 `#cached-token: 960`，只 prefill 42；
- A/B 交替 3 轮答案各自正确（不串扰）；
- 用 13992-token 的 C 把池压到 86% 后，A/B 仍命中 960；
- 停服 dump 出 `0-960`，重启 `restored 960`，A/B 仍正确。

### 6.3 检查 SSD 完整性

```powershell
$dir = "C:\Users\X303\Documents\FreetokenP\kv_persist"
Get-ChildItem $dir -Recurse -Filter index.json | ForEach-Object {
    $j = Get-Content $_.FullName -Raw | ConvertFrom-Json
    $j.entries.PSObject.Properties | ForEach-Object {
        "{0}  len={1}  kv_start={2}  bytes={3}" -f $_.Name, $_.Value.length, $_.Value.kv_start, $_.Value.bytes
    }
}
```

要求：

- 索引条目与 `entries/*.pt` 文件一一对应；
- `kv_start` 链从 0 开始连续（`0 → a → b → ...`），不能有洞；
- 每个 `.pt` 的 `tokens` 长度 == `length`，`kv_k/kv_v` 的 token 维 == 段长，`mamba` 含 `conv_states`/`recurrent_states`/`ple_conv`/`ple_ngram_ctx`。

### 6.4 QSA index tiers 校验（长上下文必做）

`FREETOKEN_KV_RESTORE_VERIFY=1` 后，恢复路径会逐元素比对 K/V、压缩 index 行、mrope 位置与 mamba 快照，追加写到 `C:\AI_Models\logs\restore_verify.log`。

```powershell
$env:FREETOKEN_KV_RESTORE_VERIFY = "1"
# 1) 发一个 >8192 token 的 prompt（务必超过 indexer 的 2048 组阈值，即 8192 token）
# 2) 按 5.2 停服 dump
# 3) 重启，发完全相同的 prompt
# 4) 看日志：必须同时出现 dCmp / dRope
```

期望（本机 22272-token 前缀实测）：

```
KV restore verify: max|dK|=0 max|dV|=0 skip=0 add=22272 length=22272 max|dCmp|=0 max|dRope|=0
mamba restore verify: max|d|=0 keys=4
```

注意：小于 8192 token 的 prompt 也能恢复，但 indexer 不触发稀疏选择，`dCmp/dRope` 即使缺失也不会立刻表现为错误，**不能**用它判断修复是否生效。

---

## 7. 版本升级注意事项（重点）

### 7.1 先确认能不能改

FreeToken 的运行时由 **预编译 `.pyd`** + **纯 Python `.py`** 混合组成。升级后先确认目标文件是哪种：

| 目录/文件 | 类型 | 能否打补丁 |
|---|---|---|
| `engine/`（`engine.py` 等） | `.pyd` | ❌ 不能改 |
| `kernel/*` | `.pyd` | ❌ |
| `daemon/*` | `.pyd` | ❌ |
| `models/*/weight*` | `.pyd` | ❌ |
| `distributed/impl` | `.pyd` | ❌ |
| `checkpoint/ftw` | `.pyd` | ❌ |
| `moe/offload_kernels` | `.pyd` | ❌ |
| `scheduler/*` | **0.2.0 起 `.pyd`** | ❌ 改用运行时 monkeypatch（见第 16 节） |
| `kvcache/*` | `.py` | ✅ |
| `layers/*` | `.py` | ✅ |
| `moe/offload_cache.py`、`moe/prefetch.py` | `.py` | ✅ |
| `server/api_server.py` | **0.2.0 起 `.pyd`** | ❌（本项目已不需要） |
| `tokenizer/server.py`、`message/*` | `.py` | ✅（0.2.0 起无需改） |
| `cli.py` | `.py` | ✅ |

判断方法：

```powershell
Get-ChildItem "C:\Users\X303\AppData\Local\FreeToken\venv\Lib\site-packages\freetoken\scheduler" | Select Name
```

### 7.2 v0.2.0 的包路径变了

上游 v0.2.0 起源码树是 `python/freetoken/`（不是 `freetoken/`），安装后的 site-packages 仍是 `freetoken/`。本补丁的新增目录 `freetoken/kv_swap/` 在 v0.2.0 里**不存在**，需要手动复制进去（纯 Python，不涉及编译）。

### 7.3 上游可能冲突的改动

v0.2.0 相对 0.1.3 动过这些与补丁重叠的地方：

- **radix insert**：`#559 Clone inserted radix keys to release request buffers` —— `insert()` 的签名/返回可能变；
- **scheduler**：overlap 调度、`max_tokens` 收尾（#546）、`rebuild` 逻辑；
- **MoE 缓存**：`#689` 拆 slot pool（`--moe-cache-scope`）、`#700` 每格式一个 pool、`#742` page-locked banks —— 不影响 KV swap，但会影响 `moe` 相关代码；
- **kvcache**：`#367 Reserve paged KV at allocation granularity`（v0.1.3）—— `allocate_paged` 语义；
- **`hybrid_radix_cache.py`**：`match_prefix`、`insert`、`dec_lock`、`_cascade_tombstone_leaves` 是补丁依赖的核心，升级后必须逐行 diff。

### 7.4 移植顺序（建议）

> ⚠ 本节是 0.1.3 → 0.2.0 之前的预案。0.2.0 实际移植结果与差异见**第 16 节**（`scheduler.py` 已不可改，改为 `kv_swap/runtime.py` monkeypatch）。

1. 复制 `freetoken/kv_swap/` 三个文件（原样）；
2. 改 `scheduler/cache.py`：`CacheManager.__init__` 加 `host_store` 参数、`match_req` 透传 `mamba_value`、`attempt_host_restore`、`_attach_evict_hook`；
3. 改 `scheduler/prefill.py`：`_try_allocate_one` 的锁契约 + 循环恢复 + `mamba_value`；
4. 改 `scheduler/scheduler.py`：`create_host_store`、`run_when_idle`、`_dump_kv_to_ssd`、`_execute_pending_rebuild` 的 `dump_only` 分支、`_restore_linear_states`；
5. 改 message/tokenizer/api 三处 `dump_only` 透传；
6. 改 `cli.py` 的 `KeyboardInterrupt`；
7. 更新脚本的 `FREETOKEN_KV_SWAP_PIN=0`。

### 7.5 必须重跑的验证

升级后至少重跑：

1. 6.1 最小 E2E（dump → 重启 → restore）；
2. 6.2 共享前缀（A/B 交替 + 驱逐 + 恢复）；
3. 锁契约：`check_integrity` 不报 `assert cur.ref_count >= 0`；
4. 长会话（>74k tokens）dump 后索引链从 0 连续；
5. idle preload 日志出现 `captured=N` 且无 `failed=` 增长；
6. 停服后日志无 `KeyboardInterrupt` traceback。

### 7.6 不要做的事

- 不要改 `.pyd` 文件（改了也不会被加载，且可能破坏 ABI）；
- 不要把 `FREETOKEN_KV_SWAP_PIN` 设回 1（生产会 OOM）；
- 不要在生成中的请求上停服（dump 会 `busy`，脚本 300s 后强杀，KV 丢失）；
- 不要跨模型/跨量化复用 `kv_persist` 目录（signature 会挡住，但白占空间）；
- 不要用 `--moe-strategy fused` 时启用本补丁（`attempt_host_restore` 依赖 `kv_pool.gather_kv`，fused 走 expert banks，不适用）。

---

## 8. 已知坑与修复记录

### 8.1 锁泄漏（最严重）

`match_req` 返回的 handle **没有加锁**；`attempt_host_restore` 在结尾做 `unlock(old) + lock(new)`（迁移锁）。因此：

- `_try_allocate_one` 必须在 restore 循环**之前** `lock(handle)`；
- restore 循环里 `handle = extended` 后**不能**再 lock；
- 所有提前 `return None` 的路径（KV 空间不足、mamba 槽不足、SWA 不足）必须先 `unlock(handle)`。

违反后果：`ref_count` 变负 → `dec_lock` 的 `assert cur.ref_count >= 0` 失败 → scheduler 退出。

### 8.2 mamba 恢复源用错

`restore_src` 必须取**最终** `handle.node.mamba_value`，不能用最初的 `mr.mamba_value`：

- host restore 把 KV 前缀拉长到 N，但 `mr.mamba_value` 只到 M（M < N）；
- GPU 命中为 0 时 `mr.mamba_value` 是 None，而 `pool.alloc()` **不清零**，live slot 里是上一个请求的残留状态 → 模型“记错”前缀。

### 8.3 pinned 内存 OOM

`capture_node` 曾用 `pin_memory=True` 按节点大小分配。生产 98304 池捕获到 ~11.6GiB pinned 后耗尽 Windows pin 配额 → `cudaErrorMemoryAllocation`，随后 `match_prefix` 的 `torch.cat` 也 OOM → scheduler 退出。

修复：

- `FREETOKEN_KV_SWAP_PIN` 默认 0；
- `capture_node` 内部捕获异常并计 `failed`，不冒泡；
- idle 单次预算 4GiB → 1GiB；
- OOM 后 `_idle_capture_disabled` 熔断。

### 8.4 64-token 空洞

dump 的段必须从 0 连续。`min_tokens` 用 512 时会跳过 64..511 的节点，留下洞 → 恢复在洞前停住。dump/idle/evict 三条路径都必须用 `page_size`（64）作阈值。

### 8.5 不足一页的尾部

`insert` 只缓存 `align_down(len, page_size)` 的部分。分叉尾部不足 64 tokens 时每次重算（42 tokens），属预期，不影响正确性。

### 8.6 CUDA graph capture 期间不捕获

`capture_node` 开头检查 `torch.cuda.is_current_stream_capturing()`，捕获中直接 `skipped`，避免在 graph capture 里做 D2H/分配。

### 8.7 驱逐内部节点只回收 mamba 槽

`evict_full` 对内部节点走 `else` 分支：只 `_free_node_mamba`，**KV 不释放**。共享前缀因此不会被“驱逐主会话”连带删掉；代价是 mamba 快照可能被回收，`match_prefix` 会退到更早的快照点。

### 8.8 停服 traceback

uvicorn 的 `capture_signals` 在 `Application shutdown complete` 后会重新 raise `CTRL_C_EVENT`。`cli._run_serve` 捕获 `KeyboardInterrupt` 返回 130，避免打印堆栈。

### 8.9 QSA index tiers 缺失（长上下文“记忆错乱”）

现象：长会话（>8192 token）重启恢复后，模型答非所问、忘记前文；短会话与 KV 数值校验全部正常，因此极易误判为“模型长上下文衰减”。

根因：`qsa_sparse` 后端的 `_cmp_k_buffer`（每 `index_ratio=4` 个 token 一行的压缩 index key）和 `_rope_positions`（每 token 三轴 rope 位置）不在 K/V 页里，`gather_kv/scatter_kv` 不搬。恢复后新池这两份是全 0，稀疏注意力据此选块、压缩组据此取 rope，长上下文必然错。

修复：`QSAKVCache.gather_index_tiers/scatter_index_tiers` + `HostEntry.cmp_index/rope_pos` + SSD payload 两个新字段 + restore 时按 `skip` 截断写回（第 15 节）。

注意：旧版本 dump 的 `*.pt` 没有这两个字段，`lookup` 用 `data.get` 读成 `None` 并跳过，不会崩，但**仍然错乱**。升级后必须让长会话全量 prefill 一次重新 dump（或删掉旧条目）。

---

## 9. 回滚

```powershell
# 1) 停服（先 dump）
& scripts\stop_server_isolated.ps1
# 2) 删除/改名 kv_swap 目录，恢复备份的 4 个 .py
# 3) 把环境变量 FREETOKEN_KV_SWAP / FREETOKEN_KV_PERSIST 设为 0
# 4) 重启
```

`kv_persist` 目录可保留（signature 不匹配时不会加载），确认不再需要后手动删除。

---

## 10. 新增文件全文

### 10.1 `freetoken/kv_swap/__init__.py`

```python
from __future__ import annotations

from .manager import HostEntry, KVHostStore, create_host_store

__all__ = ["KVHostStore", "HostEntry", "create_host_store"]
```

### 10.2 `freetoken/kv_swap/manager.py`

```python
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

GIB = 1 << 30


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


@dataclass
class HostEntry:
    tokens: torch.Tensor
    kv_k: torch.Tensor
    kv_v: torch.Tensor
    mamba: Optional[dict]
    kv_start: int
    last_used: float = field(default_factory=time.monotonic)

    @property
    def length(self) -> int:
        return int(self.tokens.numel())

    @property
    def nbytes(self) -> int:
        total = self.tokens.numel() * self.tokens.element_size()
        total += self.kv_k.numel() * self.kv_k.element_size()
        total += self.kv_v.numel() * self.kv_v.element_size()
        if self.mamba:
            for t in self.mamba.values():
                total += t.numel() * t.element_size()
        return int(total)


class KVHostStore:
    def __init__(
        self,
        kv_pool,
        linear_pool,
        page_size: int,
        capacity_bytes: int | None = None,
        min_tokens: int | None = None,
        pin_memory: bool | None = None,
    ) -> None:
        self.kv_pool = kv_pool
        self.linear_pool = linear_pool
        self.page_size = int(page_size)
        self.capacity_bytes = (
            _env_int("FREETOKEN_KV_SWAP_HOST_BYTES", 16 * GIB)
            if capacity_bytes is None
            else int(capacity_bytes)
        )
        self.min_tokens = (
            _env_int("FREETOKEN_KV_SWAP_MIN_TOKENS", 512)
            if min_tokens is None
            else int(min_tokens)
        )
        self.pin = (
            _env_bool("FREETOKEN_KV_SWAP_PIN", False) if pin_memory is None else bool(pin_memory)
        )
        self.device = kv_pool.device if kv_pool is not None else torch.device("cuda")
        self._entries: List[HostEntry] = []
        self._by_first: Dict[object, List[HostEntry]] = {}
        self._bytes = 0
        self._lock = threading.RLock()
        self.ssd = None
        self._ssd_flushed: set[str] = set()
        self._captured: set[str] = set()
        self.persist_dir = os.environ.get("FREETOKEN_KV_PERSIST_DIR") or os.path.join(
            os.getcwd(), "kv_persist"
        )
        if _env_bool("FREETOKEN_KV_PERSIST", False):
            try:
                from freetoken.kv_swap.ssd import SSDStore

                self.ssd = SSDStore(
                    self.persist_dir,
                    _env_int("FREETOKEN_KV_PERSIST_BYTES", 64 * GIB),
                    self.page_size,
                    self.min_tokens,
                )
            except Exception:
                self.ssd = None
        self.signature = self._signature()
        self.stats = {
            "saved": 0,
            "saved_bytes": 0,
            "skipped": 0,
            "failed": 0,
            "restored": 0,
            "restored_bytes": 0,
            "host_evicted": 0,
            "ssd_saved": 0,
            "ssd_loaded": 0,
        }

    def _signature(self) -> str:
        """Shape/dtype fingerprint: entries saved by another model or quant are never loaded."""
        parts = [f"page={self.page_size}"]
        buf = getattr(self.kv_pool, "_kv_buffer", None)
        if buf is not None:
            parts.append(f"kv={buf.dtype}|{tuple(buf.shape[1:2])}|{tuple(buf.shape[3:])}")
        for name in ("conv_states", "recurrent_states"):
            t = getattr(self.linear_pool, name, None)
            if isinstance(t, torch.Tensor):
                parts.append(f"{name}={t.dtype}|{tuple(t.shape[1:])}")
        return "|".join(parts)

    @property
    def bytes_used(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._entries)

    def _page_key(self, chunk: torch.Tensor):
        vals = chunk.tolist()
        if self.page_size == 1:
            return vals[0]
        return tuple(vals)

    def _token_prefix(self, node) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        cur = node
        while not cur.is_root():
            parts.append(cur._key.detach().to("cpu", torch.int64))
            cur = cur.parent
        parts.reverse()
        if not parts:
            return torch.empty(0, dtype=torch.int64)
        return torch.cat(parts)

    def capture_node(self, node, min_tokens: int | None = None) -> bool:
        floor = self.min_tokens if min_tokens is None else int(min_tokens)
        # a capture-time eviction must not synchronize or allocate device temporaries
        if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            with self._lock:
                self.stats["skipped"] += 1
            return False
        node_len = int(node.length)
        if node_len < floor:
            with self._lock:
                self.stats["skipped"] += 1
            return False
        if self._bytes >= self.capacity_bytes and self.ssd is None:
            return False
        prefix = self._token_prefix(node)
        if prefix.numel() < floor:
            with self._lock:
                self.stats["skipped"] += 1
            return False
        import hashlib

        key = hashlib.sha1(prefix.numpy().tobytes()).hexdigest()
        with self._lock:
            if key in self._captured:
                return False

        k_cpu = None
        v_cpu = None
        copy_chunk = _env_int("FREETOKEN_KV_SWAP_COPY_CHUNK", 8192)
        try:
            for start in range(0, node_len, copy_chunk):
                k, v = self.kv_pool.gather_kv(node.value[start : start + copy_chunk])
                if k_cpu is None:
                    shape = (k.shape[0], node_len, *k.shape[2:])
                    k_cpu = torch.empty(shape, dtype=k.dtype, pin_memory=self.pin)
                    v_cpu = torch.empty(shape, dtype=v.dtype, pin_memory=self.pin)
                end = start + k.shape[1]
                k_cpu[:, start:end].copy_(k, non_blocking=True)
                v_cpu[:, start:end].copy_(v, non_blocking=True)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        except Exception as exc:  # noqa: BLE001 - a failed capture must not break eviction
            with self._lock:
                self.stats["failed"] += 1
            logger.warning(f"host KV capture failed ({node_len} tokens): {exc!r}")
            return False

        mamba = None
        if node.mamba_value is not None and self.linear_pool is not None:
            mamba = self.linear_pool.snapshot_slot(node.mamba_value)

        entry = HostEntry(
            tokens=prefix,
            kv_k=k_cpu,
            kv_v=v_cpu,
            mamba=mamba,
            kv_start=int(prefix.numel()) - node_len,
        )
        nbytes = entry.nbytes
        if nbytes > self.capacity_bytes:
            return False
        with self._lock:
            if self.ssd is not None:
                self._make_room(nbytes)
            self._insert(entry)
            self._trim()
        return True

    def _insert(self, entry: HostEntry, count: bool = True) -> None:
        key = self._page_key(entry.tokens[: self.page_size])
        self._entries.append(entry)
        self._by_first.setdefault(key, []).append(entry)
        self._bytes += entry.nbytes
        if count:
            self.stats["saved"] += 1
            self.stats["saved_bytes"] += entry.nbytes

    def _remove(self, entry: HostEntry) -> None:
        try:
            self._entries.remove(entry)
        except ValueError:
            return
        key = self._page_key(entry.tokens[: self.page_size])
        bucket = self._by_first.get(key)
        if bucket is not None:
            try:
                bucket.remove(entry)
            except ValueError:
                pass
            if not bucket:
                del self._by_first[key]
        self._bytes -= entry.nbytes

    def _spill(self, victim: HostEntry) -> None:
        key = self._entry_key(victim)
        if self.ssd is not None:
            try:
                # save() is a cheap stat when the file already exists, and re-writes it when
                # the SSD tier evicted it since the last flush
                if self.ssd.save(victim, self.signature):
                    self.stats["ssd_saved"] += 1
                    self._ssd_flushed.add(key)
            except Exception:
                pass
        self._remove(victim)
        self.stats["host_evicted"] += 1

    def _make_room(self, nbytes: int, protected: Optional[HostEntry] = None) -> None:
        while self._bytes + nbytes > self.capacity_bytes:
            candidates = [e for e in self._entries if e is not protected]
            if not candidates:
                return
            self._spill(min(candidates, key=lambda e: e.last_used))

    def _trim(self, protected: Optional[HostEntry] = None) -> None:
        self._make_room(0, protected=protected)

    @staticmethod
    def _entry_key(entry: HostEntry) -> str:
        import hashlib

        return hashlib.sha1(entry.tokens.contiguous().numpy().tobytes()).hexdigest()

    def flush_ssd(self, max_bytes: int | None = None) -> int:
        """Persist resident entries not yet on SSD, up to ``max_bytes`` per call (called from
        the idle loop; a full host tier can be many GiB, so it drains over several cycles)."""
        if self.ssd is None:
            return 0
        budget = (
            _env_int("FREETOKEN_KV_PERSIST_FLUSH_BYTES", 1 * GIB)
            if max_bytes is None
            else int(max_bytes)
        )
        saved = 0
        written = 0
        with self._lock:
            for entry in list(self._entries):
                if written >= budget:
                    break
                key = self._entry_key(entry)
                if key in self._ssd_flushed:
                    continue
                try:
                    if self.ssd.save(entry, self.signature, write_index=False):
                        self._ssd_flushed.add(key)
                        self.stats["ssd_saved"] += 1
                        saved += 1
                        written += entry.nbytes
                except Exception:
                    pass
            if saved:
                self.ssd.flush_index()
        return saved

    @property
    def dump_flag_path(self) -> str:
        return os.path.join(self.persist_dir, ".dump_request")

    def capture_resident(self, cache, max_bytes: int | None = None) -> int:
        """Capture unlocked radix leaves into the host tier WITHOUT evicting them.

        Used by the shutdown/dump path so the SSD tier learns every prefix still resident in
        GPU KV, even when the pool never fills (98304-token pools rarely do). Repeated calls
        are cheap: captured prefixes are remembered in ``_captured`` and skipped.
        """
        budget = (1 << 62) if max_bytes is None else int(max_bytes)
        leaves = getattr(cache, "_leaves", None)
        if leaves is None:
            return 0
        captured = 0
        used = 0
        seen: set[int] = set()
        for leaf in sorted(leaves(), key=lambda n: -int(n.length)):
            if used >= budget:
                break
            # A node stores only its own KV segment, so walk the whole ancestry (root ->
            # leaf) or a stop-time dump would persist nothing but the last segment.
            chain = []
            node = leaf
            while node is not None and not node.is_root():
                chain.append(node)
                node = getattr(node, "parent", None)
            for item in reversed(chain):
                ident = id(item)
                if ident in seen:
                    continue
                seen.add(ident)
                if item.ref_count != 0 or int(item.length) < self.page_size:
                    continue
                before = self._bytes
                if self.capture_node(item, min_tokens=self.page_size):
                    captured += 1
                    used += max(0, self._bytes - before)
                if used >= budget:
                    break
        return captured

    def lookup(self, input_ids, min_len: int = 0) -> Optional[HostEntry]:
        if input_ids.numel() == 0:
            return None
        ids = input_ids.detach().to("cpu", torch.int64)
        key = self._page_key(ids[: self.page_size])
        with self._lock:
            bucket = list(self._by_first.get(key, ()))
        best = None
        for entry in bucket:
            length = entry.length
            if length <= min_len or length > ids.numel():
                continue
            # only an entry whose KV begins where the caller's hit ends can extend it
            if entry.kv_start != min_len:
                continue
            if torch.equal(ids[:length], entry.tokens):
                if best is None or length > best.length:
                    best = entry
        if best is not None:
            best.last_used = time.monotonic()
            return best
        if self.ssd is None:
            return None
        try:
            entry = self.ssd.lookup(ids, min_len, self.signature)
        except Exception as exc:
            entry = None
        if entry is None or entry.nbytes > self.capacity_bytes:
            return None
        with self._lock:
            self._insert(entry, count=False)
            self._ssd_flushed.add(self._entry_key(entry))
            self.stats["ssd_loaded"] += 1
            self._trim(protected=entry)
        return entry

    def take(self, entry: HostEntry) -> None:
        with self._lock:
            self._remove(entry)
            self.stats["restored"] += 1
            self.stats["restored_bytes"] += entry.nbytes

    def summary(self) -> str:
        text = (
            f"host_kv entries={len(self._entries)} bytes={self._bytes / GIB:.2f}GiB "
            f"saved={self.stats['saved']} restored={self.stats['restored']} "
            f"skipped={self.stats['skipped']} failed={self.stats['failed']} "
            f"host_evicted={self.stats['host_evicted']}"
        )
        if self.ssd is not None:
            text += (
                f" | {self.ssd.summary()} | ssd_promoted={self.stats['ssd_saved']}"
                f" ssd_loaded={self.stats['ssd_loaded']}"
            )
        return text


def create_host_store(kv_pool, linear_pool, page_size: int) -> Optional[KVHostStore]:
    if not _env_bool("FREETOKEN_KV_SWAP", False):
        return None
    if kv_pool is None or not hasattr(kv_pool, "gather_kv"):
        return None
    return KVHostStore(kv_pool, linear_pool, page_size)
```

### 10.3 `freetoken/kv_swap/ssd.py`

```python
"""SSD tier for :class:`~freetoken.kv_swap.manager.KVHostStore`.

One file per captured prefix under ``<base>/<signature>/entries/`` plus an ``index.json``
listing their length, first-page bucket and content hash. A prefix that no longer fits in the
pinned host tier is written here instead of being dropped, so it survives a restart; ``lookup``
falls back to the index when the host tier misses.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from typing import Dict, List, Optional

import torch

GIB = 1 << 30


def _sha1(tokens: torch.Tensor) -> str:
    return hashlib.sha1(tokens.contiguous().numpy().tobytes()).hexdigest()


class SSDStore:
    def __init__(
        self,
        base_dir: str,
        capacity_bytes: int,
        page_size: int,
        min_tokens: int = 0,
    ) -> None:
        self.base_dir = os.path.abspath(base_dir)
        self.capacity_bytes = int(capacity_bytes)
        self.page_size = int(page_size)
        self.min_tokens = int(min_tokens)
        self._dir: Optional[str] = None
        self._index: Dict[str, dict] = {}
        self._by_first: Dict[object, List[str]] = {}
        self._bytes = 0
        self._lock = threading.RLock()
        self.stats = {"saved": 0, "loaded": 0, "dropped": 0, "evicted": 0, "failed": 0}

    @property
    def bytes_used(self) -> int:
        return self._bytes

    def _page_key(self, tokens: torch.Tensor):
        vals = tokens[: self.page_size].tolist()
        return vals[0] if self.page_size == 1 else tuple(vals)

    def _signature_dir(self, signature: str) -> str:
        return os.path.join(self.base_dir, hashlib.sha1(signature.encode()).hexdigest()[:16])

    def _ensure_dir(self, signature: str) -> None:
        if self._dir is not None:
            return
        root = self._signature_dir(signature)
        os.makedirs(os.path.join(root, "entries"), exist_ok=True)
        self._dir = root
        self._load_index()

    def _index_path(self) -> str:
        return os.path.join(self._dir, "index.json")

    def _load_index(self) -> None:
        path = self._index_path()
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, ValueError):
            return
        entries = raw.get("entries") or {}
        for name, meta in entries.items():
            full = os.path.join(self._dir, "entries", name)
            if not os.path.exists(full):
                continue
            self._index[full] = meta
            self._by_first.setdefault(tuple(meta["first"]), []).append(full)
            self._bytes += int(meta["bytes"])

    def _write_index(self) -> None:
        if self._dir is None:
            return
        entries = {}
        for full, meta in self._index.items():
            entries[os.path.basename(full)] = meta
        tmp = self._index_path() + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"entries": entries}, fh)
            os.replace(tmp, self._index_path())
        except OSError:
            self.stats["failed"] += 1

    def save(self, entry, signature: str, write_index: bool = True) -> bool:
        if entry.length < max(self.min_tokens, 1):
            return False
        with self._lock:
            self._ensure_dir(signature)
            digest = _sha1(entry.tokens)
            name = f"{digest[:20]}_{int(entry.kv_start)}_{entry.length}.pt"
            path = os.path.join(self._dir, "entries", name)
            if path in self._index and os.path.exists(path):
                self._index[path]["atime"] = time.time()
                self.stats["saved"] += 1
                if write_index:
                    self._write_index()
                return True
            payload = {
                "tokens": entry.tokens,
                "kv_k": entry.kv_k,
                "kv_v": entry.kv_v,
                "mamba": entry.mamba,
                "kv_start": int(entry.kv_start),
            }
            tmp = path + ".tmp"
            try:
                torch.save(payload, tmp)
                os.replace(tmp, path)
                size = os.path.getsize(path)
            except OSError:
                self.stats["failed"] += 1
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                return False
            first = list(entry.tokens[: self.page_size].tolist())
            meta = {
                "first": first,
                "length": entry.length,
                "kv_start": int(entry.kv_start),
                "sha1": digest,
                "bytes": int(size),
                "atime": time.time(),
            }
            if path not in self._index:
                self._bytes += int(size)
                self._by_first.setdefault(tuple(first), []).append(path)
            else:
                old = self._index[path]
                self._bytes += int(size) - int(old["bytes"])
            self._index[path] = meta
            self.stats["saved"] += 1
            self._evict()
            if write_index:
                self._write_index()
            return True

    def flush_index(self) -> None:
        with self._lock:
            self._write_index()

    def _evict(self) -> None:
        while self._bytes > self.capacity_bytes and self._index:
            path = min(self._index, key=lambda p: self._index[p]["atime"])
            self._drop(path)
            self.stats["evicted"] += 1

    def _drop(self, path: str) -> None:
        meta = self._index.pop(path, None)
        if meta is None:
            return
        self._bytes -= int(meta["bytes"])
        bucket = self._by_first.get(tuple(meta["first"]))
        if bucket is not None:
            try:
                bucket.remove(path)
            except ValueError:
                pass
            if not bucket:
                del self._by_first[tuple(meta["first"])]
        try:
            os.remove(path)
        except OSError:
            pass

    def lookup(self, ids: torch.Tensor, min_len: int, signature: str):
        from freetoken.kv_swap.manager import HostEntry

        with self._lock:
            self._ensure_dir(signature)
            key = tuple(ids[: self.page_size].tolist())
            bucket = list(self._by_first.get(key, ()))
        best_meta = None
        best_path = None
        for path in bucket:
            meta = self._index.get(path)
            if meta is None:
                continue
            length = int(meta["length"])
            if length <= min_len or length > ids.numel():
                continue
            # the entry's KV starts where the GPU hit ends, so only its own ancestors may be
            # served (an entry captured deeper down the tree cannot stand in for a shallow hit)
            if int(meta.get("kv_start", min_len)) != min_len:
                continue
            if best_meta is None or length > int(best_meta["length"]):
                if _sha1(ids[:length]) != meta["sha1"]:
                    continue
                best_meta, best_path = meta, path
        if best_meta is None:
            return None
        try:
            data = torch.load(best_path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, ValueError):
            self.stats["failed"] += 1
            with self._lock:
                self._drop(best_path)
            return None
        entry = HostEntry(
            tokens=data["tokens"].to(torch.int64).contiguous(),
            kv_k=data["kv_k"],
            kv_v=data["kv_v"],
            mamba=data.get("mamba"),
            kv_start=int(data["kv_start"]),
        )
        if entry.length != int(best_meta["length"]) or not torch.equal(
            ids[: entry.length], entry.tokens
        ):
            self.stats["dropped"] += 1
            return None
        with self._lock:
            best_meta["atime"] = time.time()
            self.stats["loaded"] += 1
            self._write_index()
        return entry

    def summary(self) -> str:
        return (
            f"ssd_kv entries={len(self._index)} bytes={self._bytes / GIB:.2f}GiB "
            f"saved={self.stats['saved']} loaded={self.stats['loaded']} "
            f"evicted={self.stats['evicted']} dropped={self.stats['dropped']} "
            f"failed={self.stats['failed']}"
        )
```

---

## 11. 修改文件逐处

### 11.1 `freetoken/scheduler/scheduler.py`

> ⚠ **0.2.0 起该文件是 `.pyd`，本节全部内容由 `freetoken/kv_swap/runtime.py` 的 monkeypatch 等价实现（见 16.3）。此处的代码片段仅作为逻辑参考，直接改源码已不生效。**

#### 11.1.1 顶部 import

```python
from freetoken.kv_swap import create_host_store
```

#### 11.1.2 创建 host store（`Scheduler.__init__` 里 `CacheManager(...)` 的构造）

```python
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type,
            linear_state_pool=self.engine.linear_state_pool,
            swa_pool=self.engine.kv_cache,
            sliding_window_size=next(
                (g.sliding_window for g in config.model_config.kv_cache_group_specs() if g.is_swa),
                None,
            ) or getattr(self.engine.kv_cache, "sliding_window_size", None),
            kv_pool=self.engine.kv_cache,
            host_store=create_host_store(
                self.engine.kv_cache, self.engine.linear_state_pool, config.page_size
            ),
        )
```

#### 11.1.3 待执行的重建（`__init__` 末尾附近）

```python
        # A received-but-not-yet-executed runtime cache rebuild (CacheRebuildBackendMsg),
        # run at the next idle safe point in overlap_loop. None when no rebuild is pending.
        self._pending_rebuild: CacheRebuildBackendMsg | None = None
```

#### 11.1.4 `run_when_idle`（完整替换原方法体）

```python
    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()
        import time as _time

        now = _time.monotonic()
        store = getattr(self.cache_manager, "host_store", None)
        if store is not None:
            if now - getattr(self, "_host_store_log_ts", 0.0) > 30.0:
                self._host_store_log_ts = now
                logger.info_rank0(store.summary())
            # SSD persistence is explicit: POST /v1/cache/dump (the stop script) or an orderly
            # shutdown dumps resident prefixes. The idle loop only reports the tiers.
            flag = getattr(store, "dump_flag_path", None)
            if flag and os.path.exists(flag):
                try:
                    result = self._dump_kv_to_ssd()
                    logger.info_rank0(
                        f"host KV dump: status={result['status']} captured={result['captured']} "
                        f"persisted={result['persisted']} | {result.get('summary', '')}"
                    )
                finally:
                    try:
                        os.remove(flag)
                    except OSError:
                        pass
            # Pre-stage evictable prefixes into the host tier while nobody is waiting. A later
            # request can then evict them for free (the host copy already exists) instead of
            # paying a synchronous D2H on the decode path -- or losing them outright, which is
            # what happens when CAPTURE_ON_EVICT is off (the default).
            preload = os.environ.get("FREETOKEN_KV_SWAP_PRELOAD_ON_IDLE", "1").strip().lower()
            if preload not in ("", "0", "false", "no", "off") and not getattr(
                self, "_idle_capture_disabled", False
            ):
                if now - getattr(self, "_idle_capture_ts", 0.0) > 1.0:
                    self._idle_capture_ts = now
                    try:
                        moved = store.capture_resident(
                            self.cache_manager.prefix_cache,
                            max_bytes=int(
                                os.environ.get("FREETOKEN_KV_SWAP_IDLE_BYTES", str(1 << 30))
                            ),
                        )
                        if moved:
                            logger.info_rank0(
                                f"host KV idle preload: captured={moved} | {store.summary()}"
                            )
                    except Exception as exc:  # noqa: BLE001 - background optimization
                        if "out of memory" in str(exc).lower():
                            self._idle_capture_disabled = True
                            logger.warning(
                                f"host KV idle preload disabled after CUDA OOM: {exc!r}"
                            )
                        else:
                            logger.warning(f"host KV idle preload failed: {exc!r}")
        moe = getattr(self.engine, "moe_offload_cache", None)
        prefetch = getattr(moe, "prefetch", None) if moe is not None else None
        if prefetch is not None and now - getattr(self, "_prefetch_log_ts", 0.0) > 30.0:
            self._prefetch_log_ts = now
            from freetoken.env import ENV
            from freetoken.moe.prefetch import format_per_layer, format_summary, summarize

            window = prefetch.take_window()
            stats = summarize(window)
            # ...（原有 MoE prefetch 统计打印，保持不变）
```

> 注：`run_when_idle` 的尾部 MoE prefetch 统计是上游/本地已有代码，本补丁只在其前面插入 host store 的两段。

#### 11.1.5 消费 `CacheRebuildBackendMsg`（`_process_one_msg` 分支）

```python
        elif isinstance(msg, CacheRebuildBackendMsg):
            if msg.dump_only:
                # Persist resident prefixes without resizing anything (stop-script path).
                if getattr(self.cache_manager, "host_store", None) is None:
                    self._reply_rebuild(msg.request_id, "unsupported", "KV swap is disabled")
                elif self.prefill_manager.runnable or self.decode_manager.runnable:
                    self._reply_rebuild(msg.request_id, "busy")
                else:
                    self._pending_rebuild = msg
            # v1 scope: only if_idle, single-rank, non-owned-KV. drain mode and TP rebuild
            # need the drain-gate / all-rank failure-agreement machinery (deferred), so we
            # reject them cleanly rather than ship hang-prone half-wired paths.
            elif not self.cache_manager.supports_runtime_rebuild:
                self._reply_rebuild(
                    msg.request_id, "unsupported", "this model's cache does not support runtime rebuild"
                )
            elif msg.mode != "if_idle":
                self._reply_rebuild(
                    msg.request_id, "unsupported", f"mode {msg.mode!r} unsupported (use if_idle)"
                )
            elif self.config.tp_info.size > 1:
                self._reply_rebuild(
                    # ...（原有 TP 拒绝逻辑）
                )
```

#### 11.1.6 `_execute_pending_rebuild` 的 dump 分支

```python
    def _execute_pending_rebuild(self) -> None:
        from freetoken.engine.engine import CacheRebuildRejected

        msg = self._pending_rebuild
        assert msg is not None
        self._pending_rebuild = None
        if msg.dump_only:
            result = self._dump_kv_to_ssd()
            logger.info_rank0(
                f"host KV dump: status={result['status']} captured={result['captured']} "
                f"persisted={result['persisted']} | {result.get('summary', '')}"
            )
            self._reply_rebuild(msg.request_id, result["status"])
            return
        requested = {
            "moe_cache_size": msg.moe_cache_size,
            "num_pages": msg.num_pages,
            "num_mamba_slots": msg.num_mamba_slots,
            "num_swa_pages": msg.num_swa_pages,
        }
        # ...（原有 rebuild 逻辑不变）
```

#### 11.1.7 新增 `_dump_kv_to_ssd`

```python
    def _dump_kv_to_ssd(self) -> dict:
        """Capture resident prefixes into the host tier, then flush it to SSD. Idle-only."""
        store = getattr(self.cache_manager, "host_store", None)
        if store is None:
            return {"status": "unsupported", "captured": 0, "persisted": 0}
        captured = 0
        persisted = 0
        if hasattr(store, "capture_resident"):
            captured = store.capture_resident(self.cache_manager.prefix_cache, max_bytes=None)
        if hasattr(store, "flush_ssd"):
            persisted = store.flush_ssd(max_bytes=1 << 62)
        return {
            "status": "ok",
            "captured": captured,
            "persisted": persisted,
            "summary": store.summary() if hasattr(store, "summary") else "",
        }
```

#### 11.1.8 新增 `_restore_linear_states`

在 `overlap_loop` / `normal_loop` 里，`_prepare_batch` 之后、forward 之前调用：

```python
    def _restore_linear_states(self, batch) -> None:
        """COW-restore a hybrid prefix hit's GDN snapshot into its freshly-allocated live slot
        (first chunk only). MUST run on the ENGINE stream so it is program-ordered after the
        prior batch's snapshot writes and before this forward reads the live slot."""
        pool = self.engine.linear_state_pool
        if pool is None or not batch.is_prefill:
            return
        for req in batch.reqs:
            if req.mamba_restore_src is not None:
                pool.copy_from(req.mamba_restore_src, req.linear_slot_idx)
                req.mamba_restore_src = None  # consumed: restore exactly once
```

调用点（两处，`overlap_loop` 和 `normal_loop`）：

```python
            forward_input = self._prepare_batch(...)
            self._restore_linear_states(forward_input.batch)
```

---

### 11.2 `freetoken/scheduler/cache.py`

#### 11.2.1 `CacheManager.__init__` 增加 `host_store`

```python
    def __init__(
        self, num_pages, page_size, page_table, type, *, linear_state_pool=None,
        swa_pool=None, sliding_window_size=None,
        kv_pool=None, host_store=None):
        # ...
        self.kv_pool = kv_pool
        self.host_store = host_store
```

#### 11.2.2 `_attach_evict_hook`（新增方法，`__init__` 末尾调用）

```python
    def _attach_evict_hook(self) -> None:
        # Capture-on-evict does a synchronous D2H inside the eviction path and can stall
        # decode on long contexts; the default is to leave it off and dump at stop time
        # (KVHostStore.capture_resident). Opt in with FREETOKEN_KV_SWAP_CAPTURE_ON_EVICT=1.
        import os

        enabled = os.environ.get("FREETOKEN_KV_SWAP_CAPTURE_ON_EVICT", "0").strip().lower()
        if enabled in ("", "0", "false", "no", "off"):
            return
        if self.host_store is not None and hasattr(self.prefix_cache, "evict_hook"):
            # page_size, not the 512-token host-store default: the dump/restore chain must stay
            # contiguous from token 0, so a skipped 64..511-token node would break it.
            page_size = self.page_size
            self.prefix_cache.evict_hook = lambda node: self.host_store.capture_node(
                node, min_tokens=page_size
            )
```

#### 11.2.3 `match_req` 透传 mamba

```python
    def match_req(self, req: PendingReq) -> MatchResult:
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        ids = req.input_ids[: input_len - 1]
        if self.is_swa:
            from freetoken.kvcache.swa_radix_cache import SWACacheHandle
            m = self.prefix_cache.match_prefix(ids)
            return MatchResult(SWACacheHandle(m.cached_len, m.node, m.kv_indices))
        if self.is_hybrid:
            from freetoken.kvcache.hybrid_radix_cache import HybridCacheHandle
            m = self.prefix_cache.match_prefix(ids)
            return MatchResult(
                HybridCacheHandle(m.cached_len, m.node, m.kv_indices), mamba_value=m.mamba_value)
        return self.prefix_cache.match_prefix(ids)
```

#### 11.2.4 新增 `attempt_host_restore`

```python
    def attempt_host_restore(self, input_ids: torch.Tensor, handle: BaseCacheHandle) -> BaseCacheHandle:
        """Extend a GPU radix hit with a longer prefix that was offloaded to host memory.

        Requires the GPU match to end exactly where the host entry's KV begins (the entry's
        parent chain). Allocates fresh pages, copies K/V + the GDN snapshot back, re-inserts
        the prefix into the radix tree, and migrates the lock from ``handle`` to the new one.
        """
        store = self.host_store
        if store is None or self.kv_pool is None or self.is_swa:
            return handle
        cached_len = handle.cached_len
        entry = store.lookup(input_ids, min_len=cached_len)
        if entry is None or entry.kv_start != cached_len or entry.length <= cached_len:
            return handle
        add = entry.length - cached_len
        if add <= 0 or add % self.page_size != 0:
            return handle
        slots = self._page_to_token(self._allocate(add // self.page_size))
        mamba_slot = None
        try:
            self.kv_pool.scatter_kv(slots, entry.kv_k, entry.kv_v)
            if self.device.type == "cuda":
                torch.cuda.current_stream(self.device).synchronize()
            if self.is_hybrid:
                pool = self.linear_state_pool
                if pool.num_free_slots < 1:
                    self.ensure_mamba_slots(1)
                if pool.num_free_slots < 1:
                    raise RuntimeError("LinearStatePool exhausted while restoring host KV")
                mamba_slot = pool.alloc(1)[0]
                if entry.mamba is not None:
                    pool.restore_slot(mamba_slot, entry.mamba)
                else:
                    pool.reset(mamba_slot)
            ids = input_ids[: entry.length]
            matched = handle.get_matched_indices()[:cached_len].to(slots.device)
            full_slots = torch.cat([matched, slots])
            if self.is_hybrid:
                from freetoken.kvcache.hybrid_radix_cache import HybridCacheHandle
                _, existed = self.prefix_cache.insert(ids, full_slots, mamba_slot)
                if existed:
                    raise RuntimeError("host KV restore collided with an existing snapshot")
                m = self.prefix_cache.match_prefix(ids)
                new_handle = HybridCacheHandle(m.cached_len, m.node, m.kv_indices)
            else:
                new_handle = self.prefix_cache.insert_prefix(ids, full_slots).handle
        except Exception:
            if mamba_slot is not None:
                self.linear_state_pool.free([mamba_slot])
            self._free(slots)
            raise
        self.unlock(handle)
        self.lock(new_handle)
        store.take(entry)
        logger.info(
            f"host KV swap-in: restored {add} tokens (prefix {entry.length}), "
            f"{store.summary()}"
        )
        return new_handle
```

> `lock` / `unlock` 本身是上游已有方法，本补丁不改；但**调用契约**由 `prefill._try_allocate_one` 保证。

---

### 11.3 `freetoken/scheduler/prefill.py`

#### 11.3.1 `_try_allocate_one`（完整替换）

```python
    def _try_allocate_one(self, req: PendingReq):
        if self.table_manager.available_size == 0:
            return None

        # TODO: consider host cache match case
        mr = self.cache_manager.match_req(req)
        handle = mr.cuda_handle
        # Hold a reference before restoring: attempt_host_restore migrates this lock from the
        # matched handle to the longer one (unlock(old) + lock(new)), so the caller must own it.
        self.cache_manager.lock(handle)
        if self.cache_manager.host_store is not None:
            # The host/SSD tier stores one entry per radix node (each with its own kv_start),
            # so keep extending until no entry continues the prefix we already hold.
            try:
                while True:
                    extended = self.cache_manager.attempt_host_restore(req.input_ids, handle)
                    if extended is handle:
                        break
                    handle = extended
                    # Stop extending once the pool can no longer fit the request itself:
                    # a half-restored prefix would reserve pages and starve prefill forever.
                    need = self._kv_reservation_size(
                        req.input_len + req.output_len, handle.cached_len
                    )
                    if need + self.reserved_size > self.cache_manager.available_size:
                        break
            except Exception as exc:  # noqa: BLE001 - host restore is an optimization, never fatal
                logger.warning(f"host KV restore failed, falling back to GPU prefix: {exc!r}")
        cached_len = handle.cached_len
        # TODO: better estimate policy
        extend_len = req.input_len - cached_len
        estimated_size = self._kv_reservation_size(
            req.input_len + req.output_len, cached_len
        )

        if estimated_size + self.reserved_size > self.cache_manager.available_size:
            self.cache_manager.unlock(handle)
            return None

        # Second currency (hybrid GDN): reserve 1 live + 2 ping-pong state slots; evict tree
        # snapshots if the pool is short, fail admission if still short (mirrors the KV gate).
        if self.cache_manager.is_hybrid:
            pool = self.cache_manager.linear_state_pool
            if pool.num_free_slots < 3:
                self.cache_manager.ensure_mamba_slots(3)
            if pool.num_free_slots < 3:
                return self.cache_manager.unlock(handle)

        # Third currency (SWA): refuse admission unless the swa pool can seat this request's first
        # chunk / one window (the per-chunk charge is in _add_one_req; the reclaim -- radix
        # evict_swa -- happens in allocate_paged, so no ensure here; swa_available_size already
        # folds the evictable tree). For naive (no tree) this can only refuse, which is correct.
        if self.cache_manager.swa_paged:
            ps = self.cache_manager.page_size
            # swa is charged per WHOLE page (allocate_paged -> alloc_swa), so the seat check is
            # in page units too; identical at page_size==1.
            need_swa = div_ceil(
                min(max(extend_len, 1), self.cache_manager.sliding_window_size) + 1, ps
            ) * ps
            if self.cache_manager.swa_available_size - self.reserved_swa < need_swa:
                return self.cache_manager.unlock(handle)

        table_idx = self.table_manager.allocate()
        if cached_len > 0:  # NOTE: set the cached part
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]
            device_ids.copy_(_maybe_pinned(req.input_ids[:cached_len]), non_blocking=True)
            # Write the matched indices into the TAIL of the page_entry: a cache may return
            # fewer matched indices than cached_len, in which case only the trailing n slots are
            # known-live. Today both the generic radix and the SWA radix match a prefix whose
            # full-loc row is entirely live (n == cached_len), so the tail IS the whole prefix.
            # (DSV4 reads this table too: its pool's full_loc_map is attached to it.)
            matched = handle.get_matched_indices()
            n = int(matched.numel())
            self.table_manager.page_table[table_idx][cached_len - n : cached_len].copy_(matched)

        linear_slot_idx = ping_pong = None
        if self.cache_manager.is_hybrid:
            pool = self.cache_manager.linear_state_pool
            linear_slot_idx = pool.alloc(1)[0]
            ping_pong = tuple(pool.alloc(2))

        # A host restore lengthens the prefix past the original match, so the GDN snapshot to
        # COW into the live slot must come from the FINAL node. Using mr.mamba_value here would
        # pair a long KV prefix with the shorter prefix's recurrent state (or with a stale slot
        # when the GPU match was empty) -- the model then "remembers" the wrong prefix.
        mamba_value = getattr(handle.node, "mamba_value", None)
        if mamba_value is None:
            mamba_value = mr.mamba_value

        return handle, table_idx, linear_slot_idx, ping_pong, mamba_value
```

#### 11.3.2 `_add_one_req` 签名与赋值

```python
    def _add_one_req(
        self,
        pending_req: PendingReq,
        cache_handle: BaseCacheHandle,
        table_idx: int,
        cached_len: int,
        linear_slot_idx: int | None = None,
        ping_pong: tuple | None = None,
        next_track_idx: int = 0,
        restore_src: int | None = None,
        swa_evicted_seqlen: int = 0,
    ) -> Req | None:
        # ...
        req.linear_slot_idx = linear_slot_idx
        req.mamba_ping_pong = ping_pong
        req.mamba_next_track_idx = next_track_idx
        req.mamba_restore_src = restore_src
        req.swa_evicted_seqlen = swa_evicted_seqlen
        return req
```

#### 11.3.3 `try_add_one` 解包 5 元组

```python
        if resource := self._try_allocate_one(pending_req):
            cache_handle, table_idx, linear_slot_idx, ping_pong, restore_src = resource
            req = self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cache_handle.cached_len,
                linear_slot_idx=linear_slot_idx,
                ping_pong=ping_pong,
                next_track_idx=0,
                restore_src=restore_src,
            )
            if req is None:
                # no aligned chunk this pass: undo the admission (a continuation keeps its
                # resources -- they belong to the prior chunk's Req)
                self.cache_manager.unlock(cache_handle)
                self.table_manager.free(table_idx)
                if linear_slot_idx is not None:
                    self.cache_manager.linear_state_pool.free([linear_slot_idx, *ping_pong])
            return req
```

---

### 11.4 `freetoken/message/backend.py`

> ⚠ **0.2.0 起已删除 `dump_only` 字段，本补丁不再需要改这个文件**（落盘统一走 `kv_persist/.dump_request` 标记文件）。以下为历史参考。

```python
@dataclass
class CacheRebuildBackendMsg(BaseBackendMsg):
    # tokenizer worker -> scheduler: request a runtime KV/MoE/GDN cache resize.
    request_id: str
    moe_cache_size: int | None = None
    num_pages: int | None = None
    num_mamba_slots: int | None = None
    num_swa_pages: int | None = None
    mode: str = "if_idle"  # only "if_idle" is supported; "drain" is deferred (rejected)
    dump_only: bool = False  # persist resident prefixes to SSD without resizing anything
```

---

### 11.5 `freetoken/message/tokenizer.py`

> ⚠ **0.2.0 起已删除 `dump_only` 字段，无需再改。**

```python
@dataclass
class CacheRebuildMsg(BaseTokenizerMsg):
    # api server -> tokenizer worker (pure passthrough to CacheRebuildBackendMsg).
    request_id: str
    moe_cache_size: int | None = None
    num_pages: int | None = None
    num_mamba_slots: int | None = None
    num_swa_pages: int | None = None
    mode: str = "if_idle"
    dump_only: bool = False
```

---

### 11.6 `freetoken/tokenizer/server.py`

> ⚠ **0.2.0 起无需改（`dump_only` 已删除）。历史参考：**

```python
            # Cache-rebuild control messages are pure passthrough (no tokenization):
            # CacheRebuildMsg (api -> scheduler) and CacheRebuildResultMsg (scheduler -> api).
            for m in pending_msg:
                if isinstance(m, CacheRebuildMsg):
                    send_backend.put(
                        CacheRebuildBackendMsg(
                            request_id=m.request_id,
                            moe_cache_size=m.moe_cache_size,
                            num_pages=m.num_pages,
                            num_mamba_slots=m.num_mamba_slots,
                            num_swa_pages=m.num_swa_pages,
                            mode=m.mode,
                            dump_only=m.dump_only,
                        )
                    )
                elif isinstance(m, CacheRebuildResultMsg):
                    send_frontend.put(
                        CacheRebuildReply(
                            request_id=m.request_id,
                            status=m.status,
                            moe_cache_size=m.moe_cache_size,
                            num_pages=m.num_pages,
                            mamba_slots=m.mamba_slots,
                            num_swa_pages=m.num_swa_pages,
                            error=m.error,
                        )
                    )
```

---

### 11.7 `freetoken/server/api_server.py`

> ⚠ **0.2.0 起该文件是 `.pyd`；且本补丁从不依赖 HTTP dump 端点（统一用标记文件），本节已整体废弃。**

#### 11.7.1 `dispatch_rebuild` 增加 `dump_only`

```python
async def dispatch_rebuild(
    state: FrontendManager,
    *,
    moe_cache_size: int | None,
    num_pages: int | None,
    num_mamba_slots: int | None = None,
    num_swa_pages: int | None = None,
    mode: str = "if_idle",
    timeout: float = 300.0,
    dump_only: bool = False,
) -> Dict[str, Any]:
    # ...
    await state.send_one(
        CacheRebuildMsg(
            request_id=request_id,
            moe_cache_size=moe_cache_size,
            num_pages=num_pages,
            num_mamba_slots=num_mamba_slots,
            num_swa_pages=num_swa_pages,
            mode=mode,
            dump_only=dump_only,
        )
    )
```

#### 11.7.2 新增 `POST /v1/cache/dump`

```python
@app.post("/v1/cache/dump")
async def cache_dump():
    """Persist every prefix still resident in GPU KV to the host/SSD tiers, then return.

    Used by the stop script: Windows cannot run Python cleanup when a process is killed, so the
    dump must be requested over HTTP before the workers are torn down. Does not resize or evict
    anything -- the GPU prefix cache keeps serving."""
    state = get_global_state()
    if state.maintenance_state in ("loading", "rebuilding", "stopping"):
        return JSONResponse({"status": state.maintenance_state}, status_code=503)
    result = await dispatch_rebuild(
        state,
        moe_cache_size=None,
        num_pages=None,
        dump_only=True,
        timeout=600.0,
    )
    if result["status"] == "timeout":
        return JSONResponse(result, status_code=504)
    return JSONResponse(result, status_code=200 if result["status"] == "ok" else 503)
```

> 注意：`ft.exe` 主进程可能不加载隔离目录的 `api_server.py`（实测 `POST /v1/cache/dump` 返回 404），所以停服脚本走**标记文件**而不是这个 HTTP 端点。端点保留给 worker 直接加载 `api_server.py` 的场景。

---

### 11.8 `freetoken/cli.py`

```python
def _run_serve(argv: list[str]) -> int:
    from freetoken.server import launch_server

    try:
        launch_server(argv=argv, prog="ft serve")
    except KeyboardInterrupt:
        # Windows CTRL_C_EVENT: uvicorn re-raises the captured signal after the app has
        # already shut down cleanly, so this is an expected exit, not an error.
        return 130
    return 0
```

### 11.9 `freetoken/kvcache/qsa_pool.py`

只在 `QSAKVCache` 上新增两个方法（其他后端没有这两个方法，调用方用 `hasattr` 判断）：

```python
    def gather_index_tiers(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        idx = slots.to(device=self._device, dtype=torch.long)
        rows = idx[:: self._index_ratio] // self._index_ratio
        cmp_rows = self._cmp_k_buffer.index_select(1, rows).clone()
        rope_rows = (
            self._rope_positions.index_select(0, idx).clone()
            if self._rope_positions is not None
            else None
        )
        return cmp_rows, rope_rows

    def scatter_index_tiers(self, slots, cmp_rows, rope_rows) -> None:
        idx = slots.to(device=self._device, dtype=torch.long)
        if cmp_rows is not None:
            rows = idx[:: self._index_ratio] // self._index_ratio
            if cmp_rows.shape[1] != rows.numel():
                raise ValueError("index tier size mismatch")
            self._cmp_k_buffer.index_copy_(
                1, rows, cmp_rows.to(device=self._device, dtype=self._index_dtype)
            )
        if rope_rows is not None and self._rope_positions is not None:
            self._rope_positions.index_copy_(
                0, idx, rope_rows.to(device=self._device, dtype=self._rope_positions.dtype)
            )
```

---

## 12. 脚本全文

### 12.1 `scripts/start_server_isolated.ps1`

```powershell
param(
    [int]$NumTokens = 98304,
    [int]$Port = 1919,
    [string]$LogDir = "C:\AI_Models\new_log",
    [string]$KvPersistDir = "C:\Users\X303\Documents\FreetokenP\kv_persist",
    [string]$Gpu = "GPU-54e316a5-5b97-c3dd-ebbc-d54e2f80cf22",
    [switch]$Detach,
    [switch]$NoMoeOverlap,
    [switch]$NoPrefetch,
    [switch]$PrefetchMeasure,
    [switch]$NoPersist,
    [switch]$NoSwap
)

$ISOLATED = "C:\Users\X303\Documents\FreetokenP"
$FT       = "C:\Users\X303\AppData\Local\FreeToken\venv\Scripts\ft.exe"
$MODEL    = "C:\AI_Models\Qwen3.8-Flash-Next-NVFP4"

if (-not (Test-Path -LiteralPath $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}
if (-not (Test-Path -LiteralPath $KvPersistDir)) {
    New-Item -ItemType Directory -Path $KvPersistDir -Force | Out-Null
}

$stamp   = Get-Date -Format "yyyyMMdd-HHmmss"
$logFile = if ($Detach) { Join-Path $LogDir "serve_isolated.log" } else { Join-Path $LogDir "freetoken-serve-$stamp.log" }
$errFile = "$logFile.err"

$env:PYTHONPATH = "$ISOLATED;$env:PYTHONPATH"
$env:TQDM_DISABLE = "1"
$env:FREETOKEN_KV_SWAP = if ($NoSwap) { "0" } else { "1" }
$env:FREETOKEN_KV_SWAP_HOST_BYTES = "17179869184"
$env:FREETOKEN_KV_SWAP_MIN_TOKENS = "256"
$env:FREETOKEN_KV_SWAP_PIN = "0"
$env:FREETOKEN_KV_SWAP_COPY_CHUNK = "2048"
$env:FREETOKEN_KV_PERSIST = if ($NoPersist -or $NoSwap) { "0" } else { "1" }
$env:FREETOKEN_KV_PERSIST_DIR = $KvPersistDir
$env:FREETOKEN_KV_PERSIST_BYTES = "68719476736"
$env:FREETOKEN_MOE_COPY_OVERLAP = if ($NoMoeOverlap) { "0" } else { "1" }
# measure costs ~25% decode throughput (it re-runs the gate on a side stream every step);
# keep it off by default and enable only for diagnostics.
$env:FREETOKEN_MOE_PREFETCH = if ($PrefetchMeasure -and -not $NoPrefetch) { "measure" } else { "off" }

$ftArgs = @(
    "serve",
    "--model", $MODEL,
    "--port", "$Port",
    "--moe-strategy", "offload",
    "--image-max-tokens", "512",
    "--memory-ratio", "0.95",
    "--num-tokens", "$NumTokens",
    "--moe-cache-size", "512",
    "--moe-cpu-layers", "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,34,35,36,37,38,39,40,41,42,43,44,45,46,47",
    "--max-running-requests", "2",
    "--cuda-graph-max-bs", "1",
    "--moe-cpu-threads", "16",
    "--gpu", $Gpu,
    "--host", "127.0.0.1"
)

if ($Detach) {
    $proc = Start-Process -FilePath $FT -ArgumentList $ftArgs `
        -WorkingDirectory $ISOLATED -RedirectStandardOutput $logFile `
        -RedirectStandardError $errFile -PassThru -WindowStyle Hidden
    try { $proc.PriorityClass = "High" } catch { }
    Write-Host "Started detached FreeToken, PID=$($proc.Id), port=$Port, num_tokens=$NumTokens"
    Write-Host "stdout: $logFile"
    Write-Host "stderr: $errFile"
    Write-Host "kv persist: $KvPersistDir"
}
else {
    try {
        [System.Diagnostics.Process]::GetCurrentProcess().PriorityClass = [System.Diagnostics.ProcessPriorityClass]::High
        Write-Host " Priority: High" -ForegroundColor Green
    }
    catch { }

    Write-Host ""
    Write-Host "==========================================================" -ForegroundColor DarkCyan
    Write-Host " FreeToken Qwen3.8-Flash-Next-NVFP4 (isolated, port $Port)" -ForegroundColor Cyan
    Write-Host " Log file: $logFile" -ForegroundColor Cyan
    Write-Host " KV persist dir: $KvPersistDir" -ForegroundColor Cyan
    Write-Host " Stop server with Ctrl+C" -ForegroundColor Cyan
    Write-Host "==========================================================" -ForegroundColor DarkCyan
    Write-Host ""

    & $FT @ftArgs 2>&1 | ForEach-Object {
        if ($_ -is [System.Management.Automation.ErrorRecord]) {
            $line = $_.Exception.Message
        }
        else {
            $line = "$_"
        }
        $line | Out-Host
        Add-Content -LiteralPath $logFile -Value $line -Encoding UTF8
    }
}
```

### 12.2 `scripts/stop_server_isolated.ps1`

```powershell
param(
    [int]$Port = 1919,
    [int]$TimeoutSec = 8,
    [string]$KvPersistDir = "C:\Users\X303\Documents\FreetokenP\kv_persist",
    [switch]$Force
)

# Stop the isolated FreeToken instance on $Port.
#
# Default: try CTRL_C_EVENT first (orderly uvicorn shutdown), then fall back to
# terminating the process tree. -Force skips the CTRL_C attempt.
#
# Host KV is flushed to SSD at every idle transition, so killing the workers does not
# lose prefixes captured by completed requests.
#
# Orphan sweep: Windows spawn workers whose parent died keep the whole model in VRAM.
# After the listener is gone, any `python -c spawn_main` process whose parent_pid is no
# longer alive belongs to a dead server and is terminated too.

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class ConsoleCtl {
    [DllImport("kernel32.dll", SetLastError = true)] public static extern bool FreeConsole();
    [DllImport("kernel32.dll", SetLastError = true)] public static extern bool AttachConsole(uint dwProcessId);
    [DllImport("kernel32.dll", SetLastError = true)] public static extern bool GenerateConsoleCtrlEvent(uint dwCtrlEvent, uint dwProcessGroupId);
    [DllImport("kernel32.dll", SetLastError = true)] public static extern bool SetConsoleCtrlHandler(IntPtr handler, bool add);
}
"@ -ErrorAction SilentlyContinue

function Get-ProcessTree {
    param([int]$RootPid)
    $all = Get-CimInstance Win32_Process
    $ids = New-Object 'System.Collections.Generic.HashSet[int]'
    [void]$ids.Add($RootPid)
    $frontier = @($RootPid)
    while ($frontier.Count -gt 0) {
        $next = @()
        foreach ($p in $frontier) {
            foreach ($c in $all) {
                if ([int]$c.ParentProcessId -eq $p -and $ids.Add([int]$c.ProcessId)) {
                    $next += [int]$c.ProcessId
                }
            }
        }
        $frontier = $next
    }
    return @($ids)
}

function Stop-Tree {
    param([int]$RootPid)
    $tree = Get-ProcessTree -RootPid $RootPid
    foreach ($procId in ($tree | Sort-Object -Descending)) {
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    }
    return $tree.Count
}

function Stop-OrphanWorkers {
    $all = Get-CimInstance Win32_Process
    $alive = @{}
    foreach ($p in $all) { $alive[[int]$p.ProcessId] = $true }
    $orphans = @($all | Where-Object {
        $_.Name -eq 'python.exe' -and
        $_.CommandLine -match 'spawn_main' -and
        -not $alive.ContainsKey([int]$_.ParentProcessId)
    })
    foreach ($o in $orphans) {
        Stop-Process -Id $o.ProcessId -Force -ErrorAction SilentlyContinue
    }
    return $orphans.Count
}

function Wait-PortFree {
    param([int]$Port, [int]$Seconds)
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        if (-not (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)) {
            return $true
        }
        Start-Sleep -Seconds 1
    }
    return $false
}

$conn = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
if (-not $conn) {
    Write-Output "Nothing is listening on port $Port."
}
else {
    $root = [int]($conn.OwningProcess | Select-Object -First 1)

    if (-not $Force) {
        # The API-server main process can be a frozen bundle that never sees our patched
        # routes, so trigger the dump through a flag file the (patched) scheduler consumes on
        # its next idle transition. A 1-token request produces that transition.
        $flag = Join-Path $KvPersistDir ".dump_request"
        New-Item -ItemType Directory -Path $KvPersistDir -Force | Out-Null
        Set-Content -LiteralPath $flag -Value "dump" -Encoding ASCII
        Write-Output "Requesting KV dump (GPU prefixes -> host -> SSD)..."
        try {
            $body = '{"model":"Qwen3.8-Flash-Next-NVFP4","messages":[{"role":"user","content":"ping"}],"max_tokens":1}'
            Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post `
                -ContentType "application/json" -Body $body -TimeoutSec 120 | Out-Null
        }
        catch {
            Write-Output ("Dump trigger request failed: " + $_.Exception.Message)
        }
        $deadline = (Get-Date).AddSeconds(300)
        while ((Test-Path -LiteralPath $flag) -and ((Get-Date) -lt $deadline)) {
            Start-Sleep -Seconds 2
        }
        if (Test-Path -LiteralPath $flag) {
            Write-Output "KV dump flag was not consumed within 300s; proceeding with stop."
        }
        else {
            Write-Output "KV dump completed."
        }
    }

    if ($Force) {
        $n = Stop-Tree -RootPid $root
        Write-Output "Force-stopped $n process(es) in the tree of PID $root."
    }
    else {
        Write-Output "Sending CTRL_C_EVENT to PID $root (port $Port)..."
        [void][ConsoleCtl]::FreeConsole()
        if ([ConsoleCtl]::AttachConsole([uint32]$root)) {
            [void][ConsoleCtl]::SetConsoleCtrlHandler([IntPtr]::Zero, $true)
            [void][ConsoleCtl]::GenerateConsoleCtrlEvent(0, 0)
            [void][ConsoleCtl]::FreeConsole()
        }
        else {
            Write-Output "Could not attach to the target console; falling back to terminate."
        }

        if (Wait-PortFree -Port $Port -Seconds $TimeoutSec) {
            Write-Output "Port $Port released after graceful shutdown."
        }
        else {
            Write-Output "Graceful stop timed out; terminating the process tree."
            [void](Stop-Tree -RootPid $root)
        }
    }
}

    $totalReaped = 0
    for ($i = 0; $i -lt 3; $i++) {
        if ($i -gt 0) { Start-Sleep -Seconds 2 }
        $totalReaped += Stop-OrphanWorkers
    }
    if ($totalReaped -gt 0) {
        Write-Output "Reaped $totalReaped orphaned backend worker(s) (parent already gone)."
    }

Start-Sleep -Seconds 2
if (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) {
    Write-Output "Port $Port is still in use."
    exit 1
}
Write-Output "Port $Port released."
```

---

## 13. 补丁应用检查表

打完后逐项确认：

- [ ] `python -m py_compile` 通过：`scheduler.py`、`cache.py`、`prefill.py`、`kv_swap/*.py`、`cli.py`；
- [ ] `FREETOKEN_KV_SWAP=1` 时启动日志出现 `host_kv entries=0 ...`（30s summary）；
- [ ] `POST /v1/cache/dump` 或停服 flag 能产生 `host KV dump: status=ok`；
- [ ] `kv_persist/<sig>/index.json` 的 `kv_start` 链从 0 连续；
- [ ] 重启后同 prompt 出现 `host KV swap-in: restored N tokens`；
- [ ] `check_integrity` 无 `assert cur.ref_count >= 0`；
- [ ] idle 时出现 `host KV idle preload: captured=N`，且 `failed=0`；
- [ ] 停服后无 `KeyboardInterrupt` traceback；
- [ ] 共享前缀 A/B 交替答案正确。

---

## 14. 参考日志片段

```
host_kv entries=0 bytes=0.00GiB saved=0 restored=0 skipped=0 failed=0 host_evicted=0 | ssd_kv entries=0 ...
host KV idle preload: captured=1 | host_kv entries=1 bytes=0.32GiB saved=1 restored=0 skipped=0 failed=0 ...
host KV swap-in: restored 9152 tokens (prefix 9152), host_kv entries=0 bytes=0.00GiB ... ssd_loaded=1
host KV dump: status=ok captured=9 persisted=9 | host_kv entries=9 bytes=2.91GiB ... ssd_promoted=9
```

## 15. QSA index tiers 持久化（阶段八）

### 15.1 为什么 K/V 之外还有状态

`qsa_sparse` 后端（`Qwen3.8-Flash-Next`）的注意力不是逐 token 的全量 attention：

- **压缩 index key**：`_cmp_k_buffer`，每 `index_ratio`（本模型 4）个 token 压成一行，供 indexer 打分选块。行号 = 组首 token 的 slot `// index_ratio`，因此它是 K/V 页的 1/4 shadow（`page_size % index_ratio == 0` 保证一组不跨页）；
- **mrope 位置**：`_rope_positions`，每 token 三轴（t/h/w）int32。压缩组在组首 token 的 rope 位置上旋转，无法从逻辑位置反推。

两者都是 per-token 状态，但不在 `_k_buffer/_v_buffer` 里，`gather_kv/scatter_kv` 不会搬。

### 15.2 dump 时怎么取

`KVHostStore.capture_node` 按 `copy_chunk` 分块，与 `gather_kv` 并行调用：

| 数据 | 行选择 | shape |
|---|---|---|
| `cmp_index` | `slots[::ratio] // ratio` | `[index_layers, ceil(N/ratio), index_head_dim]` |
| `rope_pos` | `slots` | `[N, 3]` int32 |

两者转 CPU 后存入 `HostEntry.cmp_index / rope_pos`，随 K/V 一起进 host tier 或 spill 到 SSD；`nbytes` 计入容量计算。

### 15.3 restore 时怎么写

`attempt_host_restore` 在 `scatter_kv` 之后：

```python
cmp_src = entry.cmp_index[:, skip // ratio:] if entry.cmp_index is not None else None
rope_src = entry.rope_pos[skip:] if entry.rope_pos is not None else None
if hasattr(self.kv_pool, "scatter_index_tiers") and (cmp_src is not None or rope_src is not None):
    self.kv_pool.scatter_index_tiers(slots, cmp_src, rope_src)
```

`skip` 是 `page_size` 的整数倍，`page_size % ratio == 0`，因此 `skip // ratio` 与行数 `ceil((N - skip) / ratio)` 严格对齐；`scatter_index_tiers` 会校验行数，不匹配就抛错（宁可全量 prefill 也不写错）。

### 15.4 旧条目兼容

`HostEntry` 两个新字段默认 `None`；`SSDStore.save` 写入，`lookup` 用 `data.get(...)` 读取。升级前 dump 的文件没有这两个键，读成 `None` → 跳过 index 搬运 → **不崩，但长上下文仍错乱**。处理方式：

- 让长会话全量 prefill 一次，重新 dump；或
- 删除旧 `entries/*.pt`（本机已把旧目录移到 `kv_persist.pre-index.bak`）。

### 15.5 验证

见 6.4。本机 22272-token 前缀实测：

```
max|dK|=0 max|dV|=0 max|dCmp|=0 max|dRope|=0, mamba max|d|=0
全量 prefill 34.94 s → restore 8.49 s
```

### 15.6 涉及的四个文件

| 文件 | 改动 |
|---|---|
| `kvcache/qsa_pool.py` | 新增 `gather_index_tiers` / `scatter_index_tiers` |
| `kv_swap/manager.py` | `HostEntry` 加 `cmp_index` / `rope_pos`；`nbytes`；`capture_node` 收集 |
| `kv_swap/ssd.py` | payload 加两个键；`lookup` 兼容读取 |
| `scheduler/cache.py` | restore 写回；VERIFY 增加 dCmp / dRope 输出 |

---

## 16. 阶段九：FreeToken 0.2.0 适配实录（2026-10-22）

### 16.1 版本与同步方式

| 项 | 升级前 | 升级后 |
|---|---|---|
| FreeToken | 0.1.3+gfa0e9537d | **0.2.0+g425dc7bf1** |
| torch | 2.11.0+cu130 | **2.14.1+cu130** |
| Python | 3.12.14 | 3.12.14 |
| kernel 包 | — | `freetoken_kernel_cache-0.2.0+cu130.g425dc7bf1` |

同步命令（把官方 site-packages 全量覆盖到隔离目录，保留 `kv_swap/`）：

```powershell
robocopy "C:\Users\X303\AppData\Local\FreeToken\venv\Lib\site-packages\freetoken" `
         "C:\Users\X303\Documents\FreetokenP\freetoken" `
         /MIR /XD kv_swap __pycache__ /XF *.pyc
```

要点：
- 升级前完整备份到 `freetoken.pre-0.2.0.bak`（回滚用）；
- `/MIR` 会删掉 repo 里旧的 `scheduler/scheduler.py`、`server/api_server.py` —— **必须删**，否则 `.py` 会与 `.pyd` 抢导入（同名时 `.pyd` 优先，但残留源码会误导排障）；
- 官方包中**不含** `kv_swap/`，补丁目录需保留。

### 16.2 0.2.0 的接口变化（导致补丁必须重写的地方）

| 位置 | 变化 | 影响 |
|---|---|---|
| `scheduler/scheduler.py` | 变成 `scheduler.pyd` | 不能再改源码 → 改 `kv_swap/runtime.py` monkeypatch |
| `server/api_server.py` | 变成 `api_server.pyd` | dump 端点补丁废弃（本来就不依赖） |
| `CacheManager.__init__` | 去掉 `kv_pool` 参数 | 需在 `Scheduler.__init__` 之后注入 `self.cache_manager.kv_pool` |
| `CacheManager` | 新增 `prefix_replay_tokens`、`match_req(req, max_len=)` | 补丁需保持这些属性可用；QSA 模型 `prefix_replay_tokens == 0` |
| `prefill._try_allocate_one` | 改为「估算 → `lock(handle)` → 复核」 | host restore 必须插在 **lock + 复核之后**，并在恢复后重算 `cached_len` / `extend_len` / `estimated_size` |
| `kvcache/base.py`、`mha_pool.py` | 删除 `gather_kv` / `scatter_kv`，只留 `k_cache(layer_id)` / `v_cache(layer_id)` / `store_kv(k,v,out_loc,layer_id)` / `unit_bytes()` | **必须重建** `gather_kv` / `scatter_kv`，否则 dump/restore/capture 全部失效 |
| `message/backend.py`、`message/tokenizer.py`、`tokenizer/server.py` | 删除 `dump_only` 字段 | 这三处补丁废弃；落盘只走 `.dump_request` 标记文件 |
| `init_logger()` | 返回的 logger 没有 `error_rank0` | `check_integrity` 的报错改用 `logger.warning` |
| `ft serve` 参数 | `--moe-backend` 改名 `--moe-strategy`（旧名保留为别名） | 启动脚本无需改 |

### 16.3 调度器补丁的新位置：`freetoken/kv_swap/runtime.py`

对 `scheduler.pyd` 里的 Cython 类做 monkeypatch（已验证可行）：

- `Scheduler` 是普通 Python `type`，`setattr` 可替换 `__init__` / `run_when_idle` / `shutdown`；
- 替换后的函数 `__globals__` 里能拿到 `CacheManager` 等符号；
- `Scheduler.__init__.__globals__["CacheManager"]` 就是我们打过补丁的 `scheduler/cache.py` 里的类。

`install()` 做三件事（幂等，由 `scheduler/__init__.py` 在导入 `Scheduler` 后调用）：

1. 包裹 `Scheduler.__init__`：调用原实现后，直接构造 `KVHostStore(kv_pool, linear_pool, page_size)` 并挂到 `cache_manager` 上，再 `_attach_evict_hook()`。
   - **不能**再用旧的 `create_host_store()`：它内部检查 `hasattr(kv_pool, "gather_kv")`，0.2.0 原本没有该方法会直接返回 `None`（这是本次升级最初「tier 没装上」的根因）。
2. 包裹 `run_when_idle`：保留原逻辑，追加 `_kv_diag("idle")` 页账、`check_integrity`（含孤儿页回收）、30 s 摘要、`.dump_request` → `_dump_kv_to_ssd`、idle `capture_resident` 预载。
3. 包裹 `shutdown`：先 dump 再走原关闭流程。另挂 `Scheduler._dump_kv_to_ssd`。

启动诊断日志：`C:\AI_Models\logs\kv_boot.log`（可用 `FREETOKEN_KV_BOOT_LOG` 覆盖），记录环境变量、tier 是否启用、前 3 次 idle 的 `host_store` 状态。

### 16.4 重建 `MHAKVCache.gather_kv` / `scatter_kv`

0.2.0 的 buffer 布局是「每层一个扁平 K/V 张量」：

- `self._k_buffer[dense]` / `self._v_buffer[dense]`：形状 `num_pages * page_size * head_dim` 的扁平存储；
- `self._storage_shape = (num_pages * page_size, num_kv_heads, head_dim)`；
- `self._kv_buffer.shape[1]` = 存储层数（含 mrope 展开等）。

实现（与 0.1.3 语义一致）：

```python
def gather_kv(self, slots):
    idx = slots.to(device=self._device, dtype=torch.long)
    n = self._kv_buffer.shape[1]
    shape = (n, int(idx.numel()), *self._storage_shape[1:])
    k = torch.empty(shape, device=self._device, dtype=self.dtype)
    v = torch.empty(shape, device=self._device, dtype=self.dtype)
    for d in range(n):
        k[d].copy_(self._k_buffer[d].view(self._storage_shape).index_select(0, idx))
        v[d].copy_(self._v_buffer[d].view(self._storage_shape).index_select(0, idx))
    return k, v

def scatter_kv(self, slots, k, v):
    idx = slots.to(device=self._device, dtype=torch.long)
    for d in range(self._kv_buffer.shape[1]):
        self._k_buffer[d].view(self._storage_shape).index_copy_(0, idx, k[d].to(self.dtype))
        self._v_buffer[d].view(self._storage_shape).index_copy_(0, idx, v[d].to(self.dtype))
```

`QSAKVCache` 继承 `MHAKVCache`，自动获得这两个方法。

### 16.5 最终补丁清单（0.2.0）

| 文件 | 改动 |
|---|---|
| `kvcache/mha_pool.py` | **新增** `gather_kv` / `scatter_kv` |
| `kvcache/qsa_pool.py` | `gather_index_tiers` / `scatter_index_tiers` 原样重加 |
| `kvcache/linear_state_pool.py` | `snapshot_slot` / `restore_slot` 重加 |
| `scheduler/cache.py` | `__init__` 加 `kv_pool=None, host_store=None`；`_attach_evict_hook`；`attempt_host_restore`；`_kv_diag` / `_tree_page_starts` / `_recycle_orphan_pages`；`cache_req` / `fin` 诊断；`check_integrity` 先回收再报错（`logger.warning`）；`rebuild` 末尾重挂 hook |
| `scheduler/prefill.py` | lock + 复核之后插 restore 循环；恢复后重算估算；异常降级为 warning；`mamba_value` 优先取 `handle.node.mamba_value` |
| `cli.py` | `KeyboardInterrupt → return 130` |
| `kv_swap/runtime.py` | **新增** monkeypatch（见 16.3） |
| `scheduler/__init__.py` | 导入 `Scheduler` 后调用 `install()` |

以下旧补丁**已废弃**：`message/backend.py`、`message/tokenizer.py`、`tokenizer/server.py` 的 `dump_only`；`server/api_server.py` 的 dump 端点。

### 16.6 验证结果（0.2.0 实测）

测试 prompt：22312 tokens（`%TEMP%\opencode\long.json`），host/SSD 中存在 0.1.3 时代写入的 22272 前缀条目。

```
# 冷启动（无命中）
prompt=22312 completion=64 elapsed=53.08s

# 重启后 restore（VERIFY=1）
KV restore verify: max|dK|=0 max|dV|=0 skip=0 add=22272 length=22272 max|dCmp|=0 max|dRope|=0
mamba restore verify: max|d|=0 keys=4
restore elapsed=6.41s
生产能力（VERIFY=0）下再测：5.27s

# 重复请求（GPU 前缀命中）：5.89s
# 多轮会话：prompt 90 → 109 递增，handle=64 命中，idle gap=0
# dump flag：被 run_when_idle 消费，新条目 05483020afb290a2bfc1_0_22272.pt（649 MB）
```

结论：**跨版本（0.1.3 写入 → 0.2.0 读取）KV/mamba/QSA index/rope 全部 bit-exact**，页面账在 idle 时闭合（`gap=0`）。

### 16.7 运维坑（新增）

1. **停服务必须杀进程树**：`ft.exe` 会派生 `python.exe` worker 并占住 GPU。只杀监听端口的进程，会留下占显存的孤儿 → 下一个实例卡在 CUDA 初始化（CPU 停在 5~6 s、GPU 约 4.5 GB 不动、`kv_boot.log` 不生成）。
   ```powershell
   $pid1919 = (Get-NetTCPConnection -State Listen -LocalPort 1919).OwningProcess
   taskkill /PID $pid1919 /T /F
   ```
   确认 `nvidia-smi --query-gpu=memory.used --format=csv,noheader` 回到 0 MiB 再启动。
2. **PowerShell here-string 会展开 `$env:`**：用 `@"..."@` 跑 Python 脚本改 `.ps1` 时，`$env:FREETOKEN_KV_DIAG` 会被替换成空串，替换静默失效。改带 `$` 的脚本请写文件或用 `@'...'@`。
3. 含中文的 `.ps1` 必须 **UTF-8 带 BOM**；`.md` 保持 UTF-8 无 BOM。
4. `kv_persist\index.pre-0.2.0.json` 是升级前的 index 备份，可留作对照。

---

*文档结束。任何与上游冲突的地方，以 `freetoken/kv_swap/` + 本文第 11/16 节的函数为准，升级后逐行 diff。*


## 17. 启动参数与性能调优（0.2.0）

### 17.1 prefill 速度的两个前提

1. **`--moe-cpu-layers auto`**：引擎做 pin probe，把 48 层 expert 权重 pin 到系统内存
   （~63.6 GiB，日志 `pinned 48 MoE layers (63.63 GiB) ... locked 0 []`）。缺了它 expert
   走 pageable 内存搬运，prefill 掉约 5 倍（278 t/s vs 1745 t/s）。
2. **不要在启动参数里直接写 `--moe-cache-size 512 --disable-moe-prefill-overlap`**。
   `--disable-moe-prefill-overlap` 会让引擎在初始化时只分配**单** expert-layer buffer，
   prefill 无法重叠 expert 拷贝 → ~300 t/s。即使之后 rebuild 到 512 也回不来（buffer 已定）。

### 17.2 正确流程（与官方加载器一致）

1. 启动：`--moe-strategy auto`（解析为 offload），**不**指定 `--moe-cache-size`、**不**指定
   `--num-tokens`、**不**加 `--disable-moe-prefill-overlap`。引擎自动解析
   `moe_cache_size=1234`、`prefill_overlap=True` → 分配双 buffer。
2. 就绪（`/health` 返回 `status=ok`）后，POST 运行时 rebuild：
   ```json
   {"moe_cache_size":512,"num_pages":1648,"num_mamba_slots":4}
   ```
   日志会打印 `Disabling MoE prefill overlap on rebuild: ...`，但双 buffer 保留，prefill 仍
   1761 t/s。
3. 本仓库用 `kv-rebuild-after-ready.ps1` 自动完成第 2 步（由 `start-freetoken-flash-next.ps1`
   在启动时 spawn）。

### 17.3 端点与约束

- `POST /v1/cache/rebuild` → `{"status":"ok","moe_cache_size":512,"num_pages":1648,
  "mamba_slots":4,"num_swa_pages":0,"error":null}`
- `GET /v1/cache/status` → `geometry.unit_bytes`：`kv_per_token=25344`、
  `moe_per_expert=2780160`、`mamba_per_slot=115642376`；`cache_budget_bytes`（本机约 4.36 GiB）；
  `limits.mamba_slots.min=4`（**不能设 0**）、`limits.moe_experts.min=512`、`limits.kv_tokens.max`
- rebuild 受预算约束，超了返回 `status:"rejected"` 且 `old cache kept, still serving`。
  例：`moe=512 + kv=1610 + mamba=8` 需要 3.76 GiB > 预算 3.39 GiB → 被拒；改 `mamba=4` 通过。

### 17.4 实测（RTX 5080 16 GB，Qwen3.8-Flash-Next-NVFP4）

| 配置 | 70k prefill | decode |
|---|---|---|
| auto 启动（KV 8256） | 6885 tok = 2363 t/s | 35.4 t/s |
| auto 启动 + rebuild 512/1648/4 | **1745 t/s** | **30.6 t/s** |
| 启动即 512 + disable-overlap | 302 t/s | 30.3 t/s |

### 17.5 与 KV swap 补丁的关系

补丁**不影响** prefill 速度：用官方参数 + 本仓库 PYTHONPATH（补丁生效）实测 2363 t/s。
KV swap 的 dump/restore 在新几何（105472 tokens、512 slots）下正常（dump 写出 1726 MB 条目）。


## 18. 0.2.0 完整验证记录（2026-10-22）

### 18.1 修复：restore 命中「树里已存在」前缀时的孤儿页

`HybridRadixCache.match_prefix` 只把命中截断到最深的 `mamba_value` 非空节点。当某个会话的
GDN 快照被驱逐、但其 KV 页仍在树里时，共享该前缀的请求命中长度为 0，会走 host restore 并
新分配页；`insert` 此时看到节点已存在（`prefix_len == insert_len`）就不接管传入
`kv_indices`，新页成为孤儿 → `check_integrity` 报 `free + cache != num_pages` 并退出 worker。

修复（`scheduler/cache.py::attempt_host_restore`）：利用 `insert` 返回的 `prefix_len`
把未接管的新页还回池。

```python
_prefix_len, existed = self.prefix_cache.insert(ids, full_slots, mamba_slot)
if existed:
    raise RuntimeError("host KV restore collided with an existing snapshot")
_unused = _prefix_len - cached_len if _prefix_len > cached_len else 0
if _unused > 0:
    self._free(slots[:_unused])
```

### 18.2 mamba slots 与「可续算会话数」

`match_prefix` 依赖 live `mamba_value`；mamba 池 slot 数由预算决定（本机
`/v1/cache/status` → `limits.mamba_slots = {min:4, max:40}`，每 slot ≈110 MiB）。
slot 用尽后 LRU 清快照 → 会话前缀（KV + host/SSD 条目）全部不可复用 → 全量 prefill。

| 几何 | 多会话交替 |
| --- | --- |
| KV 1648 页 / mamba 4 / MoE 512（105472 token） | 约 4 个会话后互相驱逐，旧会话全量重算 |
| KV 1024 页 / mamba 12 / MoE 512（65536 token） | 复用正常（约 1.8~2.4 s / 22k） |

### 18.3 端到端验证汇总

| 用例 | 结果 |
| --- | --- |
| dump → 重启 → restore（22336 前缀） | `max\|dK\|=max\|dV\|=max\|dCmp\|=0`，`mamba max\|d\|=0`；耗时 14.3 → 2.7 s |
| 共享前缀分叉（A/B） | `add=22272`，`dK/dV/dCmp=0`，页账 `gap=0` |
| 多会话交替 | 命中复用（约 1.8 s） |
| 短会话（<256 token） | 不 dump、不崩 |
| 停服 dump（`stop_server_isolated.ps1`） | SSD 4 → 6 条，优雅停，GPU 无残留 |
| 并发 2 请求（max-running=1） | 串行化，均成功 |
| undo（更短前缀） | 命中更短节点 |
| compact（新前缀） | 冷 prefill，不崩 |
| 页账 idle | 全程 `gap=0` |

### 18.4 安全说明

`HostStore.lookup` 用 `torch.equal(ids[:length], entry.tokens)`、`SSDStore.lookup` 用
`sha1`，均要求 token 序列**完全一致**，不会跨会话误命中（实测两个只差第 47 个 token 的会话
各自只命中自己的条目）。


## 19. GDN 快照随驱逐落盘（mamba-on-evict，2026-10-22）

### 19.1 动机
`HybridRadixCache.match_prefix` 只把命中截断到**最深的、`mamba_value` 非空的节点**。GDN 是
recurrent 的，从中间续算必须先有该边界的 CM（conv + recurrent）快照；slot 被 LRU 清掉后
`mamba_value=None`，该前缀（哪怕 KV 页仍在树里、host/SSD 里还有条目）就再也无法复用。
4-slot 几何下约 4 个会话后开始互相驱逐 → 旧会话全量 prefill。

### 19.2 设计
- **只在驱逐现场补抓 GDN 快照**，不抓 KV（`snapshot_slot`，一次 D2H ≈110 MiB，毫秒级）；
  KV 继续由 idle preload 负责。全量 capture-on-evict 会同步 `gather_kv` 并卡住 decode。
- 钩子挂在 `HybridRadixCache._free_node_mamba`（全仓库唯一清 `mamba_value` 的地方），
  同时覆盖两条驱逐路径：
  * `evict_mamba`（内部/共享节点）→ tombstone，**KV 与 children 保留**；
  * `evict_full`（叶子）→ KV 页释放，快照仍抓（KV 只能靠 host 已有的条目配对）。
- **恢复**用 `_try_reattach_mamba`：定位 tombstone 节点 → 从 host 取 mamba 条目 →
  `alloc slot` + `restore_slot` → 重挂 `node.mamba_value` → 迁移锁。**零 KV 搬运**
  （日志 `reattach ok ... add=0`）。
- mamba-only 条目通过 `require_kv=False` 与常规 KV 恢复隔离；SSD 同文件名（按 tokens sha1），
  用 meta 的 `mamba_only` 标记保证"KV 覆盖 mamba-only、反向不覆盖"。

### 19.3 边界与已知行为
- **恢复仍要现场占 1 个 slot** → 同时活跃会话数仍受 slots 限制；软化的是"历史会话不必常驻"。
- **`mamba=None` 的条目保持不可恢复语义**（引擎原有拒绝逻辑不动）：快照只在 slot 活着时抓得到。
- **rebuild**（`/v1/cache/rebuild`，含 `num_mamba_slots` 变更）会整片丢弃快照，两处 WARNING。
- **abort** 走 pyd 的 `_free_req_slots`，只归还 slot 且受 `keep_live` 保护，实测 mamba 未被误清。
- 限流：`FREETOKEN_MAMBA_CAPTURE_MAX_PER_EVICT`（默认 8，0.5 s 窗口内超出则跳过），
  防止一次驱逐风暴长时间阻塞；每次捕获打耗时日志（节点/total ms）。

### 19.4 实测
| 场景 | 结果 |
| --- | --- |
| 8 会话交替（4 slots）后第 1 会话回来 | 1.6~1.9 s（reattach），`max|d|=0` |
| rebuild（丢 1 快照） | `rebuild drops 1 mamba snapshot(s)`，不崩 |
| 长请求中途 abort | 服务存活，后续请求正常，计数一致 |
