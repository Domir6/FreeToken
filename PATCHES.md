# FreeToken KV Swap 补丁集 — 目标版本 FreeToken 0.2.0+g425dc7bf1

本仓库在官方 FreeToken **0.2.0+g425dc7bf1** 之上打了「GPU KV → host RAM → SSD」三级缓存
与 SSD 持久化补丁。本文件说明同步到 GitHub / 新机器时怎么重建。

## 1. 基线

| 项 | 值 |
| --- | --- |
| FreeToken | 0.2.0+g425dc7bf1 |
| torch | 2.14.1+cu130 |
| Python | 3.12 |
| 官方包 | `%LOCALAPPDATA%\FreeToken\venv\Lib\site-packages\freetoken` |
| 隔离目录 | `C:\Users\X303\Documents\FreetokenP`（`PYTHONPATH` 指向它，覆盖官方包中的同名 `.py`） |

`0.2.0` 起 `scheduler/scheduler.py`、`server/api_server.py` 等变成 `.pyd`（不可改），
调度器相关补丁因此改为运行时 monkeypatch（见 `freetoken/kv_swap/runtime.py`）。

## 2. 修改的官方文件（`patches/*.diff`）

| 文件 | +/- | 说明 |
| --- | --- | --- |
| `kvcache/mha_pool.py` | +26 | 新增 `gather_kv` / `scatter_kv`（0.2.0 删除了这两个方法，dump/restore 依赖它们） |
| `kvcache/qsa_pool.py` | +40 | 新增 `gather_index_tiers` / `scatter_index_tiers`（QSA 压缩 index key + mrope 位置） |
| `kvcache/linear_state_pool.py` | +30 | 新增 `snapshot_slot` / `restore_slot`（GDN/mamba 状态快照） |
| `scheduler/cache.py` | +711 -7 | `CacheManager` 注入 `kv_pool`/`host_store`；`attempt_host_restore`；页账诊断 `_kv_diag`/`_recycle_orphan_pages`；`check_integrity` 先回收孤儿页再报错 |
| `scheduler/prefill.py` | +36 -1 | 在 0.2.0 的 `lock(handle)` 之后插入 host restore 循环并重算估算；`mamba_value` 取最终节点 |
| `scheduler/__init__.py` | +6 | 导入 `Scheduler` 后调用 `kv_swap.runtime.install()` |
| `cli.py` | +6 -1 | `KeyboardInterrupt → return 130` |

生成方式（本机命令，可重跑）：

```powershell
python tools\mk_patches.py   # 逐文件 unified diff，输出到 patches\
```

## 3. 新增文件（不属于官方包）

| 文件 | 说明 |
| --- | --- |
| `freetoken/kv_swap/__init__.py` | 包入口 |
| `freetoken/kv_swap/manager.py` | `HostEntry`（tokens + kv_k/kv_v + mamba + cmp_index/rope_pos）、`KVHostStore`（capture/lookup/take/summary） |
| `freetoken/kv_swap/ssd.py` | `SSDStore`（index.json + sha1 校验 + LRU） |
| `freetoken/kv_swap/runtime.py` | **0.2.0 关键**：monkeypatch `Scheduler.__init__` / `run_when_idle` / `shutdown`，注入 host store、消费 `.dump_request`、退出落盘 |
| `start-freetoken-flash-next.ps1` | 启动脚本（官方参数 + `--moe-cpu-layers auto`，见 §6） |
| `kv-rebuild-after-ready.ps1` | 启动后等 `/health=ok` 再 POST `/v1/cache/rebuild`（恢复双 expert buffer 性能，见 §6） |
| `stop_server_isolated.ps1` | 停服：写 `.dump_request` → 发 1-token 触发 idle dump → 等 flag 消费 → CTRL_C → 兜底 tree kill |
| `tools/mk_patches.py` | 重新生成 `patches/*.diff` 的脚本 |

## 4. 应用步骤（新机器）

1. 安装官方 FreeToken 0.2.0+g425dc7bf1（得到 `...\venv\Lib\site-packages\freetoken`）。
2. 把官方 `freetoken` 整树复制到隔离目录，例如
   `robocopy <official>\freetoken <repo>\freetoken /MIR /XD kv_swap __pycache__ /XF *.pyc`。
3. 应用 `patches/*.diff`（7 个文件），或直接用本仓库已打好补丁的 `.py`。
4. 复制新增文件：`freetoken/kv_swap/` 4 个 `.py` + 3 个 `.ps1`。
5. 设环境变量（见 §5），用 §6 的方式启动。

## 5. 环境变量

启动脚本会设置（`FREETOKEN_KV_*`）：

| 变量 | 值 | 说明 |
| --- | --- | --- |
| `PYTHONPATH` | `C:\Users\X303\Documents\FreetokenP` | 让补丁 `.py` 覆盖官方同名模块 |
| `FREETOKEN_KV_SWAP` | `1`（默认开）| 启用 host 层；`0` 关闭（唯一总开关）|
| `FREETOKEN_KV_SWAP_HOST_BYTES` | `17179869184` | host 层上限 16 GiB |
| `FREETOKEN_KV_SWAP_MIN_TOKENS` | `256` | 小于该长度的前缀不入库 |
| `FREETOKEN_KV_SWAP_PIN` | `0` | host 层不 pin（生产避免 OOM） |
| `FREETOKEN_KV_PERSIST` | `1`（默认开）| 启用 SSD 层；`0` 关闭 |
| `FREETOKEN_KV_PERSIST_DIR` | `<repo>\kv_persist` | SSD 目录 |
| `FREETOKEN_KV_PERSIST_BYTES` | `68719476736` | SSD 上限 64 GiB |
| `FREETOKEN_KV_RESTORE` | `1`（默认开）| 允许恢复；`0` 关闭 |
| `FREETOKEN_KV_DIAG` | `0` | `1` 时写页账诊断到 `C:\AI_Models\logs\kv_diag.log` |
| `FREETOKEN_KV_RESTORE_VERIFY` | `0` | `1` 时写数值校验到 `restore_verify.log` |
| `FREETOKEN_KV_BOOT_LOG` | 可选 | 覆盖启动诊断日志路径（默认 `C:\AI_Models\logs\kv_boot.log`） |
| `FREETOKEN_KV_SWAP_MAMBA_ON_EVICT` | `1` | 驱逐前只补抓 GDN 快照（默认开）；`0` 关闭 |
| `FREETOKEN_MAMBA_CAPTURE_MAX_PER_EVICT` | `8` | 单次驱逐风暴最多补抓几条快照，超出跳过 |

## 6. 启动 / 停止

**启动**（单条命令，不要与 kill 混在同一条，否则子进程可能被一起收掉）：

```powershell
Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
  '-File','C:\Users\X303\Documents\FreetokenP\start-freetoken-flash-next.ps1' -WindowStyle Normal
```

启动参数（对齐官方加载器性能的关键）：

```
serve --model <模型> --port 1919
  --moe-strategy auto          # → offload
  --moe-cpu-layers auto        # pin 48 层 expert 到内存（locked 0），prefill 提速 ~5x 的前提
  --text-model-only
  --memory-ratio 0.95
  --max-running-requests 1
  --moe-cpu-threads 32
```

**不要**在启动时就写 `--moe-cache-size 512 --disable-moe-prefill-overlap`：那样只分配**单**
expert buffer，prefill 会掉到 ~300 t/s。正确做法是 `kv-rebuild-after-ready.ps1` 在就绪后
POST `{"moe_cache_size":512,"num_pages":1216,"num_mamba_slots":10}`。

**几何选择（实测，2026-10-22）**：`cache_budget_bytes` 约束的是 `KV + mamba`（MoE cache 另算），
本机该预算 ≈2.96 GiB，即 `num_pages*1622016 + num_mamba_slots*115642376 <= 3177893888`。
- **单会话长上下文优先**：`num_pages=1648 (105472) / mamba=4` → 单会话最优，但多会话只有 1 个
  历史名额（3 租户交替时 2/3 会全量 prefill）。
- **多会话/子智能体优先（当前采用）**：`num_pages=1216 (77824) / mamba=10` → 3 租户 6/6 命中
  （~2.9 s），单会话 66.8k 冷 prefill 1783 t/s。
- 经验公式：**每个活跃请求占 6 个 slot**（3 个工作 + 引擎强制预留 3 个 free），
  每多一个"历史会话"多 1 个；所以 mamba slots ≈ `6 + 历史会话数`（X 个并行租户留 X~2X 个）。

> 内存压力：长时间运行后 `freeRAM` 降到 ~23 GiB 时，冷 prefill 会从 ~1780 t/s 掉到 ~600 t/s
> （系统分页/内存压缩所致，pin 的 MoE banks ~63 GiB + host KV store 累积）。**重启即恢复**；
> 长期跑建议定期重启或下调 `FREETOKEN_KV_SWAP_HOST_BYTES`。

**停止**：

```powershell
& C:\Users\X303\Documents\FreetokenP\stop_server_isolated.ps1
```

必须先落盘再停：脚本会写 `kv_persist\.dump_request`、发一个 1-token 请求触发 idle、等 flag
被消费，然后 CTRL_C；超时或 `-Force` 才 tree kill。**直接强杀不会落盘**。

## 7. 已知限制与运维注意

1. **mamba state slots 曾是「同时可续算的会话数」的硬上限，现在被 `mamba-on-evict` 软化。**
   `match_prefix` 只把命中截断到最深的 `mamba_value` 非空的节点；过去 slots 用尽后 LRU 清掉旧
   会话的 GDN 快照，其 KV 前缀（含 host/SSD 条目）**永久无法复用**，只能全量 prefill。
   现在驱逐前会**只补抓 GDN 快照**（一次 D2H，不搬 KV），之后该会话再次请求时由
   `_try_reattach_mamba` 把快照写回一个新 slot 并重挂到树上（`add=0`，零 KV 搬运）。
   - 实测（KV 1648 页 / mamba 4 / MoE 512）：8 个会话交替后第 1 个会话回来 1.6 s（reattach），
     而非全量 14~18 s；`mamba reattach verify: max|d|=0`。
   - 注意仍有下限：恢复要现场占 1 个 slot，所以**同时活跃**会话数仍受 slots 限制（4 slots ≈ 4）。
     要更多并行会话就把 `num_mamba_slots` 调大（受 `cache_budget_bytes` 约束）。
   - 用 `/v1/cache/status` 的 `geometry.limits.mamba_slots` 查看范围（本机 min=4, max=40）。
2. **页账**：`idle` 时 `gap` 必须为 0（`FREETOKEN_KV_DIAG=1` 时可见于 `kv_diag.log`）。
   历史上有一个孤儿页 bug（restore 到「树里已存在」的前缀时重复分配页）已修复。
3. **停服务必须走 `stop_server_isolated.ps1`**（或至少先写 `.dump_request` 并触发一次请求）。
   只用 `taskkill /T /F` 会丢 resident 前缀。
4. `ft serve` 的 `--moe-backend` 是 `--moe-strategy` 的弃用别名；`--num-tokens` 必须是
   resolved page size（QSA 下为 64）的整数倍。
5. QSA 的后端页面大小被引擎强制为 64；`--num-tokens` given 值按 64 对齐（103040 / 105472 均可）。

## 8. 参数语义对照（易混淆项）

| 参数 | 官方英文原文（`ft serve --help`） | 实际含义 | 本机值 |
| --- | --- | --- | --- |
| `--moe-strategy` | "How the routed experts are served" | expert 的服务方式（auto→offload） | `auto` |
| `--moe-cache-size` | "The number of unified **MoE expert slots** on GPU" | **GPU 上 MoE 专家槽位**（独立池） | `512` |
| `--moe-cpu-layers` | "which MoE layers **compute on the CPU executor** instead of the GPU offload/PCIe path" | **哪些层 locked 到 CPU 计算**（不是"pin 哪些层"） | `auto`（= pin 48 层, locked 0） |
| `--moe-cpu-threads` | "Number of CPU worker threads for `--moe-strategy cpu` decode experts. 0 = auto (physical cores)" | CPU 计算 expert 的线程数 | `32` |
| `num_mamba_slots`（rebuild 参数） | `--memory-ratio` 侧文档："mamba **state slots**"（`LinearStatePool`, `linear_state_cache_ratio`） | **GDN/SSM 每请求 recurrent 状态槽**（活跃请求占 3，其余为可复用快照） | `4` |
| `--num-tokens` | "Total KV-cache capacity in tokens" | KV 页总容量（须为 resolved page size 的倍数，QSA 下 64） | `103040` |

> ⚠ **`num_mamba_slots` 最易误读**：中文常译作"并行推理时保留的 **SSM 状态**槽位数"。
> 这句没错（SSM = 状态空间模型 = GDN 线性注意力层），但"SSM 状态"极易被读成"MoE 专家"。
> 二者是**两个独立的显存池**：
> - `--moe-cache-size` → GPU 上 **MoE 专家**槽位（本机 512，每 slot ≈ 2.65 MiB）
> - `num_mamba_slots` → **GDN/SSM recurrent 状态**槽位（本机 4，每 slot ≈ 110 MiB）
>
> 另外 `num_mamba_slots` 并不等于"并行会话数"：并行度由 `--max-running-requests` 决定；
> 每个活跃请求要占 **3 个** mamba 槽（1 live + 2 ping-pong），剩余槽才是"可复用的历史会话快照"。
> 遇到拿不准的参数，**以 `ft serve --help` 的英文原文为准**。

## 9. mamba 快照 pinned-staging 异步捕获（spec v1.1，2026-10-24 落地）

驱逐前落盘能力以**异步形态恢复**（KNOWN TRADE「已停用」→「已以异步形态恢复」）：

- **捕获（驱逐路径，同步 ~5-15ms）**：`_free_node_mamba` wrapper → `capture_evicted_mamba`
  （保留 `FREETOKEN_MAMBA_CAPTURE_MAX_PER_EVICT=8` 突发限流）→ `capture_node_mamba`：
  CUDA-graph guard → sha1 查重（`dedup_hit`）→ patch 判定（host 已有同前缀 KV 条目且缺
  mamba → 任务带 `patch_existing`）→ pinned staging ring（**4 槽 × ~110MB，写死常量不做 env**）
  `copy_in`：**独立 side stream** 上 D2H + event sync（主流 kernel 队列不阻塞拷贝，这是
  旧版 47-62ms 停顿的根因）→ **入队即预标记** `_mamba_captured` → 立即返回调度。
  ring 满 → 跳过（`skip_ring_full`），禁止等待。
- **drain（idle-gated，调度线程内同步执行，永不与 decode/prefill 并行）**：
  每 idle 周期最多 `FREETOKEN_MAMBA_DRAIN_MAX_PER_IDLE=4` 个任务，16MB 分片
  `copy_`（staging pinned → pageable）→ 落 host store 为 mamba-only 条目
  （`kv_k=None`/`kv_start=0`）；**host 满按 LRU 只驱逐不落盘**（不走 `_spill`，磁盘 I/O
  永不进写入路径）；**mamba-only 条目只活两级（VRAM→host），`flush_ssd` 跳过它们**
  （SSD 留给 KV 大条目，刻意取舍）。
- **退出/停服**：`drain_all()` 同步排空队列（替代线程 join），挂在 idle dump 与
  shutdown dump 两处 `flush_ssd` **之前**——实测停服时 `staged=2 → drained=2`，无丢失。
- **统计**：`store.summary()` 每 30s 输出
  `| mamba_staging staged=N drained=N skip_ring_full=N dedup_hit=N ring_free=M inflight=K`，
  slot 账恒等 **`staged = drained + inflight`**（skip 不占 slot，独立计数）。
- **env**：`FREETOKEN_KV_SWAP_MAMBA_ON_EVICT`（**默认 1，恢复为开**；ps1 亦有显式行）、
  `FREETOKEN_MAMBA_DRAIN_MAX_PER_IDLE=4`。禁止新增 ring/pin 类 env。
- **验收（2026-10-24 实测）**：
  - `tools/test_staging.py` 28 项全 PASS（ring 满/字段一致/账恒等/drain_all/patch/flush 过滤/LRU 无 SSD 写）；
    `invariant_suite.py` 全绿（含新增 3 项动态不变量：账恒等、rebuild 不变项、drain 无悬挂）。
  - rotation（`tools/test_rotation.py`，8 会话 × ~22k × 2 轮）：lap-2 全命中（worst 2.2-2.8s
    对 lap-1 冷 13s）、restored 0→14、**15 KV + 15 mamba verify 全部 max|d|=0**、账平（staged=9=drained）。
  - compact 风暴同条件对照（空树、真冷 75.6k×3）：OFF 1093/966/980 t/s vs ON
    **1954/1732/1712 t/s**（ON 不慢于 OFF，时段噪声带内）；stage 停顿量化值 0/15/46ms
    （Windows 计时器 15.6ms 量化，真实 ~5-15ms；storm 总停顿 <0.2%）。
  - decode 26.9/27.4 t/s：decode 路径架构性零参与（无驱逐→钩子不触发→drain 只在 idle），
    数值处于历史时段噪声带（21.1-29.4）内。



### 已知坑（WDDM 显存口径）

- `torch.cuda.mem_get_info()`（cudaMemGetInfo）在引擎按 memory-ratio 吃满配额后**恒报 0**：
  它是 CUDA 上下文视角（本进程还能 cudaMalloc 多少），不是驱动/物理视图；Windows WDDM 下
  两者常严重不一致。要看"还有多少物理显存可抢"（17 t/s 案里外来图形进程的占用），
  只能用 `nvidia-smi --query-gpu=memory.free`（30s 统计行的 mem 探针已用此口径，
  失败才回退 cudaMemGetInfo）。同理由不适用于 Linux 的多数场景。

### 显存账（第三口径：驱动私房开销）

- 三个口径不相等：Sysinternals（D3DKMT 承诺）/ NVML / DXGI 各有差值，总计 ~0.7 GB——
  这是 WDDM 管理结构、分页表、驱动内部堆等**私有开销**：不挂任何进程名下，
  nvidia-smi 与任务管理器都看不见，但物理上占着显存。
- 后果：nvidia-smi 显示 0.7G 空余时，驱动层可能已承诺满（16/16）——那 0.7G 是
  "看得见拿不到"的幻影；squatter 一申请图形分配就撞驱动层全满 → 分配排队/失败/
  引擎侧重试（17 t/s 案的机制解释）。
- **规则：驱动层承诺（D3DKMT）通常比 CUDA 视图高 0.5~0.8G，该差额为驱动私有开销，
  不可用；一切余量规划以 nvidia-smi 口径 −0.5G 计。**
- 判读阈值随之更新：freeVRAM **< 1.0G = 危险区**（原 0.8G）；
  规划余量目标 **1.5G**（原 1.2G）。探针口径不换（nvidia-smi 继续做触发与曲线）。
- **实测（2026-10-25，Get-GPUMemoryD3DKMT.ps1 3s 采样）**：
  `VRAMTotalMB=15977 / DedicatedUsedMB=15964`（任务管理器同源）→ **物理真实余量 13 MiB**；
  同刻 nvidia-smi free=150 MiB（**虚高 ~137 MiB**）；D3DKMT `LocalAvailMB=7732` 报 7.7G
  **纯预算口径、含可换页/共享承诺，禁用于物理余量判断**。
  → 最准的物理剩余 = `VRAMTotalMB − DedicatedUsedMB`（工具 CSV 持续可读）；
  nvidia-smi 口径的打折随负载波动（实测 0.13G，保守规则仍按 −0.5G 计）。

## 10. 会话级整段快照（session snapshot，spec v1.0，2026-10-25 落地）

GPU 零命中（cached_len=0）时整会话恢复，替代 ~25s+ 全量 prefill：

- **键空间隔离**：`seg:`（radix 节点段，原有 `_captured`）/ `mb:`（mamba-only）/ `sess:`
  （整段，新 `_sess_captured`）三集合独立；SSD meta 携带 kind，`_notify_drop` 按 kind 清
  对应负缓存。
- **捕获**：idle 转换时 `capture_sessions`（**每次 idle 最多 1 个会话**，写死；候选按整链
  长降序；叶子 GDN slot 已失的跳过——无线性态的快照不可恢复）→ `_gather_full_chain`：
  root→leaf 链逐节点 gather 拼整段 + `snapshot_slot(leaf)` → `HostEntry(kind="sess",
  kv_start=0)`。**先清后存**：insert 前清同会话 seg 条目（replace-not-grow；seg 的 SSD
  归档保留，属回退层）。
- **LRU 权重**：host 驱逐优先级 `mb(0) → seg(1) → sess(2)`，LRU 在同 kind 内生效——
  整段是唯一冷启动路径，最后死。
- **恢复**：`attempt_host_restore` 在 `cached_len==0` 时走 `store.lookup_session`
  （host → SSD 归档），命中后**复用现有 scatter/re-insert/锁迁移尾部**；mamba restore_slot。
- **开关**：`FREETOKEN_KV_SESS_SNAPSHOT`（默认 1）。总开关之外的节奏常量全部写死。
- **预算驱动容量账**：整段大小 = tokens×25344 B + 110 MB（45k ≈ 1.25 GB）；
  可容纳会话数 = HOST_BYTES ÷ 整段大小（24G≈19-21、16G≈13-14、8G≈6-7）；
  预算内整段优先、超出 LRU 丢整段（不阻塞、不膨胀）。
- **验收（2026-10-25 实测）**：
  - rotation（tools/test_sess_rotation.py，2×45k 两轮，87040 池强制互驱）：
    lap-2 整段恢复 **1.8s / 1.6s**（≤3s 红线；对照全量 25s+，10 倍改善）；
  - VERIFY=1 轮（数值证明，原始行）：
    `KV restore verify: max|dK|=0 max|dV|=0 skip=0 add=44736 length=44736 max|dCmp|=0`
    `mamba restore verify: max|d|=0 keys=4` —— **skip=0 = 整链签名**（非段续长）；
  - 单测 test_sess_snapshot.py 22 项全 PASS；invariant_suite 全绿（+8 项：sess 静态 4 +
    动态并入）；
  - **顺带修复原有 bug**：`capture_node` 查重后从不写 `_captured` → idle preload 的查重
    失效、同批叶子每 idle 重抓（saved 计数暴涨 churn 的主因）。已修：insert 成功后标记。
- **decode 汇报纪律**：decode t/s 必须带 context_length 成对出现（如 `decode 18.4 @ 76k ctx`）；
  缺 context 的 decode 数字视为无效数据，不进任何对比。
### 10.1 修复（2026-10-25 午后）：take 的“消费”语义导致每次回访都走 SSD



`take()` 原语义 = 恢复完成后把 host 条目删除（消费）。后果：驱逐→回场循环中 host 恒空，

每次回访都被迫走 SSD 慢路（torch.load ~2 GB，首捞后 host 又空）——实测表现为“恢复每次

都读 SSD/暂停后像失效”；耐力测试里 s2 锁死 8.3-8.4 t/s（s1 正常 28，因 s1 的条目恰

好未被消费）。修复：**take 只计数（restored/restored_bytes），不再 remove**——

条目常驻 host，保留交给 LRU（24G 预算、mb→seg→sess 淘汰序）。

验收：8 轮 × 双会话 × 500 tok 真 decode——修复前 s2 lap2 起锁死 8.3-8.4 t/s；

修复后 **28.0-29.7 t/s @ ~71k ctx 零掉速**（s1 26.5/28.0-28.9、s2 9.4/28.3-29.7，

首笔为 SSD 冷捞一次性成本）。
### 10.2 追加（2026-10-25 下午）：压力触发捕获 + seg 保留 + 替换归档



- **压力触发**：prefill 入口（准入检查前）检查池压力——available < 25% 时、在新请求

  prefill 之前，先抓最大的 1 个未抓会话快照（~1-2s）。动机：子代理是背靠背 tool 循环，

  步间没有 idle 转换，idle 钩子的抓取永远轮不到它们；而 >80% 压力意味着 prefill 马上

  要驱逐整会话——抓一把（1-2s）把"白丢"变成"下一步 2s 级恢复"。低压（<80%）零开销。

- **seg 保留**：DIVERGE 分析（toolset 动态切换 / EOS-消息边界回填渲染）证明客户端

  回填的历史与生成态在 token 深处分叉（实测 73659/76397=96.4%、10700/45486=76.5%），

  分叉点之前的节点段正是 seg 续长的本钱——`_drop_session_segs` 不再清 seg，

  只替换同会话的被取代 sess 快照。

- **替换归档**：被取代的 sess 快照走 `_spill` 降级进 SSD（回退层），不再裸删。

- **验收（场景 3 重跑，300 tok 真 decode/步）**：20/20 步 11.8-12.9s（除末步 19.7s），

  A/B 中位 12.04/12.49s 完全对称；对照修复前 B 的 step4/7/10 = 173/194/216s 全量。

  restore final 打点：`cached=79104/81859 (was 0)` = 96.6% 部分恢复。

- DIVERGE 三例的 tokenizer 解码定性：①toolset 动态切换（"read"→"question"，客户端行为，

  服务端修不了）②不同会话内容的正确拒绝（非 bug）③decode 延伸的模板 token
  （<|im_end|>/<|im_start|>）与回填渲染的差异（radix 部分命中天然兜住）。
### 10.3 追加（2026-10-25 傍晚）：快照=版本史 + free_slots 压力判据



- **快照=不可变版本史（用户裁决，取代 10.2 的"替换"）**：全抓、不清旧——被取代的

  快照是"回退到 compact 之前"的时间机器；retention 全交 host LRU（24G，

  mb→seg→sess 淘汰序）+ SSD 归档（64G），SSD 驱逐清负缓存（可重抓闭环）。

- **压力判据修正**：available_size 把 evictable 树页算作"可用"（准入语义），

  以它为压力信号则**永不触发**（部署后零触发实锤）。改用 **free_slots 真空闲页**：

  `free_tokens*100 < (tree+free)*25` → 触发。修后打点实证全程在岗：

  `pressure capture: no new snapshot (free=10688→2752)`（随池压满递减）。

- **restore 耗时打点**：`restore final: ... took=937-1563ms`（78-84k 快照的 H2D

  物理拷贝，pageable ~2GB/s；radix 命中时 took=125ms）。"识别时间长"的量化：

  是拷贝不是查找。优化空间（未来项）：host 条目 pinned 化可到 ~0.1s，但 pinned

  配额敏感（阶段七事故），不动。

- **粒度澄清**：64 = page_size = 抓取/恢复最小粒度（引擎页）；min_tokens=256 =

  会话快照门槛。两者不同层，均保持。

- **验收（场景 3 终态）**：20/20 步 11.7-13.7s（300 tok decode @ 60-84k ctx），

  A/B 中位 12.30/12.40s 全对称，零全量；压力触发全程在岗（free 递减曲线）。
