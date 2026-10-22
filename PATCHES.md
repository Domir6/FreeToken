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
