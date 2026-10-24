# FreeToken 改造进度记录

> 本文件由 AI 会话维护，用于跨会话/上下文压缩后恢复进度。最后更新：2026-10-20

## 环境事实（已确认）

- 隔离目录：`C:\Users\X303\Documents\FreetokenP`
- pip 包：`C:\Users\X303\AppData\Local\FreeToken\venv\Lib\site-packages\freetoken`
- GitHub 源码：`C:\Users\X303\Downloads\FreeToken-main\python\freetoken`（注意：不是 MasterPlan 里写的双层目录）
- venv python：`C:\Users\X303\AppData\Local\FreeToken\venv\Scripts\python.exe`
- **运行时**：用户可能在运行 FreeToken。需要重启/实测时必须先确认（端口 1919）。
- Python 3.12 / Windows 11 / PowerShell 5.1（无 rg，用 grep 工具）
- **编码坑**：绝不要用 `Get-Content`/`Set-Content`（默认 ANSI/GBK）处理含中文的 UTF-8 文件（会不可逆损坏）；
  用 read/write/edit 工具或 `[System.IO.File]::ReadAllText/WriteAllText`。

## 阶段零：隔离部署（已完成，方案已改）

- [x] 创建 scripts/patches/logs/backup/kv_persist
- [x] 复制 pip freetoken → `backup\freetoken_pristine` + `freetoken`（663 文件）
- ~~GitHub 474 个 `.py` 全量覆盖~~ **已废弃**（见"ABI 适配结论"）
- [x] 删除 `freetoken\moe\offload_cache.cp312-win_amd64.pyd`、`freetoken\scheduler\scheduler.cp312-win_amd64.pyd`
      （强制加载对应 `.py`）
- [x] 创建 `scripts\start_isolated.ps1` / `scripts\start_server_isolated.ps1`
- 版本：pip `0.1.3+gfa0e9537d`
- 注意：`engine/engine.cp312-win_amd64.pyd` **仍保留** → `engine.py` 的补丁不生效

## ABI 适配结论（2026-10-20 实测确认）

pip 只把底层编译成 `.pyd`（`engine`、`kernel/*`、`daemon/*`、`models/*/weight`、`distributed/impl`、
`checkpoint/ftw`、`moe/offload_kernels`）；`scheduler`、`layers`、`kvcache`、`models/qwen4_exp`、
`moe/offload_cache` 在 pip 里是 `.py`。GitHub main 的上层 `.py` 互相耦合，**混用 pristine 与 main 会接口错配**：

- `OffloadMoeCache.bank_views()` → main 改成按 pool 的 `pool_bank_views(layer_id)`（slot 空间也变成 per-pool）
  → 必须整套使用 main 的 `layers/moe.py` + `moe/offload_cache.py`
  （否则启动到 "Start capturing CUDA graphs" 时 `AttributeError: 'OffloadMoeCache' object has no attribute 'bank_views'`）
- `CacheManager(..., kv_pool=, host_store=)` 与 `create_host_store()` 只在 main `scheduler/scheduler.py` 里接线
  → 必须使用 main `scheduler.py`（删除 `scheduler.pyd`），否则 host/SSD store 永远不创建
- `kernel/fast_index_copy.default_blocks_per_bank` 只有 main 版有
- `models/qwen4_exp/config.py` 用 pristine（已处理 `qwen_sparse_attention`；main 只认 `indexed_attention`）
- `moe/offload_cache.py` 需 `from freetoken.env import ENV`
- **最终可用组合**：main `.py`（scheduler / layers.moe / moe.offload_cache / moe.prefetch / qwen4_exp.model /
  kvcache.* / env / kernel.fast_index_copy）+ pip `.pyd`（engine / kernel / daemon / weight / offload_kernels）

## 实测结果（2026-10-20，隔离目录，8192 / 32768 tokens）

- [x] 基线启动 + 推理正常（短问答、CUDA graph bs=1）
- [x] GPU prefix cache：同前缀第二次 `#cached-token: 2496`
- [x] host KV swap：驱逐后 `host KV swap-in: restored 896 tokens`，`#cached-token: 896`
- [x] SSD 持久化：`host KV: persisted 1 entries to SSD` → 重启后 `ssd_kv loaded=1` → swap-in 成功
- [x] #678 侧流：3 个 prompt 的 content/reasoning 与基线**逐字节一致**
- [x] #677 measure：`issued/layer=3.22 useful/layer=1.95 precision=0.606 coverage=0.378`，输出与基线一致
- 落盘只在**停服时**（用户要求，不做每次会话落盘）：`stop_server_isolated.ps1` 写
  `kv_persist\.dump_request` 标记 → 发 1-token 请求制造一次 idle 转换 → scheduler 在
  `run_when_idle` 里 `capture_resident()`（把 GPU radix 中未被驱逐的叶子 D2H 到 host，带去重）
  + `flush_ssd()` → 删除标记；脚本轮询标记消失（≤300s）后再停进程
- 为什么用标记文件而不是 HTTP：`ft.exe` 主进程（API server）不加载隔离目录的
  `api_server.py`，新加的 `POST /v1/cache/dump` 在这条路径上 404；标记文件只依赖已生效的
  `scheduler.py`（worker 侧）。非冻结启动方式下 `/v1/cache/dump` 也可用
- 驱逐路径默认**不做** D2H 捕获（`FREETOKEN_KV_SWAP_CAPTURE_ON_EVICT=0`）：长上下文接近满池时
  驱逐里的同步捕获会拖慢 decode（实测旧实例退化到 44s/请求），需要边跑边存时才设 1
- dump/restore 必须按**节点段**处理：每个 radix 节点只持有自身那一段 KV（`kv_start` 标偏移），
  所以 `capture_resident` 要沿祖先链从根到叶逐段捕获，恢复端 `prefill._try_allocate_one` 要
  循环 `attempt_host_restore` 直到没有下一段。只抓叶子会只存最后一段（实测过 1088-token 残缺条目）
- **重要 bug（已修）·锁契约**：`match_req` 返回的 handle **未加锁**，而 `attempt_host_restore`
  结尾会 `unlock(old)` + `lock(new)` 迁移锁。只要 GPU 已匹配到非空前缀（重载后第二次请求必然如此），
  就会 unlock 一个 ref_count=0 的节点 → `dec_lock` 断言 `ref_count >= 0` 失败 → 调度器直接退出
  （实测 `AssertionError` + `backend worker ... exited`）。修复：`_try_allocate_one` 在进入 restore
  前先 `lock(handle)`，函数内部迁移锁，之后不再重复 lock；所有提前 `return None` 的路径都先 unlock
- **重要 bug（已修）·记忆错乱**：restore 之后 `restore_src` 仍返回**旧的** `mr.mamba_value`。KV 前缀
  被拉长到 N，但 GDN 状态只恢复到 M<N；若 GPU 命中为 0 则 `restore_src=None`，而 `pool.alloc()`
  **不清零**，live slot 里是上一个请求的残留状态 → 模型"记得"的前缀与 KV 对不上。修复：restore 后
  改取最终 handle 节点的 `mamba_value`（即恢复前缀末端的快照）
- 验证：0-1152 + 1152-1408 两段 dump 后重启，A 恢复 1152，B 在 GPU 命中 1152 的基础上再恢复 256
  （`restored=2`），无断言/无 abort，输出仍正确答出 `ZEBRA-4417` 和三个检查项
- **停服 traceback（已修）**：uvicorn 在 `capture_signals` 退出时会把捕获的 CTRL_C_EVENT 重新
  raise → 即使应用已 `Application shutdown complete`，仍打印 KeyboardInterrupt traceback。
  dump 在它之前已完成（实测 captured=9 persisted=9）。修复：`cli._run_serve` 捕获
  KeyboardInterrupt 返回 130，不再打印堆栈
- dump 的段必须**从 0 连续**：链上小于 `FREETOKEN_KV_SWAP_MIN_TOKENS`(256) 的节点会被跳过，留下
  64-token 空洞 → 恢复在空洞前停住（实测 74496 的会话只命中 40896）。dump 路径改用 `page_size`(64)
  作阈值；出现空洞后只能重算一次再 dump 补齐
- **空闲预搬运（已加）**：`run_when_idle` 每 ≥1s 调 `capture_resident(cache, max_bytes=4GiB)`
  （`FREETOKEN_KV_SWAP_PRELOAD_ON_IDLE`，默认开；`FREETOKEN_KV_SWAP_IDLE_BYTES` 可调）。会话空闲
  （如 A 在等工具返回）时把 unlocked 前缀存进 host RAM，B 进来驱逐 A 时无需在 decode 路径做同步
  D2H，A 回来直接 restore。实测：A(9159) 冷启 12.1s → preload 0.32GiB → B(10599) 12.9s →
  A 再次请求 **3.5s**（`restored 9152`）
- evict 钩子的阈值也改成 `page_size`(64)：原先用 host store 默认 512，会跳过 64..511 的节点，
  在运行时捕获路径上同样会制造断链
- **共享前缀/子智能体实测（2026-10-21）**：A、B 共享 960-token 前缀后分叉成两个分支，
  交替请求 3 轮答案各自正确（RED-11 / GREEN-22，无串扰）；B 首次只 prefill 42
  （`#cached-token: 960`，共享 KV 只算一次）；再用 13992-token 的 C 把池子压到 86% 后，
  A/B 仍命中 960（共享节点是内部节点，驱逐只回收其 mamba 槽、KV 不释放）；停服 dump 出
  `0-960`，重启 `swap-in: restored 960`，A/B 依旧答对。不足一页的分叉尾部（42 tokens）不缓存，
  每次重算，属预期
- **idle preload CUDA OOM（已修，2026-10-21）**：`capture_node` 原先按节点大小 `pin_memory=True`
  分配；生产 98304 池捕获到约 11.6GiB pinned 后耗尽 Windows 的 pin 配额 →
  `cudaErrorMemoryAllocation`，失败后 CUDA 状态受损，随后 `match_prefix` 里的 `torch.cat` 也
  OOM → 调度器退出。修复：`FREETOKEN_KV_SWAP_PIN` 默认改 0（pageable，OS 可换出）；
  `capture_node` 内部捕获异常并计 `failed`，不再冒泡到驱逐/dump 路径；idle 单次预算 4GiB→1GiB；
  出现 OOM 后置 `_idle_capture_disabled` 熔断本进程；summary 增加 `failed=`
- 实测：9664-token prompt 首次 prefill 56.9s → dump 出 `2432-9600` 段 → 重启后 4.3s
  （`restored 2432` + `restored 7168`，`#cached-token: 9600`）
- 实测：2464-token 前缀 → `host KV dump: captured=1 persisted=1`，总耗时 9.9s，新增
  `*_0_2432.pt`；重启后 `swap-in: restored 2432 tokens` + `#cached-token: 2432` + `ssd_loaded=1`
- 停机脚本另有 `-Force`（跳过 dump 直接杀）与孤儿 worker 清理
- **KV swap 完整文档（2026-10-21）**：`docs/KV_SWAP.md`（约 2040 行），含架构/数据流、
  新增 3 个文件全文、11 处修改的 before/after、环境变量表、启停与落盘流程、E2E 验证清单、
  版本升级注意事项（`.pyd` 不可改、v0.2.0 包路径、#559/#689/#700/#742 冲突点、移植顺序）、
  已知坑与回滚
- **脚本 `FREETOKEN_KV_SWAP_PIN` 已改 0**：`scripts/start_server_isolated.ps1:35` 与
  `start-freetoken-flash-next.ps1:26` 原先写死 "1"，会覆盖 manager 的新默认值，导致 pinned OOM
  修复失效
- `_dump_kv_to_ssd` 已加容错：`capture_resident` / `flush_ssd` 各自 try/except（dump 失败不阻断
  停服），并支持 `FREETOKEN_KV_DUMP_BYTES` 限制停服 dump 期间占用的 RAM（默认不限制，超出 host
  容量的部分自动溢出到 SSD）
- **0.2.0 Windows 可行性只读检查（2026-10-21）**：结论见 `docs/WIN_020_FEASIBILITY.md`。**重要修正**：
  官方 `FlashML-org/FreeToken-Web` 的 `beta` release 提供 Windows 预编译 wheel
  （`freetoken-0.2.0+g425dc7bf1-cp312-cp312-win_amd64.whl` 5.9 MB +
  `freetoken_kernel_cache-0.2.0+cu130...win_amd64.whl` + `FreeToken-Setup-win-x64.exe`），
  解包确认**含 `_row_store.pyd` / `_pinned_tensor.pyd` / `_cpu_moe.pyd` / 9 个 ggml 变体**，
  所以**不需要 VS/CUDA**。源码 tarball 的 `setup.py` 仍写 `_row_store` Linux-only，只影响
  "从源码装"这条路径。新约束：wheel 里 `scheduler/scheduler`、`moe/offload_cache`、
  `server/api_server`、`engine/engine`、`models/weight`、`moe/host_banks` **只有 `.pyd` 没有 `.py`**，
  KV swap 补丁不能靠删 `.pyd` 回退到 `.py`，需运行时 monkeypatch 或重编 wheel；
  `scheduler/cache.py`、`scheduler/prefill.py`、`kvcache/hybrid_radix_cache.py` 仍是 `.py`，可直接改
- **FreeToken Desktop（0.2.0-beta.23）解剖（2026-10-21）**：Tauri 2.11.5 + WebView2，18.6 MB，
  安装于 `%LOCALAPPDATA%\FreeToken Desktop\`（`freetoken-desktop.exe` + `engine\install.ps1`）。
  功能不止"启动/设置/统计"：引擎安装与更新（从 `FreeToken-Web` beta 解析 wheel）、启动并监管
  `ft daemon`（daemon 再拉起 `ft serve`，带 `X-FT-Token`）、模型下载（HF）/转换/修复/基准/预热、
  记账（`/accounting/ack`、`usage.json`）、日志/指标、自动更新（`stable.json`/`beta.json`、
  `FREETOKEN_UPDATE_BASE`）。引擎本身在 `%LOCALAPPDATA%\FreeToken\`（`venv`/`uv`/`bin`）
- **胖部署坑（实测）**：`venv\pyvenv.cfg` 的 `home` 指向 `%APPDATA%\uv\python\cpython-3.12-windows-x86_64-none`
  （在 FreeToken 树外，121 MB）；`venv\Scripts\*.exe`（含 `ft.exe`）与 `bin\ft.cmd`、`ft-bin.txt`
  **内嵌绝对路径** `C:\Users\X303\AppData\Local\FreeToken\venv\Scripts\python.exe`。因此整树复制到
  不同路径/不同用户名会失效；只有"完全相同的绝对路径 + 同一用户名"才能直接复制运行，否则必须在
  目标机用本地 wheel 重装（`FREETOKEN_WHEEL`/`FREETOKEN_KERNEL_CACHE_WHEEL`，或放 `engine\dist\`；
  `FREETOKEN_UV` 指自带 `uv.exe`；VC++ 运行库装过则跳过下载；`uv venv --python 3.12` 仍需 CPython）


## 性能实测（98304 tokens，生产参数，2026-10-20）

启动参数与生产脚本一致：`--memory-ratio 0.95 --num-tokens 98304 --moe-cache-size 512
--moe-cpu-layers 0-14,34-47 --max-running-requests 2 --cuda-graph-max-bs 1 --moe-cpu-threads 16
--gpu GPU-54e...`，另加 PriorityClass High / TQDM_DISABLE=1。

| 配置 | decode 吞吐 |
|---|---|
| A: overlap 关 + prefetch 关 | 25.3 t/s |
| B: overlap 开 + prefetch 关（**当前默认**） | 26.1–27.1 t/s |
| C: overlap 开 + prefetch=measure | 18.4–19.6 t/s |
| 并发 2（B 配置，各 96 tokens） | 28.6–30.3 t/s 合计，单请求 ~15.9s |

- **#678 侧流无性能损失**（反而略快），输出与基线逐字节一致
- **`prefetch=measure` 掉约 25%**：measure 每步在预测流上重跑一次 gate（`lookahead_select`）抢 GPU 带宽；
  已改为**默认 off**，用 `-PrefetchMeasure` 开关按需开启（纯诊断）
- 98304 时 `Free memory after initialization: 0.00 GiB`（memory-ratio 0.95 吃满）；CUDA graph 捕获正常
- 98k 下驱逐极难触发（需 >98k tokens 前缀），KV swap 功能需用 `-NumTokens 8192/32768` 验证
- 风险：98k + 0 空闲显存时，若真的驱逐，`capture_node` 的分块 gather 需要少量临时显存，理论上有 OOM 可能

## 关键源码事实（阶段一设计依据）

- 目标模型 Qwen3.8 有 GDN linear attention → `cache_type = hybrid_radix`
- KV 池：`freetoken/kvcache/mha_pool.py` → `MHAKVCache._kv_buffer` shape
  `(2, num_storage_layers, num_pages, page_size, local_kv_heads, head_dim)`，纯 Python 可访问
- 页表：`engine.page_table` shape `(max_running_req+1, max_num_tokens)`，token position → KV slot
- GDN 状态池：`freetoken/kvcache/linear_state_pool.py` → `conv_states` / `recurrent_states` / `slot_states`
- Radix 树：`freetoken/kvcache/hybrid_radix_cache.py`，node._key 保存节点 token 片段，node.value 保存 KV slots，
  node.mamba_value 保存 GDN snapshot slot；`evict_full` 只驱逐 leaf，`evict_mamba` 可 tombstone 内部节点
- 调度/驱逐核心：`scheduler/cache.py`（CacheManager._allocate → evict_full/evict_mamba；_free 回 free_slots）
- 运行时 rebuild：`scheduler.py::rebuild_cache`（idle-only，会清 prefix cache）
- Qwen3.8 的 PLE 使用 linear_state_pool 的 slot_states（`ple_conv` 等）

## 阶段一实现方案（Host KV Swap / HiCache-lite）

思路：被 LRU 驱逐的 radix 节点，其 KV + GDN snapshot 先 D2H 到 pinned host 存储；下次请求在 GPU 树未命中时，
查 host 最长前缀，命中则分配页 + H2D + 重新 insert 进 radix 树。等价于 swap-out/swap-in，由显存压力驱动。

### 已完成代码

- [x] 新建 `freetoken/kv_swap/__init__.py`
- [x] 新建 `freetoken/kv_swap/manager.py`：`KVHostStore`（capture_node / lookup / take / 限额 LRU / 统计）
  - 环境变量：`FREETOKEN_KV_SWAP=1` 启用；`FREETOKEN_KV_SWAP_HOST_BYTES`（默认 16GiB）；
    `FREETOKEN_KV_SWAP_MIN_TOKENS`（默认 512）；`FREETOKEN_KV_SWAP_PIN`（默认 1）；
    `FREETOKEN_KV_SWAP_COPY_CHUNK`（默认 8192 tokens/块）
- [x] `kvcache/mha_pool.py`：新增 `gather_kv(slots)` / `scatter_kv(slots, k, v)`
- [x] `kvcache/linear_state_pool.py`：`snapshot_slot(slot)` / `restore_slot(slot, data)`
- [x] `kvcache/hybrid_radix_cache.py`：`evict_hook`（evict_full 的 leaf / evict_mamba 删除前回调）
- [x] `scheduler/cache.py`：CacheManager 持有 kv_pool/host_store；`_attach_evict_hook`；
      `attempt_host_restore()`（分配页 → H2D scatter → 恢复 GDN slot → insert 树 → 锁迁移）
- [x] `scheduler/scheduler.py`（main 版）：`create_host_store(...)` 并传入 CacheManager
- [x] `scheduler/prefill.py`：`_try_allocate_one` 在 `match_req` 后调用 restore（异常自动降级）
- [x] CPU 冒烟测试通过：`scripts/test_kv_swap_cpu.py`
- [x] py_compile + 隔离环境 import 全部通过
- [x] `scripts/start_server_isolated.ps1`（隔离启动，开关见脚本头）
- [x] `scripts/test_kv_swap_live.py`（A/B/C 长前缀填池 → replay A，观察 host swap-in）
- 日志：`host KV swap-in: restored N tokens`；idle 时 `store.summary()`

## 阶段二：MoE 优化

### #678 shared expert 侧流重叠（已完成，已实测）

- [x] `env.py`：新增 `MOE_COPY_OVERLAP`（`FREETOKEN_MOE_COPY_OVERLAP`，默认 False）
- [x] `moe/offload_cache.py`：字段 `decode_copy_overlap`、`decode_copy_stream`、per-layer fork/done 事件，
      `decode_side(layer_id)` / `join_decode_side(layer_id)`
- [x] `layers/moe.py`：`OffloadMoELayer.decode_side_applies()`
- [x] `models/qwen4_exp/moe.py`：forward 在 side stream 跑 shared gate + shared expert；
      routed 路径拿 `hidden_states.clone()`
- 启用：`FREETOKEN_MOE_COPY_OVERLAP=1`（仅 GPU decode 层；CPU/hybrid/TP>1 自动跳过）
- 本机未装 `cuda-python`，`dedicated_stream` 回退到 `torch.cuda.Stream`

### #677 router lookahead 预取（measure 模式已完成，已实测）

- [x] 新增 `moe/prefetch.py`：`ExpertPrefetcher`、`lookahead_select`（Triton）、`prefetch_count`（Triton）、
      `resolve_mode` / `default_budget` / `summarize` / `format_*`
- [x] `env.py`：`MOE_PREFETCH`(off|measure|on) / `MOE_PREFETCH_K`(16) / `MOE_PREFETCH_BUDGET`(0) / `MOE_PREFETCH_DEBUG`
- [x] `moe/offload_cache.py`：字段 `prefetch_mode`、`_init_prefetch()`、rebuild/reset 清理
- [x] `layers/moe.py`：`_lookahead` + `set_lookahead()` + `_fork_lookahead()`；
      `_decode_routed` 在 `ensure_experts` 后 fork、`copy_missing` 后 join_and_count
- [x] `models/qwen4_exp/moe.py`：`wire_router_lookahead()`；`model.py` 构建后接线
- [x] 统计输出：`scheduler.run_when_idle`（main 版）每 30s 打印 window 汇总
- 启用：`FREETOKEN_MOE_PREFETCH=measure`（只预测+计数；`on` 上游未实现，会 NotImplementedError）
- 已知限制：hybrid/CPU decode 时自动禁用并 warning；bs > 8（未捕获）跳过预测

## 阶段三：KV SSD 持久化（已完成，已实测）

- [x] 新增 `freetoken/kv_swap/ssd.py`：`SSDStore`
  - 目录 `<base>/<signature-hash>/entries/*.pt` + `index.json`（原子写）
  - 索引字段：first page / length / kv_start / sha1 / bytes / atime；按 first page 分桶
  - 容量 LRU（默认 64GiB），超限删文件
  - `lookup`：长度 + kv_start + sha1 校验 → `torch.load` → 二次全量比对
- [x] `kv_swap/manager.py`：`ssd` / `signature` / `_ssd_flushed`；`_trim()` 先落盘再删除；
      `lookup()` 内存未命中 → SSD；`flush_ssd()`（idle 时，分批 1GiB）；`summary()` 附 SSD 统计
- [x] `scheduler/scheduler.py`：`run_when_idle` 每 60s `flush_ssd()`（注意 idle 只调用一次的限制）
- [x] `scripts/test_kv_persist_cpu.py`：模拟重启 → SSD 加载 → restore 全通过
- 环境变量：`FREETOKEN_KV_PERSIST=1`、`FREETOKEN_KV_PERSIST_DIR`（默认 `./kv_persist`）、
  `FREETOKEN_KV_PERSIST_BYTES`（默认 64GiB）、`FREETOKEN_KV_PERSIST_FLUSH_BYTES`（默认 1GiB）

## 复查记录（静态检查 + 修复）

- [x] **门控**：`#678/#677` 在 `--moe-strategy offload` 下对 GPU 层仍生效（CPU 层逐层跳过），hybrid 才禁用
- [x] **数据竞争**：`#677` lookahead 读 `hidden_states` → fork 时 `clone()`
- [x] **#678 显存复用**：side 流产生的 `shared`/`gate` 需 `record_stream(main_stream)`
- [x] **SSD 恢复语义**：`lookup` 要求 `entry.kv_start == min_len`
- [x] **SSD 文件名冲突**：文件名加入 `kv_start`
- [x] **捕获期驱逐**：`capture_node` 检测 `torch.cuda.is_current_stream_capturing()` 时跳过
- [x] **SSD 刷盘分批**：`flush_ssd` 每次最多 1GiB
- [x] **SSD 满时继续捕获**：host 满且有 SSD 时先 `_make_room` 落盘最老 entry
- [x] **Triton JIT 与捕获**：`prefetch_count` 的 `n` 改为运行时参数（固定 `N_P=256`）；
      `__init__` 预热 rows∈{1,2,4,8} 的两种 kernel（bf16 + fp32 logits）
- [x] **`layers/moe.py` 用 main 版**（`bank_views` 问题根因）
- [x] 两个 CPU 测试仍通过

## 待办 / 风险

- [x] ABI 兼容已确认（见"ABI 适配结论"）
- 生产参数尚未全量实测：`--num-tokens 98304`、`--moe-cpu-layers 0-14,34-47`、`--cuda-graph-max-bs`、
  `--max-running-requests 2`、`--moe-cache-size 512`、`--gpu <uuid>`、`--cors-origins`
- `flush_ssd` 仅在"请求结束后的 idle 调用"里执行（不是周期轮询）：长空闲不刷盘；
  若要更强持久化，可在 `scheduler/prefill.py::_try_allocate_one` 里节流（30s）调用 `flush_ssd()`
- `FREETOKEN_MOE_PREFETCH=on`（真正的预取拷贝）上游未实现，当前只有 measure；
  **设 `on` 会在 `offload_cache._init_prefetch` → `resolve_mode` 抛 `NotImplementedError`，
  服务起不来**。`start-freetoken-flash-next.ps1:34` 已从 `on` 改为 `off`（诊断用 `measure`）
- `record_stream` / side stream / prefetch stream 在 CUDA graph capture 下已实测不报错（capture 成功），
  但多流拓扑正确性未做数值级验证
- `capture_node` 在驱逐时是同步 D2H，关注 prefill 首 token 延迟尖峰
- SSD `torch.load(weights_only=False)` 读取本进程写出的文件；不要将 `kv_persist` 指向不可信目录
- `engine/engine.cp312-win_amd64.pyd` 未删除 → engine.py 补丁不生效（#677 统计已改道 scheduler）
- `main_work` 完整工作树备份：`C:\Users\X303\AppData\Local\Temp\opencode\main_work\freetoken`

## 阶段四：便携胖部署包（2026-10-21，已完成并实测移动）

- 产物：`D:\FreeTokenPortable`（3.35 GB，不含模型与 `kv_persist`）
- 结构：`runtime_py`（内置 CPython 3.12.14）、`engine\{venv,uv,bin,ft-bin.txt,desktop-engine-wheels.txt,_portable_fix.py}`、
  `Desktop`（`freetoken-desktop.exe` + `engine\install.ps1`）、`setup.cmd`、`start-freetoken-flash-next.ps1`、
  `stop_server_isolated.ps1`（`$KvPersistDir` 默认 `$PSScriptRoot\kv_persist`）、`restore.cmd`、`logs`、`kv_persist`
- 环境变量安全：`setup.cmd` 首次运行把原有 12 个 `FREETOKEN_*` 值备份到 `engine\env_backup.json`（含是否原本存在、
  是否新建了 junction），`restore.cmd` 按备份逐项恢复/删除；无备份时 restore 不做任何修改（`restore.cmd force` 才硬删）
- 迁移要点：`kv_persist` 的 signature 只含 page size/dtype/shape（不含路径），可跨机器直接复用
- 关键结论（实测）：
  - `pyvenv.cfg` 的 `home` **不能写相对路径**（报 `No Python at '"..\runtime_py\python.exe'`），必须绝对；
    由 `setup.cmd` → `_portable_fix.py` 按当前根目录重写
  - `uv venv --relocatable` 生成的 `.exe` shebang 为 `#!python.exe`（相对），整棵树改名/移动后仍可用
  - `ft`/`freetoken` 控制台脚本不重装原 wheel：用 2 KB 的 `ftshim` wheel（entry_points → `freetoken.cli:main`）
    在 relocatable venv 里生成，避免覆盖已打补丁的包文件
  - 补丁**烘焙进 site-packages**（venv 的 `freetoken` 上覆盖 repo 的 `freetoken/`，并删除
    `scheduler/scheduler.pyd`、`moe/offload_cache.pyd`），不再依赖 `PYTHONPATH`；桌面端拉起的引擎也带补丁
  - 本机用户级 `PYTHONPATH`/`FREETOKEN_*` 原本为空 → 之前桌面端拉起的 serve 跑的是**未打补丁**的 venv 代码；便携包已修正
- `setup.cmd`：重写 `home` / `ft-bin.txt` / `bin\ft.cmd`；`%LOCALAPPDATA%\FreeToken` 不存在时建 junction 指向 `engine`；
  `setx` 12 个 `FREETOKEN_*`（HOME / FT_BIN / KV_SWAP / HOST_BYTES / MIN_TOKENS / PIN / COPY_CHUNK / PERSIST /
  PERSIST_DIR / PERSIST_BYTES / MOE_COPY_OVERLAP / MOE_PREFETCH=off）；`setup.cmd noenv` 只修路径
- 移动实测：`D:\FreeTokenPortable` → `D:\FTP_moved` → 跑 `setup.cmd` → `ft --version`、`import torch`、
  `kv_swap` 路径全部正确 → 改回原名再跑 `setup.cmd` 正常
- 注意：
  - 目标机首次使用、以及每次挪动目录后，都要先跑一次 `setup.cmd`
  - 不要点桌面端"安装/更新引擎"：会从 GitHub beta 拉 0.2.0 覆盖 venv
  - `FREETOKEN_KV_SWAP_HOST_BYTES=16GiB` 按本机内存设，目标机内存小须在 ps1 里调低
  - 模型 126 GB 与 `kv_persist` 未打包；ps1 顶部 `$ModelPath` 需按目标机修改
  - 本机 HKCU 环境变量已在验证后还原（`FREETOKEN*` 计数 0）

## 阶段五：按会话给 serve 日志上色（2026-10-21，待重启生效）

- 新增 `freetoken/utils/log_context.py`：`current_session` ContextVar、`session_color()`（crc32 → 256 色调色板）、
  `SessionLogMiddleware`、`install_session_log_middleware()`
- 会话 id 来源优先级：`x-session-id` / `x-ft-session` header → body 的 `user` 字段 → **第一条 user 消息指纹**
  （sha1 前 10 位，跨轮稳定：system 变化、历史增长都不影响）→ `client ip:port`；
  纯 ASGI 中间件，读 JSON body 后完整重放（不改请求内容），所以**客户端无需任何配置**
- `logger.py::ColorFormatter`：INFO/DEBUG 行用会话色（时间戳 + 正文），WARNING 及以上保留级别色（黄/红）
- 安装点：`server/access_log_filter.py::install_polling_access_log_filter()`（`api_server.py:1029`，uvicorn 启动前）
  里 patch `uvicorn.Config.load` 包一层 middleware（兼容 `uvicorn.run` 与显式 `Server`）
- 开关：`FREETOKEN_LOG_SESSION_COLOR=0` 关闭，默认开（模块级缓存）
- 验证（未重启生产）：Formatter 两种会话色不同、WARNING 不变色、middleware 三种来源正确、`Config.load` 被包且幂等、
  venv 侧 import 正常；`py_compile` 通过
- 已同步到 `C:\Users\X303\AppData\Local\FreeToken\venv\...\freetoken` 与 `D:\FreeTokenPortable\engine\venv\...\freetoken`
- 验证（body 路径）：多轮同会话（第 1 轮 vs 第 4 轮、system 变化）同色；不同首问不同色；
  Anthropic `system` + `messages`、多模态 content、body 重放完整性均通过
- 限制：scheduler/engine 线程里的日志无会话上下文（回落级别色）；两个会话首条 user 内容完全相同会撞色
- 无需改客户端；想强制指定时再传 `x-session-id`（OpenAI SDK `default_headers` / `extra_headers`）

### 2026-10-21 17:2x 关键坑：ps1 必须是 UTF-8 with BOM
- 症状：serve 起来后 PYTHONPATH 不在进程环境里（PEB 直读确认），venv 原版被加载，KV swap/SSD 全部失效。
- 根因：start-freetoken-flash-next.ps1 为 UTF-8 **无 BOM**，PowerShell 5.1 按 GBK 解码中文注释，# 让 freetoken ... （...） 行尾字节吞掉换行，下一行 $env:PYTHONPATH = ... 被并入注释，从未执行。
- 修复：脚本保存为 UTF-8 **with BOM**；用 [Parser]::ParseFile 验证第 21 行赋值可见。备份在 .bak。
- 校验方法：curl.exe -X POST http://127.0.0.1:1919/v1/cache/dump，404 = venv 原版（venv 无 kv_swap、scheduler 为 .pyd）；200/503/504 = 补丁已加载。

## 阶段六：SSD 跨重启恢复的双会话修复（2026-10-21，已完成）

### 根因
- 现象：A 会话重启后约 3.4s 恢复；B 会话回退到全量 prefill（18s）。
- 误导点：先怀疑 `cached_len` 非页对齐 / lookup 条件过严。加诊断后确认 B 的 lookup 已命中
  （`cached_len=0, entry=(0,6848)`），对齐校验也通过，异常发生在更后面。
- 真实堆栈：`store.take(entry)` -> `manager._remove` -> `list.remove(entry)` -> `HostEntry.__eq__`
  -> `RuntimeError: The size of tensor a (6400) must match the size of tensor b (6848) at non-singleton dimension 0`。
- 原因：`HostEntry` 是默认 `@dataclass`（`eq=True`），自动 `__eq__` 会对所有字段（含 `tokens`/`kv_k`/`kv_v` 张量）
  逐个比较；host 池里同时存在 A(6400) 与 B(6848) 条目时，`list.remove` 的相等性比较直接抛异常，
  恢复收尾失败并回退到全量 prefill。单会话测试因为池里只有一个长度，永远看不到这个问题。

### 修复
- `kv_swap/manager.py`：`@dataclass` -> `@dataclass(eq=False)`，`list.remove` 按对象身份比较。
- 保留改动：`manager.lookup` / `ssd.lookup` 条件为 `entry.kv_start > min_len`（允许条目回伸到 GPU 命中之前，
  由 `skip` 截断）；`cache.py::attempt_host_restore` 的 `skip` / 页对齐校验 / `kv_k[:, skip:]` 切片。
- 诊断代码已清理（probe 日志、写 traceback 到文件）。注意：`ColorFormatter` 不处理 `exc_info`，
  以后要拿堆栈需直接写文件或用标准 handler。

### 验收（跨重启冷启动，kv_persist 含 A/B 两条）
- A2 3.62s（6412 tokens；首次冷启 66s 级，修复前 27.87s）
- B2 3.40s（6892 tokens；修复前 40.82s / 18s）
- A3 3.16s / B3 3.54s；日志无 `host KV restore failed`

### 运维坑
- `Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'start-freetoken...' }`
  会匹配到执行该命令的 shell 自身，后面的 `taskkill` 循环会把自己杀掉（表现为命令无输出 /
  `ChildProcess.kill`）。过滤时应排除 `$PID`。
- 启动 serve 建议单独一条命令 `Start-Process powershell -ArgumentList ... -File <ps1>`，
  不要和 kill 循环写在同一条命令里。
## 阶段七：SSD 恢复正确性验证 + 页账兜底（2026-10-21，已完成）

### 验证结论
- KV 恢复逐元素一致：恢复后 `gather_kv(slots)` 对比 `entry.kv_k/kv_v`，`max|dK|=0 max|dV|=0`（6400 tokens）。
- mamba/GDN 恢复一致：`restore_slot` 后 `snapshot_slot` 对比 `entry.mamba`，`max|d|=0`（4 个张量）。
- 结论：SSD 缓存内容不是"记忆错乱"的来源；若错乱仍在，只能来自 handle/页表账务而非数据。

### 崩溃根因（347 页悬空）
- `RuntimeError: CacheManager integrity check failed: free_pages(39) + cache_pages(1150) != num_pages(1536)`（`cache.py::check_integrity`，idle 时调用）。
- 含义：有页既不在 `free_slots`、也不在树里（请求 abort/结束时没有归还）。
- 复现：长 prompt（6412 / 15953 tokens）中途断开（curl -m），A/B/C 组合；`kv_diag` 显示 `gap` 在 cache_req 期间会短暂 >0（`lazy_free_region` 未提交），idle 时归零。
- 未能稳定复现 347 的精确序列（多数 abort 组合 idle 后 `gap=0`）。

### 修复/兜底
- 新增 `cache.py::_tree_page_starts()`（DFS 收集树内页起点）与 `_recycle_orphan_pages()`。
- `check_integrity()` 在页账不符时不再直接杀进程：先回收"既不在 free 也不在树"的孤儿页，
  打 `logger.error_rank0("cache page accounting gap ... recycled N orphan page(s)")`，再复核；仍不符才 raise。
  注意：`run_when_idle` 只在无运行请求时调用，idle 回收是安全的（孤儿页无主）。
- `run_when_idle` 入口加 `_kv_diag("idle")`（仅 `FREETOKEN_KV_DIAG=1` 时写文件）。

### 生产配置（当前）
- `FREETOKEN_KV_RESTORE=1`（恢复已启用）、`FREETOKEN_KV_DIAG=0`、`FREETOKEN_KV_RESTORE_VERIFY=0`。
- 验收：冷启动后 A 3.93s / B 14.43s / A2 3.73s / B2 3.81s；idle 后进程与端口正常，无 error。
- 诊断文件：`C:\AI_Models\logs\kv_diag.log`（DIAG=1）、`restore_verify.log`（VERIFY=1）。

### 未决
- "记忆错乱"未复现；数值上 KV/mamba 均一致。若再出现，优先抓 `kv_diag.log` 的 `gap` 与 `restore_verify.log`。
- 347 页的精确触发序列未锁定；兜底保证不因页账杀进程，但若频繁出现 `recycled` 日志需继续追泄漏点。
## 阶段八：QSA index tiers 持久化（"记忆错乱"根因修复）

### 根因
Qwen3.8-Flash-Next 使用 QSA 压缩块稀疏注意力（index_ratio=4, indexer budget 2048）。
QSAKVCache 除 K/V 页外还有两份 per-token 状态：
- _cmp_k_buffer：压缩 index key（每 4 个 token 一行，K/V 页的 1/4 shadow）
- _rope_positions：每 token 的 mrope 三轴位置（int32）

gather_kv/scatter_kv 只搬 K/V，因此 SSD 恢复后这两份状态为 0（新建池 zero-init）。
短上下文（<8192 tokens，即 <2048 组）indexer 不触发选择性稀疏，所以之前 6400 的测试
(dK=dV=0) 完全正常；长上下文（用户 73664 tokens，18416 组）稀疏选块依据全 0 的
index key，且压缩组 rope 位置为 0，表现为“记忆错乱”。这是 restore 数据路径的缺口，
不是 KV/mamba 错位。

### 变更
- kvcache/qsa_pool.py：新增 gather_index_tiers(slots) / scatter_index_tiers(slots, cmp, rope)；
  cmp 行 = slots[::ratio] // ratio，rope 行 = slots；scatter 校验行数。
- kv_swap/manager.py：HostEntry 增加 cmp_index / ope_pos（默认 None，兼容旧条目），
  
bytes 计入；capture_node 按 chunk 收集并转 CPU。
- kv_swap/ssd.py：payload 增加 cmp_index / ope_pos；lookup 用 data.get 兼容旧文件。
- scheduler/cache.py::attempt_host_restore：scatter K/V 后按 skip//ratio、skip 截断并搬运
  index tiers；VERIFY 输出追加 max|dCmp| / max|dRope|。

### 验证（22312-token prompt，含 22272 前缀）
- 新 dump 条目 _0_22272.pt：649 MB，含 cmp_index (12, 5568, 128)、ope_pos (22272, 3)。
- restore：max|dK|=0 max|dV|=0 max|dCmp|=0 max|dRope|=0，mamba max|d|=0。
- 耗时：全量 prefill 34.94 s → restore 8.49 s；idle gap=0。

### 旧条目处理
旧条目（用户 73664/74048/74368 与 6400 测试）缺 index tiers，恢复会继续错乱，
整目录移至 kv_persist.pre-index.bak。**用户下次长会话需全量 prefill 一次**（约 2-4 分钟），
完成后按 flag dump 即生成带 index tiers 的条目，之后再恢复才正确。

### 生产配置
FREETOKEN_KV_RESTORE=1、FREETOKEN_KV_DIAG=0、FREETOKEN_KV_RESTORE_VERIFY=0。
便携包 kvcache/qsa_pool.py、kv_swap/manager.py、kv_swap/ssd.py、scheduler/cache.py 已同步（hash 一致）。

### 文档
- `docs/KV_SWAP.md` 补齐阶段八：§0 增加 QSA 状态说明、§2.2 dump 流程、§2.3 restore 步骤（修正 `kv_start <= cached_len` 与 `skip` 语义）、§4 三个开关、§6.4 长上下文校验、§8.9 已知坑、§11.9 `qsa_pool.py`、§15 全节（数据布局/兼容/验证）。
- 便携包新增 `README.md`（启动/停止/预热/KV 开关/排错），并放入 `docs\KV_SWAP.md`。

## 阶段九：升级到 FreeToken 0.2.0（scheduler 变 pyd）

### 版本
- 0.1.3+gfa0e9537d → **0.2.0+g425dc7bf1**（torch 2.11.0+cu130 → 2.14.1+cu130，Python 3.12.14 不变）
- 上游把 `scheduler/scheduler.py`、`server/api_server.py` 等改成 `.pyd`，`launch.py` 移到 `server/launch.py`，新增 `models/deepseek_v41`、`models/qwen4_exp`、`kvcache/dsv4` 等
- 全量同步：`robocopy venv\Lib\site-packages\freetoken → repo\freetoken /MIR /XD kv_swap __pycache__`（旧版残留 .py 必须删除，否则与 pyd 混用）
- 备份：`freetoken.pre-0.2.0.bak`（14 MB，回滚用）

### 补丁重打（8 个文件）
| 文件 | 变更 |
| --- | --- |
| `kvcache/mha_pool.py` | **新增** `gather_kv` / `scatter_kv`（0.2.0 的 `BaseKVCachePool` 只剩 `k_cache` / `v_cache` / `store_kv`；dump/restore 全依赖这两个） |
| `kvcache/qsa_pool.py` | `gather_index_tiers` / `scatter_index_tiers` 原样重加 |
| `kvcache/linear_state_pool.py` | `snapshot_slot` / `restore_slot` 重加 |
| `scheduler/cache.py` | `__init__` 的 `kv_pool` / `host_store` 参数、`_attach_evict_hook`、`attempt_host_restore`、`_kv_diag` / `_tree_page_starts` / `_recycle_orphan_pages`、`cache_req` / `fin` 诊断、`check_integrity` 回收、`rebuild` 重挂 hook |
| `scheduler/prefill.py` | host restore 循环插到**新版 lock 之后**（0.2.0 改成先 lock 再复核），恢复后重算 `cached_len` / `extend_len` / `estimated_size`；`mamba_value` 取最终节点 |
| `cli.py` | `KeyboardInterrupt → 130`（Windows CTRL_C_EVENT） |
| `kv_swap/runtime.py` | **新增**：monkeypatch `Scheduler.__init__`（注入 host_store）、`run_when_idle`（flag dump + idle preload + 页账）、`shutdown`（落盘）、`_dump_kv_to_ssd` |
| `scheduler/__init__.py` | 类导入后调用 `runtime.install()` |

**不再需要的旧补丁**：`message/backend.py`、`message/tokenizer.py`、`tokenizer/server.py` 的 `dump_only` 字段（0.2.0 已删除，dump 走 flag 文件）；`server/api_server.py` 的 dump endpoint（pyd，且本就不需要）。

### 0.2.0 上游行为变化（已适配）
- `CacheManager.__init__` 不再接收 `kv_pool`；由 `runtime._init` 在 `Scheduler.__init__` 后注入
- `match_req(req, max_len=)` 新增 `prefix_replay_tokens`（SWA 模型的 prompt tail 重算；QSA 为 0）
- `prefill._try_allocate_one` 改为"估算 → lock → 复核"，host restore 必须在其后
- `init_logger` 返回的 logger **没有 `error_rank0`**（改用 `logger.warning`）

### 验证（22312-token prompt，22272 前缀）
- 冷 prefill **53.08 s** → restore **5.27~6.41 s**
- `KV restore verify: max|dK|=0 max|dV|=0 skip=0 add=22272 length=22272 max|dCmp|=0 max|dRope|=0`
- `mamba restore verify: max|d|=0 keys=4`
- 重复请求（GPU 命中）5.89 s；多轮会话 prompt 90 → 109 正常递增，idle `gap=0`
- 生产配置恢复：`DIAG=0`、`RESTORE_VERIFY=0`，日志不再生成

### 运维注意
- **停服务必须杀进程树**（`ft.exe` 会派生 python worker 占 GPU）：先取 1919 的 PID 再 `taskkill /PID <pid> /T /F`；残留子进程会让下一个实例卡在 CUDA 初始化（CPU 停在 ~5 s、GPU 4.5 GB 不动、无日志）
- 诊断钩子：`kv_boot.log`（启动与前 3 次 idle 的 tier 状态），可用 `FREETOKEN_KV_BOOT_LOG` 覆盖路径
- 旧条目（0.1.3 写入）验证可恢复：22272 条目 dK/dV/dCmp/dRope 全 0；`kv_persist\index.pre-0.2.0.json` 是升级前的 index 备份
- `ft serve --help` 复核过：`--moe-strategy`（`--moe-backend` 为别名）、`--moe-cpu-layers`、`--moe-cache-size`、`--cuda-graph-max-bs`、`--num-tokens`、`--image-max-tokens` 全部仍在，ps1 参数无需改


## 阶段十：启动参数对齐官方加载器（性能调优，2026-10-22）

### 目标与结果
用户要求：自动模式、一层都不锁、KV ~103k、MoE expert cache 512、单会话，达到官方加载器的
prefill 1600+ / decode 29-31 t/s。

实测（RTX 5080 16GB，Qwen3.8-Flash-Next-NVFP4，单会话）：
- 70k token prefill：**1745 t/s**（批次稳定 ~1810 t/s）
- decode（1000 token）：**30.6 t/s**
- KV 105472 tokens（1648 页 × 64），MoE cache 512/24576，mamba 4 slots

### 关键发现（踩坑记录）
1. **`--moe-cpu-layers auto` 是 prefill 速度的决定因素**。官方日志：
   `--moe-cpu-layers auto (pin probe): pinned 48 MoE layers (63.63 GiB) in 2.4s, locked 0 []`
   —— 把 48 层 expert 权重 pin 到系统内存（`locked 0` 即"一层都不锁"），expert 搬运走 PCIe
   快路径。早期把 `--moe-cpu-layers` 整个删掉 → prefill 掉到 278 t/s（进程 WS 只有 ~10 GiB）。
   pin 后进程 WS ~70 GiB（本机 125 GiB 内存，够）。
2. **不能在启动时就设 `--moe-cache-size 512` + `--disable-moe-prefill-overlap`**。启动时禁用
   prefill overlap 只会分配**单** expert-layer buffer → prefill ~300 t/s。正确做法（与官方
   加载器一致）：启动用 auto 几何（`moe_cache_size=1234`、`prefill_overlap=True`，分配**双**
   buffer），就绪后**运行时 rebuild** 到 512；rebuild 会打印
   `Disabling MoE prefill overlap on rebuild: cache_size 512 cannot hold two whole layers...`
   并自动切换逻辑，但双 buffer 已分配并保留 → prefill 1761 t/s。
3. 结论：16 GB 上"105k KV + 512 slots + 1745 t/s prefill + 30 t/s decode"可达，前提是走
   "auto 启动 + rebuild"流程。
4. `FREETOKEN_MOE_PREFETCH` / `FREETOKEN_MOE_COPY_OVERLAP` 在 0.2.0 **已不存在**（0.1.3 遗留），
   设了无害（被忽略）。KV swap 补丁对 prefill 速度**无影响**（用官方参数 + 我们的 PYTHONPATH
   实测 2363 t/s）。

### 运行时 rebuild 端点（0.2.0 确认存在）
- `POST /v1/cache/rebuild`，body：`{"moe_cache_size":512,"num_pages":1648,"num_mamba_slots":4}`
- `GET /v1/cache/status`：返回 `geometry`（`kv_per_token=25344`、`moe_per_expert=2780160`、
  `mamba_per_slot=115642376`）、`cache_budget_bytes`、`limits`（`moe_experts`、`mamba_slots min=4`、
  `kv_tokens max`、`swa_tokens`）
- **mamba slots 下限是 4**（"能否回 0" → 不能，最小 4）
- rebuild 受 `cache_budget_bytes` 约束：`moe=512 + kv=1610 + mamba=8` 被拒
  （3.76 > 3.39 GiB，`old cache kept, still serving`）；改用 `mamba=4` 才通过

### ps1 最终形态
`start-freetoken-flash-next.ps1` 参数：
```
serve --model ... --port 1919
  --moe-strategy auto
  --moe-cpu-layers auto
  --text-model-only
  --memory-ratio 0.95
  --max-running-requests 1
  --moe-cpu-threads 32
  --host 127.0.0.1 --gpu ... --cors-origins ...
```
启动前 spawn 新增的 `kv-rebuild-after-ready.ps1`：轮询 `/health` 直到 `status=ok`，然后 POST
rebuild 到 512/1648/4，结果写 `C:\AI_Models\logs\kv_rebuild.log`。

### 验证
- 启动 → 自动 rebuild（15:45:25 `status=ok moe=512 pages=1648 mamba=4`）→ cache status 一致
- 70k prefill 1745 t/s、decode 30.6 t/s
- dump flag 正常消费，写出 1726 MB 条目（66816 前缀）


## 阶段十一：完整功能验证 + 页账 bug 修复（2026-10-22）

### 修复的 bug（真 bug）
**孤儿页 / 完整性检查杀进程**。场景：会话 A 的前缀已在 radix 树里，但它的 GDN 快照被 LRU
驱逐（`mamba_value=None`）→ `HybridRadixCache.match_prefix` 只返回「有 live snapshot 的
最深节点」，于是后续共享该前缀的会话 B 命中长度为 0 → 走 host restore 并**新分配**页；
而 `HybridRadixCache.insert` 发现节点已存在（`prefix_len == insert_len`）就**不接管**传入的
`kv_indices` → 新分配的页既不在 free 也不在树里（孤儿）→ 下一次 idle 的 `check_integrity`
报 `free_pages(1648) + cache_pages(349) != num_pages(1648)` → worker 退出、服务停摆。

修法（`scheduler/cache.py`，`attempt_host_restore`）：`insert` 的返回值第一项就是「树里已
存在的长度」，用它把未被接管的新页还回池：

```python
_prefix_len, existed = self.prefix_cache.insert(ids, full_slots, mamba_slot)
if existed:
    raise RuntimeError("host KV restore collided with an existing snapshot")
_unused = _prefix_len - cached_len if _prefix_len > cached_len else 0
if _unused > 0:
    self._free(slots[:_unused])
```

复测：原崩溃场景（A dump → 重启 → 发 A → 发 B）两段 restore `gap` 均为 0，idle `gap=0`，
服务不再退出。

### 测试矩阵（VERIFY=1 / DIAG=1 下执行）

| 编号 | 用例 | 结果 |
| --- | --- | --- |
| T1 | dump → 重启 → 同 prompt restore | ✅ `max\|dK\|=max\|dV\|=max\|dCmp\|=0`，`mamba max\|d\|=0`；14.3 s → 2.7 s |
| T2 | 共享前缀分叉（A/B 不同会话） | ✅ `add=22272`，`dK/dV/dCmp=0`，两段 `gap=0` |
| — | **页账 bug 回归** | ✅ 修复后 gap=0，不再杀进程 |
| T3 | 同会话多轮 / 会话切换 | ✅ 前缀命中（`handle>0`）；注意见「mamba slots 限制」 |
| T4 | 多会话交替 | ✅ 交替请求命中（约 1.8 s） |
| T5 | 短会话（<256 token） | ✅ 不 dump、不崩 |
| T6 | 停服 dump（`stop_server_isolated.ps1`） | ✅ SSD 条目 4 → 6（新增 649 MB + 217 MB），`Port released after graceful shutdown`，GPU 0 残留 |
| T7 | 并发 2 请求（`--max-running-requests 1`） | ✅ 串行化，均成功 |
| T8 | 页账 idle `gap=0` | ✅ 全程 idle 均为 0 |
| T9 | undo（会话回退，更短前缀） | ✅ 命中更短节点，2.4 s |
| T10 | compact（上下文压缩，新前缀） | ✅ 冷 prefill 1.85 s，不崩 |

### 关键发现：mamba slots 决定「同时可续算的会话数」
`match_prefix` 要求路径上最深的节点有 live `mamba_value`。mamba 池只有 N 个 slot（受
`cache_budget_bytes` 约束），用尽后 LRU 清掉旧会话的快照 → 其 KV 前缀（含 host/SSD 条目）
**全部无法复用**，下次请求全量 prefill。

- **slots=4**（KV 1648 页 + MoE 512 的预算下）：约 4 个会话后互相驱逐；实测 u2/ux 全部
  全量重算（~14 s / 22k）。
- **slots=12**（KV 1024 页 + MoE 512）：多会话交替复用正常（u2 2.38 s、ux 1.74 s）。

端口 `/v1/cache/status`：`geometry.limits.mamba_slots = {min:4, max:40}`；
`unit_bytes.mamba_per_slot = 115642376`（≈110 MiB）。

### 其它结论
- mamba slots 与 KV 页数、MoE cache 共享同一预算，16 GB 下需取舍（见 PATCHES.md §7）。
- `HostStore.lookup` / `SSDStore.lookup` 对 token 序列做**完全比较**（`torch.equal` / `sha1`），
  不会跨会话误命中；实测 ALPHA/UNDO 在第 47 个 token 分叉，各自只命中自己的条目。
- 曾出现一次冷 22k prefill 耗时 195 s（114 t/s）的**偶发**，同时刻 `freeRAM` 偏低；重启后
  同 prompt 恢复 1655~1694 t/s，未能复现，判断为瞬时内存/IO 压力（SSD 为 Intel P5800X，
  400 G 余量，非瓶颈）。
- `FREETOKEN_MOE_PREFETCH` / `FREETOKEN_MOE_COPY_OVERLAP` 在 0.2.0 已不存在（0.1.3 遗留），
  设置无害。

### 产出
- 修复：`scheduler/cache.py`（孤儿页）。
- 同步材料：`PATCHES.md`、`patches/*.diff`（7 个文件）、`tools/mk_patches.py`。
- 运行脚本：`start-freetoken-flash-next.ps1`、`kv-rebuild-after-ready.ps1`、
  `stop_server_isolated.ps1`。


## 阶段十二：GDN 驱逐前落盘（mamba-on-evict）+ 完整验收（2026-10-22）

### 需求
16 GB 预算下 mamba slots 只有 4 个，超过 4 个会话后旧会话的 GDN 快照被 LRU 清掉，其 KV 前缀
（含 host/SSD 条目）全部无法复用 → 只能全量 prefill。目标：把「4 slots」从硬墙变成软坡。

### 实现
1. **轻量捕获**（`kv_swap/manager.py`）：
   - `capture_node_mamba(node)`：只做 `_token_prefix` + `linear_pool.snapshot_slot()`（一次
     110 MiB D2H），**绝不调 `gather_kv` / `torch.cuda.synchronize`**（阶段七全量捕获在此处
     会同步 gather KV 并卡住 decode）。
   - 入库规则：同 token 的 KV 条目已存在且 `mamba is None` → **原地补 mamba** 并强制重写 SSD；
     否则插入 **mamba-only 条目**（`kv_k=None`）。
   - `capture_evicted_mamba(node)`：钩子入口，按驱逐风暴限流
     （`FREETOKEN_MAMBA_CAPTURE_MAX_PER_EVICT`，默认 8；0.5 s 窗口判定"同一次驱逐"），
     每次捕获打 `mamba-on-evict: <kind> N tok in X ms (evict-burst #k, total Y ms)`。
2. **钩子挂载**（`kv_swap/runtime.py`）：monkeypatch `HybridRadixCache._free_node_mamba` ——
   全仓库唯一清 `node.mamba_value` 的点，`evict_mamba`（内部 tombstone）与 `evict_full`
   （叶子）两条路径都经由它。挂载回调由 `CacheManager._attach_evict_hook` 注入
   （`FREETOKEN_KV_SWAP_MAMBA_ON_EVICT=1` 默认开）。注：原来的 `prefix_cache.evict_hook`
   属性在 0.2.0 已不存在（`hasattr` 恒 False），旧的全量捕获实际从未挂上。
3. **恢复重挂**（`scheduler/cache.py::_try_reattach_mamba`）：在 `attempt_host_restore` 之前
   调用；用 `prefix_cache._walk()` 定位最深匹配节点，若其 `mamba_value is None` 且 host 有
   `length == prefix_len` 的 mamba 条目 → `ensure_mamba_slots(1)` → `alloc` → `restore_slot`
   → 重挂 `node.mamba_value` → `match_prefix` 构造新 handle → 迁移锁。**零 KV 搬运**。
4. **隔离性**：`HostStore.lookup` / `SSDStore.lookup` 新增 `require_kv` 参数（默认 True）——
   现有 KV 恢复路径看不到 mamba-only 条目；SSD 文件名相同（`sha1(tokens)_kvstart_len.pt`），
   故 meta 加 `mamba_only` 标记，**mamba-only 不覆盖 KV 条目，KV 条目可升级 mamba-only**。
5. **rebuild 语义**：`CacheManager.rebuild` 与 `LinearStatePool.rebuild` 都会整片丢弃快照且
   不挂钩子；两处各加一行 WARNING（无条件打印 + 镜像到 `kv_rebuild_warn.log`），
   避免以后误判成 bug。
6. **watcher 修复**：`kv-rebuild-after-ready.ps1` 的 rebuild POST 改为**最多重试 12 次 ×5 s**
   （实测出现过就绪竞态 503 导致几何没设上、长 prompt 报 too long）。

### 验收（4-slot 几何 / KV 1648 页 / MoE 512）
| 项 | 结果 |
| --- | --- |
| 8 会话交替（共享 19.4k 前缀 + 300-token 唯一后缀） | 第 1 个会话回来 **1.6~1.9 s**（reattach），非全量 14~18 s |
| 日志 | `reattach ok entry=19392 add=0 gap=0`；`host KV mamba re-attach: resumed 19392 tokens` |
| VERIFY=1 | **`mamba reattach verify: max|d|=0 keys=4`**（4 次） |
| 页账 | 全程 `idle gap=0`，无 recycled 告警 |
| rebuild 行为 | `kv_rebuild_warn.log`：`rebuild drops 1 mamba snapshot(s)`；系统不崩、几何正常 |
| abort 路径 | 长请求中途断开 → 服务存活、后续短/长请求正常（S1 3.47 s）、`mambae=1 poolfree=3` 计数正常 |

### 交付
- 代码：`kv_swap/{manager,ssd,runtime}.py`、`scheduler/cache.py`、`kvcache/linear_state_pool.py`、
  `kv-rebuild-after-ready.ps1`（均已 py_compile / 语法校验）。
- 文档：本文件、`docs/KV_SWAP.md` §19、`PATCHES.md`（+`patches/*.diff` 重新生成：
  `scheduler/cache.py` 现 +711/-7）。
- 生产配置已恢复（`FREETOKEN_KV_DIAG=0`、`FREETOKEN_KV_RESTORE_VERIFY=0`），几何 1648/512/4。


## 阶段十三：多租户实测 + 最终几何（2026-10-22）

### 实测：mamba slots 的真实需求
用 3 个租户（各 ~22.3k token 独立前缀）× 2 轮交替请求实测（`live_{ALPHA,BETA,GAMMA}_r{1,2}.json`）：

| 几何 | 3 租户 × 2 轮结果 |
| --- | --- |
| KV 1648 (105472) / mamba 4 | **1/3 命中**（ALPHA/GAMMA r2 全量 15 s） |
| KV 1280 (81920) / mamba 8 | **1/3 命中**（同上） |
| KV 1024 (65536) / mamba 12 | **6/6 命中**（2.5~3.4 s） |
| **KV 1216 (77824) / mamba 10** | **6/6 命中**（2.9~3.7 s）+ 单会话 66.8k 1783 t/s |

**结论**：`prefill.py` 的准入为每个活跃请求占 **6 个 slot**（3 个工作 + 引擎强制预留 3 free），
再加每个"历史会话"1 个。所以 mamba 8 只剩 2 个历史名额（3 租户不够），mamba 12 有 6 个。
`mamba-on-evict` 在槽位这么紧时救不回来（恢复还得现场占 1 个 slot），因此**多租户的根本解法是
把 `num_mamba_slots` 加到 `6 + 历史会话数`**。

### 预算公式（本机 budget ≈2.96 GiB，约束 KV+mamba）
`num_pages*1622016 + num_mamba_slots*115642376 <= 3177893888`
- 1152 页 + 12 slots 差 0.01 GiB 被拒；1152 + 11 通过；1216 + 10 通过（最优）。
- 缩小 KV 时 rebuild 的检查基于"当前预算"而非释放后的值，必要时分两步（先降后升）。

### 最终几何（已写入 `kv-rebuild-after-ready.ps1`）
`{"moe_cache_size":512,"num_pages":1216,"num_mamba_slots":10}` = **KV 77824 token / MoE 512 /
mamba 10**：单会话 66.8k 冷 prefill 1783 t/s，3 租户多轮全部复用。

### 重要运维发现：系统内存压力会拖慢 prefill
同一条 66.8k 冷 prompt：`freeRAM=23.4 GiB` 时 110.3 s（606 t/s），`freeRAM=39.6 GiB`（重启后）
时 37.49 s（1783 t/s）。pin 的 MoE banks（~63 GiB）+ host KV store 累积会把系统内存压到分页区。
**重启即恢复**；也解释了此前一次 196 s 的偶发。长期运行建议定期重启。


## 阶段十四：三个待答问题的实测与代码级答复（2026-10-22）

### Q2b「每请求占 6 slot」的构成与生命周期 —— 修正为准确模型
上一轮"6"是从 `10-4` 倒推的经验值，代码上不准确。实际：
- **运行期固定 3 个槽**：`req.linear_slot_idx`（1 个 live）+ `req.mamba_ping_pong`（**2 个**，交替当作
  下一个 track 目标槽）。见 `core.py:54-55`、`prefill.py:260-262`。
- **准入的 3**（`prefill.py:123` `if pool.num_free_slots < 3: ensure_mamba_slots(3)`）是**瞬态检查**
  —— 只在准入那一刻保证"有 3 个可用"，**不是持续预留**；请求分配后 free 立即减少，结束即还回。
- **真正的额外占用 = 路径 donated 快照**：每跨一个 CHUNK 对齐边界就
  `insert(..., frozen)` 把 1 个 ping-pong 槽捐给树，随后立即 `alloc(1)` 补回（`cache.py:938-959`），
  所以请求仍只占 3，但**树沿路径留下多个快照**（重复前缀走 dedup）。
- **准确公式**：`mamba_slots ≳ 3(运行) + 峰值路径快照数 + 余量`，路径快照数 ∝ prompt/chunk 边界数。
  这就是 8 个 19.7k 会话要 10+ 槽的原因（不是"每会话 6"）。
- **优化空间**：① 路径快照可"只留最深 + 最近 N 个"；② 物理槽总数由 `linear_state_cache_ratio`
  （默认 2.0，未暴露 CLI，即 `4*mr + max(4, ratio*mr) + 1`）决定；③ 准入的 3 是精确下限，无冗余。

### 确认：mamba-only 条目被 `evict_full` unlink 后靠什么清理 —— 原本没有，已修
- `_captured`（`manager.py:106`）与 `_mamba_captured`（:123）**原先只有 `.add()`，全仓库无 `.discard()`**。
- host 条目 `_spill` 到 SSD 时数据还在（`_captured` 保留合理）；但 **`SSDStore._drop`（容量驱逐）
  删掉文件后 store 完全不知情** → `_captured` 永久挡住 `capture_node`/`capture_node_mamba`
  → 同一前缀**再也无法被重新抓进 host/SSD**（数据已丢、门还锁着）。无 SSD 时 `_spill` 直接丢弃同理。
- **修复**：`SSDStore` 增 `_notify_drop(meta)` → `_drop` 时回调 `manager._on_entry_dropped(sha1)`
  → `discard` 两个集合（`manager.py` + `ssd.py`）。

### 任务 1：reattach 单测（不是死代码）
`tools/test_reattach.py`（CPU，mock pool/store + 真实 `HybridRadixCache`），按生产路径构造三要素：
1. KV 在树：`pc.insert(ids, kv, 3)` 建节点；
2. mamba tombstone：`pc._free_node_mamba(node, [])`（正是 `evict_mamba` 对共享内部节点做的事），
   并断言 `match_prefix` 随即返回 0；
3. host 有 mamba-only 条目：`HostEntry(kv_k=None, mamba={...})`。
结果（`max|d|` 取自"写入 slot 的快照 vs 条目快照"）：
```
host KV mamba re-attach: resumed 256 tokens (prefix 256) without copying KV
PASS reattach returned a handle            PASS node re-attached to slot 9
PASS handle resumes at 256 == 256          PASS pool.restore_slot called
PASS max|d| = 0.0 (snapshot identical)     PASS host mamba-only entry consumed
PASS match_prefix resumes again at 256
=== RESULT: PASS ===
```
顺带纠正：`HybridRadixCache.insert()` 的返回第一个分量是**插入前已存在的长度**，不是插入后长度。

### Q2a：池满时活跃会话申请扩展页的行为（77824 池，持续增长会话实测）
| 场景 | 实测 |
| --- | --- |
| A(66840) 冷 | 45.45 s（1441 t/s），占满大部分池 |
| B(7099) 独立前缀 | 10.98 s，池仅剩 ~3.9k 页 |
| **A2 = A + ~6k append**（需 ~6k 新页） | **成功 6.86 s** → 引擎按 LRU **驱逐了 B**；重发 B 变冷 11.02 s，证实驱逐 |
| prompt 超池（4×body） | **HTTP 400** `context_length_exceeded` |
| 准入时空间不足 | `_try_allocate_one` return None → **请求排队**（不报错、不自驱逐） |
| 理论风险 | 已准入的活跃请求 decode 中申请扩展页、且可驱逐节点全被 lock → `cache.py:1054` `assert`（单会话场景未触发） |

### Q2c：内存预算复盘（不接受"定期重启"）
- 总量 125.3 GiB；`python worker WS = 74 GiB`（其中 **pinned MoE banks 63.63 GiB** = 48 层 × ~1.33 GiB，
  由 `--moe-cpu-layers auto` 的 pin probe 决定；host KV 上限 16 GiB）。
- 现象复现时 `freeRAM` 23.4 GiB、`Memory Compression` 1.5 GiB、**pagefile 实际仅用 830 MB**
  → **慢不是传统分页**，而是 **63.63 GiB 锁页把文件系统 page cache 挤掉**，模型权重 mmap 缺页重读。
- **占比**：63.63 / 125.3 = **50.8%**；+host KV 16 GiB 上限 → 79.6 / 125.3 = **63.5%**。
- **可下调项与代价**：
  1. `--moe-cpu-layers auto` → 显式减少 pin 层数（如 `0-23` = 24 层 ≈ 32 GiB，省 ~32 GiB）：
     代价是那一半层 expert 走 pageable（PCIe 需 bounce/pin-unpin）→ **decode 吞吐下降**（需实测，
     估计两位数百分比）。**收益**：page cache 空间大增，长 prefill 恢复满速。
  2. `FREETOKEN_KV_SWAP_HOST_BYTES` 16 → 8 GiB（省 8 GiB）：代价是 host 承载前缀减半、更早溢到
     SSD，恢复走磁盘（NVMe 尚可）。
  3. `FREETOKEN_KV_PERSIST_BYTES`（SSD，64 GiB）不占 RAM，无需调。
- **建议**：优先调 (1)；若仍紧再降 (2)。两项都要配对 decode/prefill 的对照实测再定，
  已列为后续任务（不并入本轮）。


## 阶段十五：assert 硬化 + Q2c 三档内存/性能对照（2026-10-22）

### 1. `cache.py` 的 KV 分配 assert 硬化
原 `_allocate` 末尾：`assert len(self.free_slots) >= needed_pages, "Eviction did not free enough
space."` —— 进程会因 assert 直接死。改为：
- 新增 `class KVAllocationError(RuntimeError)`；
- 驱逐后仍不足时，**最后手段**调 `_recycle_orphan_pages()` 回收孤儿页（本补丁已有）；
- 仍不足才 `raise KVAllocationError(...)`（带 need/have/recycled/total 诊断）；
- `allocate_paged` 捕获后 `logger.error("KV allocation failed for a batch of N request(s): …")`
  再 re-raise。
- **诚实的边界**：`allocate_paged` 由 pyd（`scheduler.py`）调用，py 层无法在其上游捕获；所以若
  pyd 不捕获，效果仍是终止 worker（但异常类型明确、可被上游识别，且多了一次孤儿页回收机会）。
  准入路径（`prefill._try_allocate_one` → `attempt_host_restore` → `_allocate`）**已有** try/except
  （`prefill.py:107`），那一路是真正的"请求级"降级 → 全量 prefill。

### 2. Q2c 三档实测（KZ 修正：`--moe-cpu-layers` 是"哪些层在 CPU 计算(locked)"，不是 pin）
| 档 | `--moe-cpu-layers` | `KV_SWAP_HOST_BYTES` | PREFILL (66.8k) | DECODE (1k) | freeRAM(加载后) |
| --- | --- | --- | --- | --- | --- |
| **T1** | `auto`(= pin 48 层, locked 0) | 16 GiB | **1499 t/s** | **30.3 t/s** | 38.5 GiB |
| **T2** | `24`(24 层 locked 到 CPU) | 16 GiB | 1204 t/s | **14.1 t/s** | 38.7 GiB |
| **T3** | `24` | 8 GiB | 1224 t/s | **9.5 t/s** | 38.8 GiB |
- **判定**：
  * **T1（auto）是唯一不牺牲 decode 的配置** —— pinned MoE banks（63.63 GiB）正是 decode 走 DMA
    直取 expert 的前提。**保留 T1**。
  * **T2**：把一半层挪到 CPU 计算可省约 37 GiB RAM（+省 GPU 显存），代价是 **decode −53%**、
    prefill −20%。仅在内存极度紧张时作应急。
  * **T3**：host KV 上限 16→8 GiB 对 prefill 无影响，decode 更低（9.5）——与 T2 的差异**超出可解释
    范围，判断为测量噪声或 host 提前溢出的 I/O 竞争**，需复测；**不作为省内存手段推荐**。
- **指标局限**：`python WS` 含 mmap 权重，`FreePhysicalMemory` 受 page cache 影响，两者都不能
  精确表示 pinned 量；三档 freeRAM 几乎相同说明该指标对本问题不敏感。要精确须用
  `Get-Counter`/`QueryWorkingSetEx`，已列为后续。
- **结论（回应"不接受定期重启"）**：内存压力的**唯一无损**对策是减少 host KV 上限（收益有限，
  因 host 未用满）或换更大 RAM 的机器；**牺牲 MoE pin 省 RAM 的代价（decode −53%）不可接受**。
  "定期重启"因此只是兜底，不是方案。

### 3. 三档切换脚本
`tools/measure_tier.ps1`（prefill+decode+mem 测量）、`tools/set_tier.py`（T1/T2/T3 切换 ps1 参数）。


## 阶段十六：uptime watcher + 重启阈值线 + 下一议题（2026-10-22）

### uptime watcher（`tools/uptime_watch.ps1`）
30 分钟粒度采样，写 `C:\AI_Models\logs\uptime_watch.csv`，列：
`timestamp,uptime_min,freeRAM_GiB,prefill_tps,gpu_MiB,status`。
- 探针：每次用**随机前缀 + ~7k token 固定正文**（保证冷 prefill，能反映真实吞吐），`max_tokens=1`；
  它留下的 KV 会像普通已完成请求一样被驱逐，不额外泄漏。
- 启动：`Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
  '-File','<repo>\tools\uptime_watch.ps1','-IntervalMinutes','30' -WindowStyle Hidden`
- 停止：`Get-CimInstance Win32_Process | ? { $_.CommandLine -match 'uptime_watch' } |
  % { Stop-Process -Id $_.ProcessId -Force }`
- 验证模式：`-IntervalMinutes 1 -StopAfter N`。
首次采样（验证，2 条）：`freeRAM 41.9→1887 t/s`、`41.1→1985 t/s`。

### 重启阈值线（据已知两点定）
长 prompt（66.8k）对照（阶段十三）：
| freeRAM | 66.8k prefill | 相对 |
| --- | --- | --- |
| 39.6 GiB | 1783 t/s | 基准 |
| 23.4 GiB | 606 t/s | **−66%** |
判定线：
- **健康**：freeRAM > 35 GiB，7k prefill ≈ 1900 t/s
- **警戒**：freeRAM 28~35 GiB（长 prompt prefill 开始下滑）
- **重启线：freeRAM < 28 GiB 或 7k prefill < 1200 t/s（−37%）** → 走 `stop_server_isolated.ps1`
  （先落盘）再重启。理由：63.63 GiB locked pages 把文件系统 page cache 挤掉后，权重 mmap 缺页
  重读导致 prefill 掉 3 倍；重启即恢复（阶段十三实测 606→1783 t/s）。

### 下一议题（列出待办，本轮不做）
**「单会话 77824 上限 vs 阶段十选型」**
- 现状：KV 1216 页 = 77824 token + mamba 10。真实会话已 73664 且持续增长，**余量仅 ~4k**。
- 阶段十选型：KV 1648 页 = 105472 token + mamba 4（单会话余量大，但多会话只有 1 个历史名额）。
- 约束：本机 `cache_budget_bytes` 只包 `KV + mamba`（≈2.96 GiB），二者线性互斥：
  `num_pages*1622016 + mamba*115642376 ≤ 3177893888`。
- 待决：是否引入"按场景切换几何"（启动参数 / 客户端设置：长文档模式 ↔ 多会话模式），
  还是固定折中（如 KV ~90k + mamba 8）。需要真实工作负载的会话长度分布数据再拍板。


## 阶段十七：定位并解决 compact/长上下文掉速（根因＝mamba-on-evict 钩子）（2026-10-23）

### 现象
- 用户长会话（prompt 80k~97k）：**compact（新前缀冷 prefill）掉到 ~350 t/s**；**稳态 decode 掉到 17.5 / 6.9 t/s**（0.3.1 同配置 24~30，0.2.0 原版 30）。
- worker **持续占 ~2 个核（CPU 83%）**，GPU 100%，服务不崩。

### 关键证据（`C:\AI_Models\logs\mamba_evict.log`，新加的文件日志）
在 compact 的两秒内（10:52:01–02）触发 **3 个 burst、16 次 `capture_evicted_mamba`**，每 burst 5~6 个节点、单次 0~32 ms、burst 累计 **64~139 ms**：
```
10:52:01 store 40896 tok 0.0 ms burst#1
10:52:01 store 58368 tok 16.0 ms burst#2
...
10:52:02 store 75264 tok 31.0 ms burst#6 total 139.0 ms
```
机制：compact/长前缀 ⇒ KV 池大量换页 ⇒ 驱逐连片 ⇒ **钩子每驱逐一个节点就同步拷 110 MiB GDN 快照**，直接串在热路径上。

### 一次性拆分验证（干净基线）
配置：`103040 KV / mamba 4 / moe 512` ＋ **`FREETOKEN_KV_SWAP_MAMBA_ON_EVICT=0`**（= 用户 0.3.1 等价，仅引擎为 0.2.0）
| 指标 | 修复前 | 修复后 |
| --- | --- | --- |
| 冷 prefill 75.6k | ~350（compact）/1764 | **1794 t/s** |
| 模拟 compact 97.9k 新前缀 | **350** | **1814 t/s** |
| 长上下文 decode 75.6k | 17.5 / 6.9 | **29.4 t/s** |
| CPU load | 83%（~2 核） | **16%** |
| `mamba_evict.log` | 密集 | **无** |

### 结论（修正前面判断）
- **不是 mamba 槽位不够**：mamba 4 完全够用，之前"要加到 9"是被钩子带偏。
- **不需要牺牲 KV**：KV 回到 103040，decode 仍 29.4。
- **根因只有一个**：`mamba-on-evict` 钩子在驱逐现场做 110 MiB 同步 D2H。

### 最终配置（C+D 双盘已同步）
`103040 KV / mamba 4 / moe 512`，`FREETOKEN_KV_SWAP_MAMBA_ON_EVICT=0`，
`FREETOKEN_KV_SWAP_SUPPRESS_ON_COLD_PREFILL=1`（保险，见下）。

### 保留的保险：冷大前缀自动静默（阶段 A）
新增 `KVHostStore.suppress_for()/is_suppressed()`；`prefill._try_allocate_one` 在
`cached_len == 0 且 input_len >= FREETOKEN_KV_SUPPRESS_MIN_TOKENS(默认 16384)` 时开一个
`input_len/1000*1.5+10` 秒的抑制窗口，期间 `capture_node/capture_node_mamba/capture_evicted_mamba/
capture_resident/attempt_host_restore/_try_reattach_mamba` 全部跳过。
钩子当前已关（默认），该保险用于"将来若重新开启钩子"时保护 compact。


## 阶段十八：审计整改与最终收敛（2026-10-23）

### 五条整改
1. **① memory-ratio**：0.92 **不可用** —— 它是"引擎可用显存上限"，降到 0.92 把 cache 预算从
   **4.36 GiB 压到 3.27 GiB** → `90k+512+mamba5`（needs 3.50）被拒 → 退回启动 auto 几何
   （`moe 1065`，吃显存）→ 冷 prefill 掉到 231、free 仅 214 MiB。**已回 0.95**。
2. **③ suppress 启发式物理删除**：删掉 `KVHostStore.suppress_for/is_suppressed` + 4 个捕获
   入口守卫 + `prefill` 冷前缀触发块 + `cache` 两处 restore/reattach 守卫 + ps1 的
   `FREETOKEN_KV_SWAP_SUPPRESS_ON_COLD_PREFILL`；全仓 `suppress_*` 残留 **0**；C/D 同步。
3. **⑤ 小修三条**：(a) `_kv_diag` 的 env 读取改模块级 `_KV_DIAG_ENABLED`（不再每次调用读）；
   (b) `capture_node_mamba` 补 CUDA-graph-capture guard（与 `capture_node` 对齐）；
   (c) 负缓存清理单测 `tools/test_captured_drop.py` **PASS**（`_captured`/`_mamba_captured`
   在 SSD drop 时被清，`SSDStore._drop` 调 `_notify_drop`）。
4. **④ 三个工具**：
   - `tools/invariant_suite.py` —— 静态 API 存在性 + suppress 无残留 + 小修 + 跑内置单测
     (+ 可选 live 页账)；当前 **20 项全 PASS**；
   - `tools/check_sync.py` —— C/D 的 11 个包文件 + 5 个顶层文件 hash 比对；**建好即查出 1 处
     drift**（`kv-rebuild-after-ready.ps1` 两边不一致，已修，现 OK）；
   - **preload 计数器** —— `runtime._idle_hooks` 每 30 s 把 `store.summary()` 追加写
     `C:\AI_Models\logs\kv_preload.log`（子进程 stdout 抓不到，这是唯一可读的
     "谁在干活"信号）。
5. **① 重测（收敛）**：`90k / mamba 5 / moe 512 / ratio 0.95` → decode **28.8 / 28.6 / 28.6 /
   28.6**（4 次极差 0.2），**GPU free 恒定 1295 MiB**（不再收紧）→ **波动收敛**。

### 误删与修复（过程记录）
删除 suppress 的脚本把 4 个方法的 `def` 行一并删了（`capture_node` / `capture_node_mamba` /
`capture_evicted_mamba` / `capture_resident`），`py_compile` 不报（裸 docstring 合法），运行时
会 `NameError`。已全部恢复并 import 验证；`invariant_suite` 现在会检查这几处存在性。

### 最终配置
`KV 92160 (1440页) / mamba 5 / moe 512 / --memory-ratio 0.95`，
`FREETOKEN_KV_SWAP_MAMBA_ON_EVICT=0`（钩子关）、`FREETOKEN_KV_SWAP_SUPPRESS_ON_COLD_PREFILL`
**已删除**。C/D 两盘一致（check_sync OK）。
