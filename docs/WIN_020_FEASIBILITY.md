# FreeToken 0.2.0 在 Windows 原生运行的可行性检查（只读）

> 日期：2026-10-21
> 检查对象：`FlashML-org/FreeToken` tag `v0.2.0`（`version.py` = `0.2.0`）
> 检查方式：**只读**。源码下载到 `C:\Users\X303\AppData\Local\Temp\opencode\ft020\FreeToken-0.2.0`，未安装、未编译、未改动现有 0.1.3 环境。
> 结论：**可行**。官方 `FreeToken-Web` 的 `beta` release 已提供 Windows 预编译 wheel，**不需要** VS / CUDA Toolkit，也不需要自己补 C++。

---

## 0.0 重要修正（2026-10-21，推翻下文 §3.1/§3.2 的源码构建结论）

`https://github.com/FlashML-org/FreeToken-Web/releases/tag/beta` 实际提供：

| 资产 | 说明 |
|---|---|
| `freetoken-0.2.0+g425dc7bf1-cp312-cp312-win_amd64.whl`（5.9 MB） | 运行时，含全部 `.pyd` |
| `freetoken_kernel_cache-0.2.0+cu130.g425dc7bf1-py3-none-win_amd64.whl` | Triton kernel cache |
| `FreeToken-Setup-win-x64.exe` | 桌面端安装包 |

下载并解包 `freetoken-0.2.0+g425dc7bf1-cp312-cp312-win_amd64.whl` 后确认，它**包含**
`freetoken/kernel/_row_store.cp312-win_amd64.pyd`（309,760 B）、`_pinned_tensor`、`_cpu_moe`、
`kernel/ggml_cpu/_*.pyd`（9 个 ISA 变体）。也就是说 Windows 轮子是在另一棵树里构建的，
**不是**本文 §3.1 分析的那个 `v0.2.0` tarball 的 `setup.py`（那份仍写着 `_row_store` Linux-only）。

因此：

- 普通安装（推荐）：`pip install` 上述 wheel + 依赖，**无需** MSVC、nvcc、CUDA Toolkit。
- §3.1 的 `_row_store` Windows 后端、§3.2 的 MSVC 参数修正，**只在"从 v0.2.0 tarball 源码安装"时才需要**。
- 本文 §3.3 的构建环境清单同样只适用于源码构建路径。

**新的主要约束（KV swap 移植）**：Windows wheel 把一部分模块只编成 `.pyd`，不再附带 `.py`：

| 模块 | wheel 内形态 | 能否按 0.1.3 的方式改 `.py` |
|---|---|---|
| `scheduler/cache.py` | `.py` ✅ | 能 |
| `scheduler/prefill.py` | `.py` ✅ | 能 |
| `kvcache/hybrid_radix_cache.py` | `.py` ✅ | 能 |
| `scheduler/scheduler` | **仅 `.pyd`** ❌ | 不能；需运行时 monkeypatch 或重编 wheel |
| `moe/offload_cache` | **仅 `.pyd`** ❌ | 同上 |
| `server/api_server` | **仅 `.pyd`** ❌ | 同上 |
| `engine/engine` | **仅 `.pyd`** ❌ | 同上 |
| `models/weight` / `moe/host_banks` / `moe/expert_banks` | **仅 `.pyd`** ❌ | 同上 |

`ft serve` 在 0.2.0 wheel 上可以正常跑；受影响的是我们自己的 KV swap 补丁的**落点**。

---

## 0. 结论摘要

| 项 | 结论 |
|---|---|
| `ft serve` 能否在 Windows 原生跑 | 能，Python 层无 POSIX 硬依赖 |
| 你的模型（Qwen3.8-Flash-Next / `qwen4_exp`）能否加载 | **能**：官方 beta wheel 自带 `_row_store.pyd`（见 §0.0） |
| 是否必须实现 Windows C++ | **不必**（用预编译 wheel）；仅源码 tarball 构建才需要 |
| 构建工具链 | 预编译路径不需要；源码路径需要 MSVC 2022 + CUDA 13.0 + ninja |
| 依赖是否齐 | 齐：torch 2.14.1 win_amd64、triton-windows 3.8.0.post29、flashlib 0.4.0 均已确认存在 |
| KV swap 补丁移植 | 接口基本保留，但 4 个关键模块只有 `.pyd`，需 monkeypatch 或重编 |
| 官方支持 | 不支持。CI 只发 manylinux；Windows wheel 由 `FreeToken-Web` beta 渠道单独发布 |

---

## 1. 检查范围

扫描了 `python/freetoken/` 全部 476 个 `.py`，以及 `kernel/csrc/` 108 个 C/C++/CUDA 文件：

1. 平台分支：`win32` / `os.name` / `sys.platform` / `platform.system`
2. POSIX-only API：`fcntl` / `resource` / `pwd` / `grp` / `SIGKILL` / `setsid` / `killpg` / `fork` / `epoll` / `AF_UNIX` / `O_DIRECT` / `preadv` / `posix_fadvise`
3. 构建链：`setup.py`、`freetoken-kernel-cache/build_backend.py`、`kernel/_toolchain.py`
4. 运行时依赖：PyPI 上 torch / triton-windows / flashlib 的 wheel 与 marker
5. KV swap 需要的接口在 0.2.0 是否还存在

---

## 2. 已验证可用（无需修改）

### 2.1 Python 层保留 Windows 分支

| 位置 | 内容 | 作用 |
|---|---|---|
| `moe/host_banks.py:320` | `_wddm()` 返回 `sys.platform == "win32" or WSL` | 显式识别 **native Windows**，pinned 探测留 3 GiB 余量 |
| `moe/host_banks.py:217-240` | `_os_lock` 用 `resource`/`mlock` | 调用点 `HostBank.lock()` 外层 `except (OSError, ImportError)` → Windows 上**降级为 pageable**，不崩 |
| `moe/expert_banks.py:27` | `_PARALLEL_READER_SUPPORTED = hasattr(os,"O_DIRECT") and hasattr(os,"preadv")` | Windows 为 False → 自动走 serial 读 |
| `moe/expert_banks.py:391-402` | 同上 guard | `parallel` 被强制 False |
| `checkpoint/ftw.py:237-243` | `getattr(os, "O_DIRECT", 0)` | 无 O_DIRECT → mmap fallback |
| `kernel/fla/utils.py:45` | `sys.platform == "win32"` 分支 | FLA 内核 Windows 适配 |
| `kernel/csrc/cpu_moe/gpu_sync_cuda.cpp:8-11` | `#if defined(_WIN32) → windows.h` | `_cpu_moe` 可编译 |
| `ggml/**` | 自带 `_WIN32` | ggml-cpu 可编译 |
| `launch.py:927` | `os.name == "nt"` + `.exe` | 启动器 |
| `gpu_select.py:80` | `os.name == "nt"` | GPU 选择 |

### 2.2 `ft serve` 不碰 daemon 的 POSIX 代码

`cli.py:58-61` 的 `_run_daemon` 是**延迟 import**；`ft serve` 只 import `freetoken.server`。因此以下断点只影响 `ft daemon`，不影响推理服务：

- `daemon/pidfile.py:29` `import fcntl`
- `daemon/osproc.py:124` `os.killpg`
- `daemon/server.py:66-82` `os.setsid`

### 2.3 依赖矩阵（PyPI 实测）

| 包 | 0.2.0 要求 | Windows 可用版本 | 备注 |
|---|---|---|---|
| torch | `>=2.14,<2.15` | `2.14.1` 有 `cp312-win_amd64.whl`（cu130） | ✅ |
| triton | `>=3.6,<3.9`，marker 仅 Linux | `triton-windows 3.8.0.post29` | ✅ 需手动装 |
| flashlib | `==0.4.0` | `flashlib-0.4.0-py3-none-any.whl`，marker 显式含 Windows | ✅ |
| transformers | `>=5.18,<5.19` | 纯 Python | ✅ |
| numpy | `>=2.0,<3` | 有 win wheel | ✅ |
| `sglang-kernel` / `flashinfer` | 可选 `accel` | 非必需 | 不装即走 Triton 回退 |

> `pyproject.toml` 把 `triton` 标成 `platform_system == 'Linux'`，Windows 上 `pip install .` **不会**装 triton。必须手动 `pip install triton-windows==3.8.0.post29`，或在本地改 marker。

---

## 3. 硬断点（不修则跑不起来）

### 3.1 `_row_store` 只对 Linux 编译 —— 主断点

> **注意：本节只适用于"从 v0.2.0 tarball 源码安装"。** 官方 beta Windows wheel 已包含
> `_row_store.cp312-win_amd64.pyd`，用 wheel 安装时本节整段可跳过（见 §0.0）。

**证据**

- `setup.py` 的 `ext_modules`：
  ```python
  # disk-backed row store (PLE / Engram tables); Linux-only until the TableFile/BatchReader seams grow Windows bodies
  *([
      CppExtension(name="freetoken.kernel._row_store", sources=[...])
  ] if sys.platform == "linux" else []),
  ```
- `kernel/row_store.py:16` 直接 `from freetoken.kernel import _row_store`，无 try/except。
- `kernel/csrc/row_store/row_store_ext.cpp:10-11` 注释已写计划：
  ```cpp
  // Platform seams: TableFile (O_DIRECT+pread; Win: NO_BUFFERING), BatchReader (io_uring,
  // pread-pool fallback = the portable shape), cumemop_* (dlopen libcuda; Win: nvcuda).
  ```
  但文件里**没有任何 `_WIN32` 分支**，只有 `dlfcn.h` / `fcntl.h` / `unistd.h` / `linux/io_uring.h` / `sys/mman.h` / `sys/syscall.h` / `posix_memalign` / `posix_fadvise`。

**为什么必须修**

- `engine/config.py:36`：`ple_backend: str = "disk"`（默认）。
- `models/qwen4_exp/model.py:173`：`if engine_config.ple_backend == "disk": from .ple_disk import DiskRowTable`。
- `models/qwen4_exp/ple_disk.py:19`：`from freetoken.kernel.pinned import alloc_pinned_tensor`；PLE 表由 `kernel/row_store.py` 的 `PleStore` 提供。
- Qwen3.8-Flash-Next 的 checkpoint 含 PLE/ngram 表（`_PLE_TABLE_INFIX = ".ple.ple_embedding.ngram_embedding."`），所以这条路径必走。

Windows 上的实际报错会是：

```
ModuleNotFoundError: No module named 'freetoken.kernel._row_store'
```

**修法（二选一）**

#### 方案 A（推荐）：给 `row_store_ext.cpp` 补 Windows 后端

需要实现三个 seam，其余算法代码可复用：

1. **`TableFile`**
   - `CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr)`
   - 对齐要求与 O_DIRECT 相同（4096）；用 `ReadFile` + `OVERLAPPED.Offset/OffsetHigh` 做定位读
   - 缓冲对齐用 `_aligned_malloc` / `_aligned_free`
   - `posix_fadvise` 用 `FILE_FLAG_NO_BUFFERING` 本身替代；`drop_cache` 可忽略
2. **`BatchReader`**
   - 无 io_uring → 直接走已有的 "pread-pool fallback" 形状，但把 `pread` 换成 `ReadFile(OVERLAPPED)`
   - 每个 worker 一个 `HANDLE`（或 `DuplicateHandle`）+ 自己的 `OVERLAPPED`
   - 保留 `#ifdef _WIN32` 分支，Linux 继续走 io_uring
3. **`cumemop_*`**
   - `dlopen("libcuda.so.1")` → `LoadLibraryW(L"nvcuda.dll")`；`dlsym` → `GetProcAddress`
   - `cuStreamWriteValue32` / `cuStreamWaitValue32` / `cuMemOp` 的语义不变

预计 200–300 行 C++（大部分是 Windows 文件 IO 样板），2–4 天含调试。

#### 方案 B（应急，不推荐）：改走 pinned backend

`--ple-backend pinned` 会调用 `models/qwen4_exp/weight.py:380 load_ple_table` → `read_range_into` → `os.O_DIRECT`（Windows 不存在）。

需要：

1. 在 `moe/host_banks.py` 的 `read_range_into` / `read_file_into` 里加非 O_DIRECT 分支（`mmap` 切片 + `memoryview` 拷贝；Windows 没有 `os.pread`/`os.preadv`）；
2. 接受 **47.7 GiB fp8 表全部驻留内存**（bf16 是 95.4 GiB），且 WDDM 下 pin 会失败降级 pageable；
3. 启动时间显著变长。

只适合"先验证非 PLE 路径"，不作为长期方案。

### 3.2 `setup.py` 的 MSVC 参数

`setup.py` 的编译参数只有 ROCm / CUDA 两分支，没有 `win32`：

```python
extra_compile = ["-O3", "-std=c++20"]
device_compile = ["-O3", "-std=c++20"] + [f"-gencode=arch=compute_{a},code=sm_{a}" ...]
...
extra_compile_args={"cxx": extra_compile + ["-pthread"], "nvcc": device_compile}
```

MSVC 上：

- `-O3` → `/O3` **无效**（cl 只支持 `/O1`、`/O2`、`/Ox`），报 `D8021: invalid numeric argument '/O3'`
- `-std=c++20` → cl 需要 `/std:c++20`（冒号）
- `-pthread` → 无效，报错
- `_row_store` 的 `extra_compile_args=["-O3","-std=c++20"]` 同理
- ggml 部分**已有** win32 分支（`/O2 /std:c11 /std:c++20 /D...`），可照抄

需要加：

```python
if sys.platform == "win32":
    host_compile = ["/O2", "/std:c++20", "/EHsc", "/utf-8"]
    cxx_extra = host_compile          # 不再拼 -pthread
else:
    host_compile = ["-O3", "-std=c++20"]
    cxx_extra = host_compile + ["-pthread"]
```

并把 `_pinned_tensor`、`_cpu_moe`、`_row_store` 三处 `extra_compile_args` 按平台分支。nvcc 侧保持 `-O3 -std=c++20`（nvcc 接受），但 `.cu` 里的 host 编译部分由 `-Xcompiler` 传递 MSVC 参数（torch 的 `BuildExtension` 会自动加 `/EHsc` 等，通常不用手写）。

另外：

- `_cuda_runtime_paths()` 用 `cuda_home / "lib64"` 和 `"lib"`；Windows 的 CUDA 是 `lib/x64`，需要补：
  ```python
  if sys.platform == "win32":
      library_dirs = [str(cuda_home / "lib" / "x64")]
  ```
- `runtime_lib = "cudart"` → Windows 链接 `cudart.lib`，setuptools 会自动补 `.lib`，通常可用；若失败改为 `"cudart_static"` 或显式 `cudart64_13.lib`。

### 3.3 构建环境前置

| 组件 | 版本 | 用途 |
|---|---|---|
| Visual Studio 2022 Build Tools | 含 "使用 C++ 的桌面开发" | cl.exe / link.exe |
| CUDA Toolkit | **13.0**（必须与 torch 2.14.1 的 cu130 同 major） | nvcc、cudart |
| ninja | 任意 | `BuildExtension.with_options(use_ninja=True)` |
| Python | 3.12（与 0.1.3 venv 一致） | cp312 wheel |

`kernel/_toolchain.py::check_nvcc_matches_torch` 会拒绝跨 CUDA major 的构建（可用 `FREETOKEN_ALLOW_CUDA_MISMATCH=1` 绕过，但不建议）。

---

## 4. 软断点（能跑，但降级/变慢）

| 位置 | 现象 | 影响 | 处理 |
|---|---|---|---|
| `models/weight.py:39-64` `_read_shard_odirect_parallel` | 无 guard，直接 `os.O_DIRECT` / `os.preadv` | 仅 `parallel=True` 时触发 | `expert_banks` 已在 Windows 强制 False；**不要**手动开 `--expert-read-parallel`（若存在） |
| `models/weight.py:110-112` | `os.posix_fadvise` 的 `except OSError` 不捕获 `AttributeError` | 同上路径 | 同上；建议顺手把 `except OSError` 改成 `except (OSError, AttributeError)` |
| `models/qwen4_exp/weight.py:438` `read_range_into` | 无 guard | 仅 `--ple-backend pinned` 触发 | 走默认 disk（修好 §3.1 后） |
| `moe/host_banks.py:474-560` | `read_file_into` / `read_range_into` 用 O_DIRECT | 未走默认路径 | 同 §3.1 方案 B 才需要 |
| `engine/graph.py:110` | `sys.platform != "linux"` 跳过某 graph 优化 | 可能的 CUDA graph 性能损失（非功能） | 先接受，实测后再看 |
| `models/*/weight.py` 的 serial 读 | Windows 无 O_DIRECT 并行读 | 首次加载权重变慢（估计 1.5–2×） | 接受；后续可选实现 `FILE_FLAG_NO_BUFFERING` 并行读 |
| WDDM pin 上限 | `pin_capped()` 为 True，probe 留 3 GiB 余量 | MoE host banks 可能降级 pageable | 用 `FREETOKEN_PIN_PROBE_MARGIN_GB` 调；不要设 `FREETOKEN_KV_SWAP_PIN=1` |

---

## 5. KV swap 补丁在 0.2.0 的移植点

0.2.0 **没有** `kv_swap/`，需要按 `docs/KV_SWAP.md` §7.4 重贴。好消息是依赖的接口基本都在：

| 0.1.3 补丁用到的接口 | 0.2.0 位置 | 变化 |
|---|---|---|
| `CacheManager.__init__(..., kv_pool=, host_store=)` | `scheduler/cache.py:34` | 无 `kv_pool` / `host_store` 参数，需加 |
| `match_req(req)` | `scheduler/cache.py:98` | **新增 `max_len` 参数**（prefix replay） |
| `lock` / `unlock` | `scheduler/cache.py:242/251` | 不变 |
| `insert_prefix` | `scheduler/cache.py:305` 调用 | 仍在（generic radix） |
| `hybrid_radix_cache.insert(ids, kv, mamba)` | `kvcache/hybrid_radix_cache.py:90` | 不变，且已 clone keys（#559） |
| `match_prefix` → `HybridMatch` | 同上 `:76` | 字段 `(kv_indices, cached_len, mamba_value, node)` |
| `evict_full` / `_leaves` | 同上 `:151` / `:258` | 不变 |
| `_allocate` / `_free` / `_page_to_token` | `scheduler/cache.py:591/613/617` | 不变 |
| `PrefillAdder._try_allocate_one` | `scheduler/prefill.py:70` | 结构接近，已返回 5 元组（含 `restore_src`） |
| `Scheduler.run_when_idle` | `scheduler/scheduler.py:161` | 仍在，可插 idle preload / dump flag |
| `CacheRebuildBackendMsg` | `scheduler/scheduler.py:597` | 仍在，可加 `dump_only` |
| `_pending_rebuild` | `scheduler/scheduler.py:122` | 仍在 |

**移植时必须注意的两点**

1. **`max_len` / prefix replay**：0.2.0 的 `match_req(req, max_len=req.input_len - prefix_replay_tokens)` 会故意少命中一段，让前缀重算。`attempt_host_restore` 如果无视 `max_len`，会把 replay cap 破坏掉。恢复循环要加：
   ```python
   cap = req.input_len - self.cache_manager.prefix_replay_tokens
   if cap > 0 and entry.length > cap:
       break   # 不越过 replay 边界
   ```
2. **`mamba_value` 仍取 `mr.mamba_value`**（`prefill.py:133`），不是 `handle.node.mamba_value`。0.1.3 的"记忆错乱"修复（`docs/KV_SWAP.md` §8.2）在 0.2.0 上**仍然需要**。

其余 diff（`host_store` 注入、`attempt_host_restore`、`_attach_evict_hook`、`_dump_kv_to_ssd`、message/tokenizer/api 的 `dump_only`）与 0.1.3 一致，可直接照搬；`api_server.py` 在 0.2.0 是 `.py`（无 `.pyd` 覆盖问题），停服可以改回 HTTP `/v1/cache/dump`。

---

## 6. 建议实施顺序

### P0：环境与构建（0.5–1 天）

1. 新建独立 venv（**不要动** `C:\Users\X303\AppData\Local\FreeToken\venv`）
2. 装 VS2022 Build Tools + CUDA 13.0
3. `pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130`
4. `pip install triton-windows==3.8.0.post29 flashlib==0.4.0`
5. 改 `setup.py` 的 MSVC 参数（§3.2）
6. `pip install -e <repo> --no-build-isolation`
7. 目标：`python -c "import freetoken.kernel._pinned_tensor, freetoken.kernel._cpu_moe"` 成功

### P1：先跑通非 PLE 模型（0.5 天）

用一个没有 PLE 表的小模型（或 `--ple-backend` 不触发的路径）验证 `ft serve` 能起、能推理。这一步能提前暴露 torch/triton/MSVC 的其余问题。

### P2：`_row_store` Windows 后端（2–4 天）

按 §3.1 方案 A 实现 `TableFile` / `BatchReader` / `cumemop_*` 的 `_WIN32` 分支，改 `setup.py` 允许 Windows 编译，跑通 Qwen3.8-Flash-Next 加载。

### P3：KV swap 移植 + E2E（1–2 天）

按 §5 移植，重跑 `docs/KV_SWAP.md` §6 的验证清单。

**总计 4–7 人日**，其中 C++ 是唯一不确定项。

---

## 7. 验证清单（每阶段）

- [ ] `import freetoken.kernel._pinned_tensor` / `_cpu_moe` / `_row_store` 成功
- [ ] `ft serve` 启动到 `/v1/models` 200
- [ ] 短 prompt 生成正确（与 0.1.3 同模型输出对比）
- [ ] 长 prompt（>8k）prefill 不崩
- [ ] Qwen3.8-Flash-Next 加载完成，PLE 表可查（disk backend）
- [ ] `--moe-cpu-layers` 生效，decode 吞吐不低于 0.1.3 的 70%
- [ ] KV swap：dump → 重启 → `swap-in: restored N`
- [ ] 共享前缀 A/B 交替不串扰
- [ ] 连续 1 小时 decode 无 `cudaErrorMemoryAllocation`

---

## 8. 风险与回退

| 风险 | 概率 | 缓解 |
|---|---|---|
| `row_store` Windows 实现有隐蔽 bug（对齐/短读） | 中 | 先写单元测试：随机 offset/len 读，与 `mmap` 结果逐字节比对 |
| MSVC 与 nvcc 的 ABI/flags 冲突 | 中 | 用 torch 官方 `BuildExtension` 默认参数，尽量少加自定义 flag |
| triton-windows 3.8 与 torch 2.14 不匹配 | 低 | 先跑 `triton.compile` 冒烟测试 |
| 上游后续版本再改 `row_store` 接口 | 高 | 把 Windows 后端做成独立 patch 文件，升级时重放 |
| WDDM pin 上限导致 MoE 降级、吞吐下降 | 高 | 接受 pageable；或减少 `--moe-cpu-layers` |
| 0.2.0 的 `prefix_replay_tokens` 与 KV swap 交互出错 | 中 | 移植时加 §5 的 cap 检查，E2E 覆盖长前缀 |

**回退**：整个过程在独立 venv + 独立源码目录，现有 0.1.3 生产环境与 `kv_persist` 不动；失败直接删目录，继续用 0.1.3。

---

## 9. 附：检查证据

```text
# 平台分支（7 处，全部有 Windows 语义）
python/freetoken/gpu_select.py:80            os.name == "nt"
python/freetoken/launch.py:927               os.name == "nt" and .exe
python/freetoken/engine/graph.py:110         sys.platform != "linux"  (跳过优化)
python/freetoken/kernel/fla/utils.py:45      sys.platform == "win32"
python/freetoken/moe/host_banks.py:321       sys.platform == "win32" or WSL
python/freetoken/models/deepseek_v41/attention.py:288/291  (变量名 win32，非平台判断)

# 硬断点
setup.py                                     _row_store 仅在 sys.platform == "linux" 编译
python/freetoken/kernel/row_store.py:16      from freetoken.kernel import _row_store  (无 try)
python/freetoken/kernel/csrc/row_store/row_store_ext.cpp:28-36  POSIX includes only
python/freetoken/engine/config.py:36         ple_backend = "disk"
python/freetoken/models/qwen4_exp/model.py:173  disk backend 分支
python/freetoken/models/qwen4_exp/weight.py:6   PLE 表 47.7 GiB fp8 / 95.4 GiB bf16

# 软断点
python/freetoken/models/weight.py:39-64      O_DIRECT 并行读（无 guard）
python/freetoken/models/weight.py:110-112    os.posix_fadvise + except OSError（漏 AttributeError）
python/freetoken/moe/expert_banks.py:27      _PARALLEL_READER_SUPPORTED（有 guard）
python/freetoken/checkpoint/ftw.py:243       getattr(os, "O_DIRECT", 0)（有 fallback）
python/freetoken/moe/host_banks.py:217-240   resource/mlock（外层捕获 ImportError）

# 仅 daemon 受影响
python/freetoken/daemon/pidfile.py:29        import fcntl
python/freetoken/daemon/osproc.py:124        os.killpg
python/freetoken/cli.py:58-61                daemon 延迟 import（serve 不受影响）

# KV swap 接口
python/freetoken/scheduler/cache.py:98/242/251/305/332
python/freetoken/scheduler/prefill.py:70-133
python/freetoken/kvcache/hybrid_radix_cache.py:76/90/151/258
python/freetoken/scheduler/scheduler.py:122/161/597
```

---

*本文只记录可行性检查结果，未执行任何安装、编译或升级。实施前请先确认 §3.2 的 MSVC 参数与 CUDA 13.0 工具链可用。*
