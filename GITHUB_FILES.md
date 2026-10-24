# 需要提交 GitHub 的文件清单（已按实际结构核对）

基线：`FreeToken-546c5fe.zip` = **官方源码树，版本 0.2.0（commit `546c5fe`）**。
官方顶层**存在**：`.github/ assets/ benchmarks/ docs/ freetoken-kernel-cache/ python/ scripts/ tests/`
官方顶层**不存在**（我们要新建）：`tools/`、`patches/`

我们改的 7 个文件在官方树里都是 `.py`，`git apply --check` 已全部通过。

---

## A. 覆盖官方文件（7 个）— 目标 `python/freetoken/`

| 本仓库 | 提交到（覆盖） |
| --- | --- |
| `freetoken/kvcache/mha_pool.py` | `python/freetoken/kvcache/mha_pool.py` |
| `freetoken/kvcache/qsa_pool.py` | `python/freetoken/kvcache/qsa_pool.py` |
| `freetoken/kvcache/linear_state_pool.py` | `python/freetoken/kvcache/linear_state_pool.py` |
| `freetoken/scheduler/cache.py` | `python/freetoken/scheduler/cache.py` |
| `freetoken/scheduler/prefill.py` | `python/freetoken/scheduler/prefill.py` |
| `freetoken/scheduler/__init__.py` | `python/freetoken/scheduler/__init__.py` |
| `freetoken/cli.py` | `python/freetoken/cli.py` |

等价做法：提交 `patches/` 下同名 7 个 `.diff`，应用时
`git apply -p1 --directory=python/freetoken patches/<x>.diff`。

## B. 新增目录 `python/freetoken/kv_swap/`（4 个文件）

| 本仓库 | 提交到（新建） |
| --- | --- |
| `freetoken/kv_swap/__init__.py` | `python/freetoken/kv_swap/__init__.py` |
| `freetoken/kv_swap/manager.py` | `python/freetoken/kv_swap/manager.py` |
| `freetoken/kv_swap/ssd.py` | `python/freetoken/kv_swap/ssd.py` |
| `freetoken/kv_swap/runtime.py` | `python/freetoken/kv_swap/runtime.py` |

（排除 `kv_swap/__pycache__/`）

## C. 新增脚本 / 文档 / 工具（目录多为新建）

| 本仓库 | 建议提交路径 | 备注 |
| --- | --- | --- |
| `start-freetoken-flash-next.ps1` | `scripts/start-freetoken-flash-next.ps1` | 官方 `scripts/` 是 Linux 构建脚本；Windows 启动器放此或用 `scripts/windows/` |
| `kv-rebuild-after-ready.ps1` | `scripts/kv-rebuild-after-ready.ps1` | 同上 |
| `stop_server_isolated.ps1` | `scripts/stop_server_isolated.ps1` | 本仓库顶层与该目录下**是同一份**，交一份即可 |
| `PATCHES.md` | `PATCHES.md` | 根目录 |
| `docs/KV_SWAP.md` | `docs/KV_SWAP.md` | 官方 `docs/` 已存在 |
| `patches/*.diff`（下列 7 个） | `patches/`（**新建**） | `kvcache__mha_pool.py.diff`、`kvcache__qsa_pool.py.diff`、`kvcache__linear_state_pool.py.diff`、`scheduler__cache.py.diff`、`scheduler__prefill.py.diff`、`scheduler____init__.py.diff`、`cli.py.diff` |
| `tools/mk_patches.py` | `tools/mk_patches.py`（**新建**） | 重新生成 patch 用 |
| `tools/test_reattach.py` | `tools/test_reattach.py`（**新建**） | reattach 单测 |
| `tools/measure_tier.ps1` | `tools/measure_tier.ps1`（**新建**） | 三档性能对照 |
| `tools/uptime_watch.ps1` | `tools/uptime_watch.ps1`（**新建**） | uptime/重启阈值监测 |

## D. 不要提交

| 项 | 原因 |
| --- | --- |
| `FreeToken-546c5fe.zip`（2.99 MB） | 官方源码包 |
| `freetoken/`（869 文件，官方包副本） | 走方案 A/B 的覆盖方式即可；若要整树仓库见文末 |
| `backup/`（663 文件）、`*.bak`、`020/`（空） | 备份/杂物 |
| `kv_persist/`（含 ~1.7 GB 条目）、`logs/` | 运行时数据 |
| `__pycache__/`、`*.pyc`（含 `kv_swap/__pycache__`） | 编译产物 |
| `patches/issue677.diff`、`patches/issue678.diff` | **别人的 PR**（改 `engine.py`/`benchmarks`），非本项目 |
| `scripts/auto-login.js`、`scripts/cdp-eval.js`、`scripts/scan-lines.js` | 浏览器调试脚本，与本项目无关 |
| `tools/check_tokens2.py`、`tools/mk_prompts.py`、`tools/mk_undo.py`、`tools/set_kv_flags.py` | 本地测试辅助；要留可一起放 `tools/` |
| `GITHUB_FILES.md`（本文件） | 提交清单本身，通常不需要入库 |
| `PROGRESS.md`、`PROGRESS_0.3.1.md`、`FreeToken_Hackathon_MasterPlan*.md`、`qwen38_freetoken_kv_eviction_plan.md`、`docs/WIN_020_FEASIBILITY.md` | 内部工作笔记，是否公开由你定 |

## E. 建议 `.gitignore` 追加

```
__pycache__/
*.pyc
*.bak
*.zip
kv_persist/
logs/
backup/
020/
```
