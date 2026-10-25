# FreeToken —  fork (unofficial)

&gt; **Personal fork of [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken).**
&gt; Not affiliated with or endorsed by the upstream team.
&gt; For the official project, see the upstream repository.

**Tested environment (only):**

- Hardware: RTX 5080 16G VRAM + 128G RAM, Windows 11
- Model: Qwen3.8-Flash-Next-NVFP4 (Qwen4Exp arch)

All changes in this fork are developed and verified **only** on this
combination. Other GPUs, operating systems, and model architectures are
untested — the QSA index-tiers work below is specific to the `qwen4exp`
sparse-attention path and does not apply to standard MHA/MoE models.

**Changes in this fork** (all tested only on the environment above;
see [PROGRESS.md](./PROGRESS.md) for full details):

- KV cache swap to host RAM / SSD with cross-restart session restore
- Restore-path completeness: persist QSA `_cmp_k_buffer` /
  `_rope_positions` alongside KV (long-context restore correctness)
- Mamba/GDN snapshot capture-before-evict (async pinned-staging, v1.1):
  slot eviction degrades to host/SSD restore instead of permanent session
  loss; capture cost ~5-15ms vs 47-62ms synchronous
- Session-level whole-chain snapshots: captured at every decode end (GDN+QSA
  naturally aligned) plus idle and pressure triggers; immutable version
  history (retention via host LRU, superseded snapshots demoted to the SSD
  archive); supports revert-then-resend and rollback-before-compaction
- Restore paths: whole-chain (skip=0), seg-extend partial restore below
  client divergence points (dynamic toolset / EOS re-rendering), prefix-view
  restore, mamba re-attach without blocking KV extension
- Idle-time KV preload, page-accounting hardening, session-colored logs
- MoE: shared-expert side-stream overlap (#678-style), router lookahead measure
- Windows-oriented: portable fat deployment, patched venv workflow

**Development & AI assistance:**

The changes in this fork were implemented **with substantial AI
assistance** (design discussion, code generation, and testing executed
collaboratively between the maintainer and AI coding agents). The
maintainer directs all design decisions, reviews every change, and
performs final verification on real hardware. Corrections are welcome
— and upstream is encouraged to adopt any of these changes.

**Known limitations (GDN snapshot layer):**

- Geometry changes / cache rebuilds rebuild the host store; SSD archives
  survive across geometries (the signature excludes page count) and restore
  on demand
- Restore allocates a mamba slot on demand (up to the engine cap); the
  request degrades to full prefill rather than queueing forever
- GDN far-context recall degrades with sequence length — this is
  architecture-inherent and not fixable at the serving layer
- Mamba slots and KV pages share one VRAM budget: raising the
  concurrent-session limit shortens max context length and vice versa
- Client-side prefix divergence (dynamic toolset per agent, EOS re-rendering,
  compaction rewrites) truncates reusable prefixes; seg-extend partial
  restore covers the intact head (typically 80-99%)
- Decode has a rare, not-yet-root-caused slowdown (~18 t/s @ ~71k ctx vs
  28-30 baseline) correlated with RAM headroom and WDDM memory pressure;
  under investigation via the watcher data
- VRAM accounting has three disagreeing views (CUDA / nvidia-smi / D3DKMT);
  driver-private overhead ~0.5-0.8G is invisible to nvidia-smi — plan
  headroom as nvidia-smi-free minus 0.5G

## Performance (measured on this fork)

RTX 5080 16G, Qwen3.8-Flash-Next-NVFP4, 1360-page (87,040 tok) KV pool,
mamba 4, moe 512, `--memory-ratio` 0.92-0.95:

| Scenario | Result |
|---|---|
| Cold prefill (75k, healthy RAM) | ~1783 tok/s |
| Cold prefill (RAM-pressured) | degrades to 312-1167 tok/s (restart restores) |
| Session restore after eviction | 1.6-2.5s (vs 25s+ full prefill), `max|d|=0` |
| Decode after restore | 28-31 t/s @ ~71k ctx (no penalty) |
| Endurance: 8 laps × 2×75k × 500 tok decode | 28.0-29.7 t/s, drift ≤5% |
| Restore instrumentation | `restore final: cached=71168/71222 took=937-1563ms` |
| Sub-agent burst (back-to-back, 20 steps) | zero full refills; median 12.2/12.5s per 300-tok step |

Decode t/s is always reported **with context length**; numbers without ctx
are not comparable.

## Current configuration (production)

```
--moe-strategy auto --moe-cpu-layers auto --text-model-only
--memory-ratio 0.92..0.95     # experiment: 0.92 (C) / 0.93 (D)
--num-tokens 87040            # 1360 pages × 64
--moe-cpu-threads 32 --max-running-requests 1
# watcher body: {"moe_cache_size":512,"num_pages":1360,"num_mamba_slots":4}
```

Key environment variables:

| Variable | Value | Purpose |
|---|---|---|
| `FREETOKEN_KV_SWAP_HOST_BYTES` | 24 GiB | host tier budget (probe toward the RAM threshold) |
| `FREETOKEN_KV_SWAP_MAMBA_ON_EVICT` | 1 | async GDN snapshot on eviction |
| `FREETOKEN_MAMBA_DRAIN_MAX_PER_IDLE` | 4 | drain tasks per idle cycle |
| `FREETOKEN_KV_SESS_SNAPSHOT` | 1 | whole-session snapshots (decode-end + pressure) |
| `FREETOKEN_KV_PERSIST_DIR/_BYTES` | repo dir / 64 GiB | SSD archive |
| `FREETOKEN_KV_RESTORE_VERIFY` | 0 (test: 1) | bit-exact restore verification |
| `FREETOKEN_API_KEY` | (optional) | from `.api_key` file; auth on all routes but /health |

All changes are experimental; correctness verified by automated tests
and hardware-in-the-loop measurement. Use at your own risk.

License: Apache-2.0 (same as upstream).

---
<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo-light.svg">
    <img alt="FreeToken" src="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo.svg" width=65%>
  </picture>
</div>

<p align="center">
| <a href="https://www.flashml.ai/"><b>Download</b></a> | <a href="https://arxiv.org/abs/2608.16157"><b>Paper</b></a> | <a href="https://join.slack.com/t/flashml/shared_invite/zt-3zpdh5j10-9dwTXrgLiqpVxizhA9KVbA"><b>Developer Slack</b></a> | <a href="https://discord.gg/MsA277cJzZ"><b>Community Discord</b></a> | <a href="https://github.com/FlashML-org/FreeToken/issues/482"><b>Community WeChat</b></a> |
</p>


Unlock datacenter-class intelligence on the hardware you already own — Run 290B+ frontier MoE models locally on your gaming PC at blistering interactive speeds.

## About

FreeToken is an edge-native Mixture-of-Experts (MoE) serving engine designed for running frontier-scale open-weight models on personal and consumer hardware. It treats heterogeneous edge resources—GPUs, CPUs, host memory, and interconnects—as a unified, elastic inference platform. Its core features include:  

- **Fast Edge-Native Runtime**: Provides efficient MoE serving with bandwidth-adaptive CPU–GPU co-execution ($q^\star$ policy), full-layer double-buffered prefill streaming, global LRU expert caching, graph-compatible execution, and the FTW fast weight format.  
- **Semantic-Aware Caching**: Features semantic anchor checkpoints for recurrent state and KV caches, allowing agentic context edits (e.g., tool calls, thinking blocks) to avoid redundant context recomputation.  
- **Elastic Memory Management**: Supports dynamic, runtime VRAM re-allocation between expert caches and KV memory without engine restarts or weight reloading.  
- **Broad MoE & Ecosystem Support**: Supports frontier open-weight MoE models (e.g., DeepSeek-V4-Flash, Qwen3.6-35B-A3B, GLM-5.2) across various parameter scales and quantization formats (e.g., MXFP4, NVFP4, FP8, BF16), with Anthropic/OpenAI-compatible APIs for seamless integration with real-world coding and tool-calling agents (e.g., Codex, Claude Code, OpenCode, OpenClaw, DeepSeek Harness). 
- **Diverse Consumer Hardware**: Scales across consumer laptops, gaming desktops, and workstation GPUs, with native support for NVIDIA RTX 30, RTX 40, and RTX 50 series GPUs.  

## Getting Started

### Desktop app

Download FreeToken for Windows or Linux at [flashml.ai](https://www.flashml.ai/). It sets the engine up for you and gives you a GUI for running models, chatting, and tuning the engine.

<div align="center">
  <img alt="FreeToken Desktop" src="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/desktop-console.png" width=92%>
</div>

### CLI

Install FreeToken with [uv](https://docs.astral.sh/uv/) (recommended) or pip:

```bash
uv pip install "freetoken[accel]"
```

Or build from source:

```bash
git clone https://github.com/FlashML-org/FreeToken.git && cd FreeToken
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
```

For More details:

- [Install FreeToken](https://github.com/FlashML-org/FreeToken/blob/main/docs/install.md)
- [Quick start](https://github.com/FlashML-org/FreeToken/blob/main/docs/quickstart.md)
- [Supported models](https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md)
- [CLI reference](https://github.com/FlashML-org/FreeToken/blob/main/docs/cli.md)
- [Repairing old FTW checkpoints](https://github.com/FlashML-org/FreeToken/blob/main/docs/ftw-hotfix.md)
- [Diagnosing a checkpoint that fails to load](https://github.com/FlashML-org/FreeToken/blob/main/scripts/checkpoint_diagnose/README.md)

## Citation

If you use FreeToken for your research, please cite our [paper](https://arxiv.org/abs/2608.16157):

```bibtex
@article{yang2026freetoken,
  title={FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution},
  author={Yang, Shuo and Fan, Xiaoze and Pan, Melissa and Xi, Haocheng and Wang, Zhe and Sun, Shanlin and Keutzer, Kurt and Han, Song and Zaharia, Matei and Xu, Chenfeng and Stoica, Ion},
  journal={arXiv preprint arXiv:2608.16157},
  year={2026}
}
```

## Acknowledgment

FreeToken was deeply inspired by [mini-sglang](https://github.com/sgl-project/mini-sglang), and
learned the design and reused code from the following projects:
[SGLang](https://github.com/sgl-project/sglang),
[vLLM](https://github.com/vllm-project/vllm),
[FlashInfer](https://github.com/flashinfer-ai/flashinfer),
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention),
[LightLLM](https://github.com/ModelTC/lightllm) and [llama.cpp](https://github.com/ggml-org/llama.cpp).

## License

[Apache License 2.0](https://github.com/FlashML-org/FreeToken/blob/main/LICENSE).
