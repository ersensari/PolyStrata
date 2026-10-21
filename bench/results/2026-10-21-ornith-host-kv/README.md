# Ornith pinned-host KV experiment: slower on this workload

Measured 2026-10-21. This is a local CUDA capacity experiment on top of GQA-16 attention commit
`7fb9d79d14d53d1c5061b7667e8fef37ba72fa98`, kept separate from upstream PR 63. It was not deployed to the
production service. The previous GPU-KV custom engine remains the selected production engine.

## What was implemented

`STRATA_POLY_HOST_KV=1` opts Qwen35's head-dimension-256, GQA-group-16 caches into `cudaHostAllocMapped` and
`cudaHostGetDevicePointer`: physical KV storage stays in pinned system RAM, with no device-resident copy.
Attention still computes on the GPU and reads those addresses over PCIe. The entire maximum-context cache
is pinned upfront (3536 MiB for this model's Q8 KV at 262144 context); this prototype does not allocate host
pages incrementally. Memory is tracked by the existing host-allocation owner and freed on engine destruction.
Other models/geometries, HIP and default execution retain the original allocation path.

`STRATA_POLY_KV_DIAGNOSTICS=1` reports physically mapped VRAM and covered rows on cache growth. Both variables
are off by default. This is not layerwise staging, hot/cold KV partitioning, CPU attention or a performance-ready
offload design. [NVIDIA's memory guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html)
describes why GPU accesses to mapped host allocations cross the CPU/GPU interconnect.

## Machine and setup

- One RTX 5060 Ti, 16311 MiB, host GPU 1 only; negotiated PCIe generation 5 x8 observed during inference;
  driver 595.84, CUDA 13.0.48, sm_120. GPU 0 was not used for these tests.
- Ryzen 9 9900X, 12 physical cores / 24 threads, 91 GiB RAM reported by Linux, NVMe model storage.
- Same AtomicChat Ornith AD-IQ4_XS_S 38-shard model as the preceding GQA-16 report; Q8 KV, 262144 maximum
  context, lazy GPU KV allocation (no `--kv-whole`), 12 CPU threads, 1024 MiB reserve, MTP window 8,
  lookup 3, tail skip 0, greedy sampling, CUDA graphs and both attention opt-ins on.
- Expert swaps disabled for comparison. An identical operator profile snapshot was restored before each new
  process; seed SHA256 `1e86817cbb8be4e25ec00a04dbb3504963146fe54b75dcb2eb01127020ca7799`.
  The operator's profile is not included. Loading is excluded from throughput; OS page cache retained.
- Production inference was stopped during the measurements. Other host services remained running.
- Experimental executable SHA256 `fe2cce331c86841a1c69cdd5980eba8b6ffd0fe928c5af2472528f43c46311a6`.
  Reference executable SHA256 `bcba061330f6ba51718a9c1adc5333d2ec4e4c008b4194a77b176247f78b5bbd`.

## Results

Synthetic coding prompt and repeated-word context are specified in `compare.py`; each timed response generated
256 tokens. Short prompt: 41 input tokens, three alternating rounds (GPU/host, host/GPU, GPU/host). Long prompt:
7246 input tokens, one GPU/host pair. These are not the operator's private UI conversation. In a preceding UI
request the server reported 581 generated tokens, 24.4 tok/s and 58.1% cache hits; the synthetic coding prompt
has different expert demand, so compare the arms within this report, not its raw speed with that UI request.

| Q8 KV storage | Short decode tok/s median (range), 3 runs | Long decode tok/s, 1 run | Long prompt tok/s, 1 run |
| --- | --- | ---: | ---: |
| GPU | 28.41 (25.64–28.67) | 29.04 | 790.7 |
| pinned host RAM | 19.90 (17.85–21.96) | 18.69 | 624.1 |

Short decode was about 30% slower with pinned host KV. The GPU cache initially maps **104 MiB** for 4096 rows;
the long run grows it to **208 MiB** covering 12288 rows, including allocation granularity and lookahead.
Moving that KV off the card adds just **20 experts at load**, 1277 → 1297 of 12544 (6.1 → 6.2 GiB rounded).
The host allocation's 3536 MiB is the full 256K capacity, not the amount of VRAM saved on the short prompt.

Short-run expert hit percentages: GPU 12.2/12.8/12.5; host 12.9/13.0/12.8. The engine's reported CPU-expert
ms/token: GPU 31.8/29.7/29.1; host 42.1/37.3/44.4. These counters include the engine's token accounting and
overlapped work; they are not an independent wall-clock breakdown. Pinned RAM also competes with the model's
file cache, so this is not an isolated PCIe microbenchmark. Output IDs and draft acceptance vary across arms
and even across ordinary GPU runs; the performance result is not a model-quality equivalence test.

## Checks and limits

- CUDA `strata-poly` builds. Both fast attention paths remain active with host KV. MTP/graphs and Q8 KV ran
  successfully in all timed arms. Default-path no-draft, frozen-placement check: reference and patched builds
  with host KV unset produced identical 256-token outputs, SHA256
  `c0a044a7acf9c67f448cfb2ba70bce1d3af36255e7a5bf92d5d4cf49a77d7595`.
- No full 256K inference, vision, conversation parking/restore, concurrent sequences, full HIP build or other
  GPU architecture validation. The opt-in is an experiment, not a generally validated implementation.
- Raw results and engine logs are included. No throughput claim is made for the default-output parity runs:
  the reference ran before the model's OS cache was warm.

## Reproduce

Build as in the preceding GQA-16 report. With only the test GPU visible and the production engine stopped:

```sh
python bench/results/2026-10-21-ornith-host-kv/compare.py \
  --model FIRST_SHARD --engine NEW_ENGINE --baseline GQA16_ENGINE \
  --tokenizer PACK/tokenizer --profile PROFILE_SNAPSHOT --out RESULTS
```

The script restores the profile, starts a new process per arm, saves the generated IDs and parses the engine's
own prompt/decode timings and counters. It also runs an FP16 host-KV smoke case. The profile snapshot and the
GGUF files are external inputs. No production config, model files, quantization or PR-63 branch was changed.
