# DegFuse-GAT: Degree-Aware Fused Sparse Attention Kernels for Graph Attention Networks

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
<!-- TODO: Add DOI badge after publication -->

A Triton-based GPU kernel that fuses SDDMM + Online Softmax + SpMM into a single pass
for GAT inference, with degree-aware warp scheduling for heterogeneous graphs.
On `ogbn-arxiv`, DegFuse-GAT achieves a **4.14x peak VRAM reduction** (6,096 MB -> 1,472 MB)
and a **1.37x latency speedup** over PyTorch Geometric's `GATConv`. On `ogbn-products` and
`ogbn-proteins` (> 79M edges), PyG OOMs on a 40 GB H100 MIG slice while DegFuse-GAT
completes successfully in under 23 GB -- enabling GAT training on consumer-grade GPUs
where PyG baselines cannot even run a forward pass.

> **Known limitation:** the fused kernel currently shows a **1.2-7.1% test-accuracy gap**
> vs. the PyG baseline (see [Results](#results-summary--known-limitations) below). This is
> an open issue, most likely caused by the kernel's scaled dot-product attention differing
> from GAT's original additive attention formulation.

<!-- TODO: Add arXiv link after publication -->

---

## Table of Contents

- [Installation](#installation)
- [Datasets](#datasets)
- [Quick Start](#quick-start)
- [Reproduce Paper Results](#reproduce-paper-results)
- [Results Summary & Known Limitations](#results-summary--known-limitations)
- [Repository Structure](#repository-structure)
- [Hardware Requirements](#hardware-requirements)
- [Citation](#citation)
- [License](#license)

## Installation

```bash
git clone https://github.com/Arulprakasam18/DegFuse-GAT.git
cd DegFuse-GAT
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -e .
```

**Tested with:** Python 3.10, PyTorch 2.4+, CUDA 12.x, Triton 2.2+, NVIDIA H100 40GB (MIG).

## Datasets

Datasets are downloaded automatically on first use via PyTorch Geometric and OGB.
They are stored under `data/` (gitignored).

| Dataset | Nodes | Edges | Source |
|---------|-------|-------|--------|
| Cora | 2,708 | 10,556 | Planetoid |
| CiteSeer | 3,327 | 9,104 | Planetoid |
| PubMed | 19,717 | 88,648 | Planetoid |
| ogbn-arxiv | 169,343 | 1,166,243 | OGB |
| ogbn-products | 2,449,029 | 61,859,140 | OGB |

See [`data/README.md`](data/README.md) for details.

## Quick Start

Train DegFuse-GAT on Cora with default hyperparameters:

```bash
python scripts/train.py --config configs/default.yaml
```

Run a 2-epoch smoke test:

```bash
python scripts/train.py --config configs/smoke_test.yaml
```

## Reproduce Paper Results

| Paper Item | Command | Expected |
|------------|---------|----------|
| Table 1 (Cora) | `python scripts/train.py --config configs/table1_accuracy_cora.yaml` | DegFuse 77.50% ± 0.46% vs PyG 82.07% ± 0.64% |
| Table 1 (CiteSeer) | `python scripts/train.py --config configs/table1_accuracy_citeseer.yaml` | DegFuse 62.00% ± 1.55% vs PyG 69.07% ± 1.20% |
| Table 1 (PubMed) | `python scripts/train.py --config configs/table1_accuracy_pubmed.yaml` | DegFuse 75.60% ± 0.87% vs PyG 76.80% ± 0.92% |
| Table 1 (ogbn-arxiv) | `python scripts/train.py --config configs/table1_accuracy_arxiv.yaml` | See [Results Summary](#results-summary--known-limitations) |
| Table 2 (Latency) | `python -m benchmarks.benchmark_forward --dataset Cora` | See results/metrics/ |
| Table 3 (Memory) | `python -m benchmarks.benchmark_memory --dataset Cora` | See results/metrics/ |
| Table 4 (OOM) | `python -m benchmarks.benchmark_oom` | See results/metrics/ |
| Table 5 (Crossover) | `python -m benchmarks.benchmark_crossover` | See results/metrics/ |
| Table 6 (Ablation) | `python -m benchmarks.benchmark_degree_ablation --dataset Cora` | See results/metrics/ |
| All Figures | `python scripts/make_figures.py` | results/figures/ |
| Correctness | `python -m tests.test_correctness` | All PASSED |
| Accuracy Test | `python -m tests.test_accuracy` | Gap < 8% |

## Results Summary & Known Limitations

Full experimental write-up (hardware, methodology, every table and figure) lives in
[`src/degfuse_gat/_legacy/RESULTS_REPORT.md`](src/degfuse_gat/_legacy/RESULTS_REPORT.md).
Headline numbers, measured on an NVIDIA H100 40GB MIG slice:

### Latency (forward pass)

| Dataset | Edges | PyG (ms) | DegFuse Auto (ms) | Speedup |
|---------|------:|---------:|-------------------:|--------:|
| Cora | 10,556 | 1.153 | 1.244 | 0.93x |
| CiteSeer | 9,104 | 1.552 | 1.677 | 0.93x |
| PubMed | 88,648 | 3.147 | 3.239 | 0.97x |
| ogbn-arxiv | 1,166,243 | 25.782 | 18.858 | **1.37x** |
| ogbn-products | 123,718,280* | OOM | 387.948 | PyG OOMs |
| ogbn-proteins | 79,122,504* | OOM | 66.213 | PyG OOMs |

\* bidirectional edge counts as loaded by PyG.

### Peak VRAM

| Dataset | PyG (MB) | DegFuse (MB) | Reduction |
|---------|---------:|-------------:|----------:|
| Cora | 113.3 | 78.7 | 1.44x |
| PubMed | 580.6 | 232.1 | 2.50x |
| ogbn-arxiv | 6,095.9 | 1,472.1 | **4.14x** |
| ogbn-proteins | OOM | 5,417.1 | PyG OOMs |
| ogbn-products | OOM | 22,974.1 | PyG OOMs |

**Takeaway:** below ~100k edges, the fused kernel's launch overhead makes it slightly
slower than PyG; the adaptive `"auto"` dispatch mode keeps that overhead under 7% by
falling back to PyG on small graphs. Above the ~100k-edge crossover, DegFuse-GAT wins on
both latency and memory, and is the only one of the two that can run at all above ~80M
edges within 40 GB of VRAM.

**Known accuracy gap:** DegFuse-GAT currently trails the PyG baseline by 1.2-7.1% test
accuracy across Cora/CiteSeer/PubMed, exceeding the project's ±0.5% target. Forward-pass
correctness is verified (`max_diff = 1.53e-03` vs. the reference implementation), so the
gap is believed to stem from DegFuse's scaled dot-product attention (`Q*K/sqrt(D)`)
differing from GAT's original additive attention (`a^T[Wh_i || Wh_j]`), rather than a bug
in the kernel. This is tracked as the top-priority follow-up; see
[`src/degfuse_gat/_legacy/RESULTS_REPORT.md`](src/degfuse_gat/_legacy/RESULTS_REPORT.md#8-accuracy-verification-table-1)
for details.

## Repository Structure

```
DegFuse-GAT/
├── configs/                          # YAML experiment configs
│   ├── default.yaml                  # Default hyperparameters
│   ├── smoke_test.yaml               # Quick verification (2 epochs)
│   └── table1_accuracy_*.yaml        # Per-dataset Table 1 configs
├── src/degfuse_gat/                  # Core package
│   ├── models/                       # Model definitions
│   │   ├── degfuse_gat_model.py      # DegFuseGAT (main model)
│   │   ├── baseline_gat.py           # BaselineGAT (PyG GATConv)
│   │   ├── degfuse_kernel.py         # Triton fused kernel
│   │   ├── degree_aware_scheduler.py # Degree-aware bucket scheduling
│   │   ├── csr_utils.py              # CSR format utilities
│   │   └── online_softmax.py         # Reference implementations
│   ├── data/                         # Dataset loading
│   │   └── datasets.py
│   ├── training/                     # Training loop
│   │   └── trainer.py
│   ├── evaluation/                   # Evaluation
│   ├── utils/                        # Utilities
│   │   ├── config.py                 # OmegaConf config loader
│   │   ├── seed.py                   # Global seed utility
│   │   ├── logging.py                # Logging setup
│   │   └── compat.py                 # OGB compatibility patches
│   └── _legacy/                      # Pre-refactor reference material
│       ├── RESULTS_REPORT.md         # Full experimental write-up (all tables/figures)
│       ├── REFERENCES.md             # Literature review / related work
│       ├── plotting/                 # Original figure-generation scripts
│       └── profiling/                # Nsight/PyTorch profiler scripts
├── scripts/                          # CLI entry points
│   ├── train.py                      # Training CLI
│   ├── eval.py                       # Evaluation CLI
│   └── make_figures.py               # Regenerate paper figures
├── benchmarks/                       # Reproducible benchmarks
│   ├── benchmark_forward.py          # Forward latency
│   ├── benchmark_memory.py           # Peak VRAM
│   ├── benchmark_oom.py              # OOM scalability
│   ├── benchmark_crossover.py        # Crossover analysis
│   └── benchmark_degree_ablation.py  # Degree ablation
├── tests/                            # Correctness tests
│   ├── test_correctness.py           # Kernel correctness
│   └── test_accuracy.py              # Accuracy comparison
├── results/                          # Generated at runtime
│   ├── metrics/                      # JSON benchmark results (reproduced)
│   └── figures/                      # Paper figures (reproduced)
├── data/                             # Datasets (auto-downloaded, gitignored)
│   └── README.md
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── CITATION.cff
```

## Hardware Requirements

- **GPU**: NVIDIA A100 or H100 (compute capability >= 8.0)
- **VRAM**: >= 10 GB
- **CUDA**: >= 12.0
- **Approximate wall-clock times**:
  - Cora training (200 epochs): ~30 seconds
  - ogbn-arxiv training (200 epochs): ~5 minutes
  - Full benchmark suite: ~15 minutes

## Citation

```bibtex
@article{arulprakasam2026degfuse,
  title={Degree-Aware Fused Sparse Attention Kernels for Graph Attention Networks},
  author={Arulprakasam, J},
  year={2026},
  note={Under review}
}
```

## License

MIT License -- see [LICENSE](LICENSE)

## Acknowledgments

This work builds on the Graph Attention Network architecture of
[Veličković et al.](https://arxiv.org/abs/1710.10903), the online/streaming softmax
formulation from [FlashAttention](https://arxiv.org/abs/2205.14135) (Dao et al.), and is
implemented using [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/),
[OGB](https://ogb.stanford.edu/), and [Triton](https://github.com/openai/triton). See
[`src/degfuse_gat/_legacy/REFERENCES.md`](src/degfuse_gat/_legacy/REFERENCES.md) for the
full literature review.
