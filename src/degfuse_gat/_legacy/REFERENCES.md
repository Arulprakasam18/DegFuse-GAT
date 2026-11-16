# DegFuse-GAT — Complete References & Bibliography

**Purpose**: Full bibliography with links, BibTeX entries, and positioning notes for the DegFuse-GAT research paper.

---

## 1. Foundational Model (What We Accelerate)

### [1] GAT: Graph Attention Networks
- **Authors**: Veličković, Cucurull, Casanova, Romero, Liò, Bengio
- **Venue**: ICLR 2018
- **Link**: https://arxiv.org/abs/1710.10903
- **GitHub**: https://github.com/PetarV-/GAT
- **Role**: The model whose 3-step pipeline (SDDMM → Softmax → SpMM) we fuse
- **Cite as**: "We accelerate the GAT [Veličković et al., 2018] pipeline..."

```bibtex
@inproceedings{velickovic2018gat,
  title     = {Graph Attention Networks},
  author    = {Veli{\v{c}}kovi{\'c}, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Li{\`o}, Pietro and Bengio, Yoshua},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2018},
  url       = {https://arxiv.org/abs/1710.10903}
}
```

### [2] GATv2: How Attentive Are Graph Attention Networks?
- **Authors**: Brody, Alon, Yahav
- **Venue**: ICLR 2022
- **Link**: https://arxiv.org/abs/2105.14491
- **Role**: Extension of GAT with dynamic attention; our kernel also supports this variant
- **Key difference**: GAT uses static attention (`a^T[Wh_i || Wh_j]`), GATv2 uses dynamic (`a^T LeakyReLU(W[h_i || h_j])`)

```bibtex
@inproceedings{brody2022gatv2,
  title     = {How Attentive are Graph Attention Networks?},
  author    = {Brody, Shaked and Alon, Uri and Yahav, Eran},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2022},
  url       = {https://arxiv.org/abs/2105.14491}
}
```

---

## 2. Technique Sources

### [3] FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
- **Authors**: Dao, Fu, Ermon, Rudra, Ré
- **Venue**: NeurIPS 2022
- **Link**: https://arxiv.org/abs/2205.14135
- **GitHub**: https://github.com/Dao-AILab/flash-attention
- **Role**: **SOURCE** of the online softmax algorithm we adapt. Original uses fixed-length sequence rows; we adapt to variable-length CSR rows.
- **Key section**: Algorithm 1 (online softmax merge step) — directly reused
- **Cite as**: "We adapt the online softmax algorithm from FlashAttention [Dao et al., 2022]..."

```bibtex
@inproceedings{dao2022flashattention,
  title     = {Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
  author    = {Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2022},
  url       = {https://arxiv.org/abs/2205.14135}
}
```

### [4] FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
- **Authors**: Dao
- **Venue**: ICLR 2024
- **Link**: https://arxiv.org/abs/2307.08691
- **Role**: Improved warp-level parallelism ideas (secondary reference)

```bibtex
@inproceedings{dao2024flashattention2,
  title     = {Flash{A}ttention-2: Faster Attention with Better Parallelism and Work Partitioning},
  author    = {Dao, Tri},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024},
  url       = {https://arxiv.org/abs/2307.08691}
}
```

---

## 3. Direct Competitors

### [5] DF-GNN: Dynamic Fusion Framework for Attention GNNs on GPUs
- **Authors**: Liu et al.
- **Venue**: LoG 2024 / arXiv November 2024
- **Link**: https://arxiv.org/abs/2411.16127
- **GitHub**: https://github.com/paoxiaode/DF-GNN
- **Role**: **CLOSEST COMPETITOR** — fuses all 3 ops but in CUDA C++
- **Gaps vs ours**:
  - CUDA C++ only (not accessible to most researchers)
  - Fixed block-level scheduling (ignores node degree)
  - No degree-aware warp allocation
  - No online softmax (two-pass)
- **Cite as**: "DF-GNN [Liu et al., 2024] fuses the same 3 operations but requires CUDA C++ expertise and uses fixed block scheduling..."

```bibtex
@article{liu2024dfgnn,
  title   = {{DF-GNN}: Dynamic Fusion Framework for Attention Graph Neural Networks on {GPU}s},
  author  = {Liu, Yuke and others},
  journal = {arXiv preprint arXiv:2411.16127},
  year    = {2024},
  url     = {https://arxiv.org/abs/2411.16127}
}
```

### [6] Fused3S: Fast Sparse Attention on Tensor Cores via 3S Fusion
- **Authors**: Li et al.
- **Venue**: ICS 2025
- **Link**: Search arXiv: "Fused3S sparse attention tensor core ICS 2025"
- **Role**: **CURRENT SOTA** — fuses 3S with Tensor Core support
- **Gaps vs ours**:
  - Pure CUDA C++
  - Requires block-structured sparse format (preprocessing overhead)
  - No degree-aware scheduling
  - OOMs on AmazonProducts and some large graphs
- **Cite as**: "Fused3S [Li et al., ICS 2025] is the current SOTA but requires block-structured format conversion and OOMs on large graphs..."

```bibtex
@inproceedings{li2025fused3s,
  title     = {Fused3S: Fast Sparse Attention on Tensor Cores via 3S Fusion},
  author    = {Li, and others},
  booktitle = {International Conference on Supercomputing (ICS)},
  year      = {2025}
}
```

### [7] FusedMM: A Unified SDDMM-SpMM Kernel for GNN Training and Inference
- **Authors**: Rahman, Suda, Minh
- **Venue**: IPDPS 2021
- **Link**: https://doi.org/10.1109/IPDPS49936.2021.00089
- **Role**: **GAP PAPER** — fuses SDDMM+SpMM but **explicitly omits softmax**
- **Key quote**: "In almost all applications except attention GNNs, messages are immediately aggregated — softmax is omitted"
- **Cite as**: "FusedMM [Rahman et al., 2021] fuses SDDMM and SpMM but explicitly omits softmax, making it inapplicable to GAT..."

```bibtex
@inproceedings{rahman2021fusedmm,
  title     = {{FusedMM}: A Unified {SDDMM}-{SpMM} Kernel for Graph Neural Network Training and Inference},
  author    = {Rahman, Md Khaledur and Suda, Renat and Minh, Ha Duc},
  booktitle = {IEEE International Parallel and Distributed Processing Symposium (IPDPS)},
  year      = {2021},
  doi       = {10.1109/IPDPS49936.2021.00089}
}
```

---

## 4. Systems Baselines

### [8] TC-GNN: Bridging Sparse GNN Computation and Dense Tensor Cores on GPUs
- **Authors**: Wang et al.
- **Venue**: USENIX ATC 2023
- **Link**: https://www.usenix.org/conference/atc23/presentation/wang-yuke
- **GitHub**: https://github.com/YukeWang96/TC-GNN_ATC23
- **Role**: Tensor Core baseline for sparse GNNs. Introduces Sparse Graph Tiling (SGT).

```bibtex
@inproceedings{wang2023tcgnn,
  title     = {{TC-GNN}: Bridging Sparse {GNN} Computation and Dense Tensor Cores on {GPU}s},
  author    = {Wang, Yuke and others},
  booktitle = {USENIX Annual Technical Conference (ATC)},
  year      = {2023},
  url       = {https://www.usenix.org/conference/atc23/presentation/wang-yuke}
}
```

### [9] FTC-GNN: Accelerating GNNs with Collaborative TC-CUDA Kernel Design
- **Authors**: Niu et al.
- **Venue**: arXiv 2024
- **Link**: https://arxiv.org/abs/2412.12218
- **Role**: Collaborative TC + CUDA core design — different parallel approach from ours

```bibtex
@article{niu2024ftcgnn,
  title   = {{FTC-GNN}: Accelerating Graph Neural Networks with Collaborative {TC-CUDA} Kernel Design},
  author  = {Niu, and others},
  journal = {arXiv preprint arXiv:2412.12218},
  year    = {2024},
  url     = {https://arxiv.org/abs/2412.12218}
}
```

### [10] FlashSparse: Sparse Matrix-Matrix Multiplication on Tensor Cores
- **Authors**: Liu et al.
- **Venue**: arXiv 2024
- **Link**: Search arXiv: "FlashSparse sparse matrix tensor cores 2024"
- **Role**: SpMM/SDDMM on Tensor Cores — no softmax fusion

```bibtex
@article{liu2024flashsparse,
  title   = {Flash{S}parse: Sparse Matrix-Matrix Multiplication on Tensor Cores},
  author  = {Liu, and others},
  journal = {arXiv preprint},
  year    = {2024}
}
```

### [17] AutoSAGE: Input-Aware CUDA Scheduling for Sparse GNN Aggregation
- **Authors**: Peng et al.
- **Venue**: arXiv 2025
- **Link**: https://arxiv.org/abs/2511.17594
- **Role**: Input-aware scheduling — related to our degree-aware approach

```bibtex
@article{peng2025autosage,
  title   = {Auto{SAGE}: Input-Aware {CUDA} Scheduling for Sparse {GNN} Aggregation},
  author  = {Peng, and others},
  journal = {arXiv preprint arXiv:2511.17594},
  year    = {2025},
  url     = {https://arxiv.org/abs/2511.17594}
}
```

---

## 5. Algorithm Papers

### [11] SGAT / Sparse GAT: Sparse Graph Attention Networks
- **Authors**: Ye & Ji
- **Venue**: IEEE TKDE 2021
- **Link**: https://arxiv.org/abs/1912.00552
- **Role**: L0 sparsity in GAT attention — algorithmic motivation. Our kernel handles sparse attention natively.

```bibtex
@article{ye2021sgat,
  title   = {Sparse Graph Attention Networks},
  author  = {Ye, Yang and Ji, Shihao},
  journal = {IEEE Transactions on Knowledge and Data Engineering (TKDE)},
  year    = {2021},
  url     = {https://arxiv.org/abs/1912.00552}
}
```

### [12] GraphSAGE: Inductive Representation Learning on Large Graphs
- **Authors**: Hamilton, Ying, Leskovec
- **Venue**: NeurIPS 2017
- **Link**: https://arxiv.org/abs/1706.02216
- **Role**: Foundational aggregation model (context for GNN systems work)

```bibtex
@inproceedings{hamilton2017graphsage,
  title     = {Inductive Representation Learning on Large Graphs},
  author    = {Hamilton, William L. and Ying, Rex and Leskovec, Jure},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2017},
  url       = {https://arxiv.org/abs/1706.02216}
}
```

---

## 6. Benchmark Datasets

### [13] OGB: Open Graph Benchmark
- **Authors**: Hu et al.
- **Venue**: NeurIPS 2020
- **Link**: https://arxiv.org/abs/2005.00687
- **Website**: https://ogb.stanford.edu
- **GitHub**: https://github.com/snap-stanford/ogb
- **Role**: ogbn-arxiv, ogbn-products, and ogbn-proteins datasets
- **Cite as**: "We evaluate on standard OGB benchmarks [Hu et al., 2020]..."

```bibtex
@inproceedings{hu2020ogb,
  title     = {Open Graph Benchmark: Datasets for Machine Learning on Graphs},
  author    = {Hu, Weihua and Fey, Matthias and Zitnik, Marinka and Dong, Yuxiao and Ren, Hongyu and Liu, Bowen and Catasta, Michele and Leskovec, Jure},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2020},
  url       = {https://arxiv.org/abs/2005.00687}
}
```

### [14] Cora, CiteSeer, PubMed (Planetoid)
- **Authors**: Sen et al. (Cora), McCallum et al. (CiteSeer)
- **Link**: https://linqs.soe.ucsc.edu/data
- **Download**: `Planetoid(root='./data/Planetoid', name='Cora')` via PyG

```bibtex
@article{sen2008collective,
  title   = {Collective Classification in Network Data},
  author  = {Sen, Prithviraj and Namata, Galileo Mark and Bilgic, Mustafa and Getoor, Lise and Gallagher, Brian and Eliassi-Rad, Tina},
  journal = {AI Magazine},
  volume  = {29},
  number  = {3},
  year    = {2008}
}
```

---

## 7. Frameworks & Tools

### [15] Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations
- **Authors**: Tillet, Kung, Cox
- **Venue**: MAPL 2019
- **Link**: https://arxiv.org/abs/2104.10756
- **GitHub**: https://github.com/triton-lang/triton
- **Docs**: https://triton-lang.org/main/index.html
- **Tutorials**: https://triton-lang.org/main/getting-started/tutorials/
- **Role**: GPU programming framework we use (instead of CUDA C++)

```bibtex
@inproceedings{tillet2019triton,
  title     = {Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations},
  author    = {Tillet, Philippe and Kung, H. T. and Cox, David},
  booktitle = {Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages (MAPL)},
  year      = {2019},
  url       = {https://arxiv.org/abs/2104.10756}
}
```

### [16] PyTorch Geometric (PyG)
- **Authors**: Fey & Lenssen
- **Venue**: ICLR Workshop 2019
- **Link**: https://arxiv.org/abs/1903.02428
- **GitHub**: https://github.com/pyg-team/pytorch_geometric
- **Docs**: https://pytorch-geometric.readthedocs.io
- **Role**: Baseline framework (GATConv)

```bibtex
@inproceedings{fey2019pyg,
  title     = {Fast Graph Representation Learning with {PyTorch Geometric}},
  author    = {Fey, Matthias and Lenssen, Jan Eric},
  booktitle = {ICLR Workshop on Representation Learning on Graphs and Manifolds},
  year      = {2019},
  url       = {https://arxiv.org/abs/1903.02428}
}
```

---

## 8. Surveys

### [18] GNN Survey: A Comprehensive Study of Graph Neural Networks
- **Authors**: Wu et al.
- **Venue**: IEEE TNNLS 2021
- **Link**: https://arxiv.org/abs/1901.00596
- **Role**: Background reading on GNN architectures

```bibtex
@article{wu2021gnnsurvey,
  title   = {A Comprehensive Survey on Graph Neural Networks},
  author  = {Wu, Zonghan and Pan, Shirui and Chen, Fengwen and Long, Guodong and Zhang, Chengqi and Yu, Philip S.},
  journal = {IEEE Transactions on Neural Networks and Learning Systems (TNNLS)},
  volume  = {32},
  number  = {1},
  pages   = {4--24},
  year    = {2021},
  url     = {https://arxiv.org/abs/1901.00596}
}
```

---

## 9. Novelty Comparison Table

| Method | Fuses SDDMM+Softmax+SpMM | Degree-Aware | Online Softmax | Triton-Native | Low VRAM (≤10GB) |
|--------|:------------------------:|:------------:|:--------------:|:-------------:|:----------------:|
| FusedMM (IPDPS 2021) [7] | ✗ | ✗ | ✗ | ✗ | ✓ |
| DF-GNN (LoG 2024) [5] | ✓ | partial | ✗ | ✗ | ✓ |
| Fused3S (ICS 2025) [6] | ✓ | ✗ | ✓ | ✗ | ✗ |
| FlashSparse (2024) [10] | partial | ✗ | ✗ | ✗ | ✓ |
| **DegFuse-GAT (ours)** | **✓** | **✓** | **✓** | **✓** | **✓** |

**DegFuse-GAT is the first to combine all five properties simultaneously.**

---

## 10. Baseline Repositories

| Baseline | Repository | Notes |
|----------|-----------|-------|
| PyG GATConv (primary) | https://github.com/pyg-team/pytorch_geometric | Docs: [GATConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html) |
| DF-GNN (closest competitor) | https://github.com/paoxiaode/DF-GNN | Key file: `df_gnn/fused_kernel.cu` |
| Original GAT (reference) | https://github.com/PetarV-/GAT | Keras/TF implementation |
| PyTorch GAT (reference) | https://github.com/gordicaleksa/pytorch-GAT | Another PyTorch implementation |
| DGL GAT (second baseline) | https://github.com/dmlc/dgl | Example: `examples/pytorch/gat/train.py` |

---

## 11. Citation for This Work

```bibtex
@article{arulprakasam2026degfusegat,
  title   = {{DegFuse-GAT}: Degree-Aware Fused Sparse Attention Kernels for Graph Attention Networks},
  author  = {Arulprakasam, J},
  year    = {2026},
  note    = {Target venues: MLSys 2026, ICS 2026, PPoPP 2026}
}
```
