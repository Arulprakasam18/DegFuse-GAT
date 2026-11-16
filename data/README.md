# Dataset Setup

Datasets are downloaded **automatically** on first use by PyTorch Geometric and OGB.
No manual download is required.

## Supported Datasets

| Dataset | Source | Size | Auto-download |
|---------|--------|------|---------------|
| Cora | [Planetoid](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html) | ~10 MB | Yes |
| CiteSeer | Planetoid | ~10 MB | Yes |
| PubMed | Planetoid | ~50 MB | Yes |
| ogbn-arxiv | [OGB](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) | ~200 MB | Yes |
| ogbn-products | [OGB](https://ogb.stanford.edu/docs/nodeprop/#ogbn-products) | ~1.5 GB | Yes |

## Expected Directory Layout

After first run, the `data/` directory will contain:

```
data/
├── Planetoid/
│   ├── Cora/
│   ├── CiteSeer/
│   └── PubMed/
├── OGB/
│   ├── arxiv/
│   └── products/
└── README.md
```

## Manual Download (Optional)

If you prefer to pre-download all datasets:

```bash
python -c "
from torch_geometric.datasets import Planetoid
for name in ['Cora', 'CiteSeer', 'PubMed']:
    Planetoid(root='./data/Planetoid', name=name)
    print(f'Downloaded {name}')
"
```

For OGB datasets (large, ~1.5 GB for products):

```bash
python -c "
from src.degfuse_gat.utils.compat import apply_patches
apply_patches()
from ogb.nodeproppred import PygNodePropPredDataset
PygNodePropPredDataset(name='ogbn-arxiv', root='./data/OGB')
print('Downloaded ogbn-arxiv')
"
```

## Licenses

- **Planetoid** (Cora, CiteSeer, PubMed): Public domain academic datasets
- **OGB**: MIT License ([ogb.stanford.edu](https://ogb.stanford.edu))
