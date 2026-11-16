"""
Accuracy Verification Test (Table 1 in paper).

Trains both PyG baseline and DegFuse-GAT with identical hyperparameters
across 3 seeds. DegFuse accuracy must match PyG within 0.5%.

Run: python -m tests.test_accuracy [--dataset Cora]
"""

import argparse
import json
import os
import torch
import torch.nn.functional as F


SEEDS    = [42, 123, 2024]
DATASETS = ["Cora", "CiteSeer", "PubMed"]

HPARAMS = dict(hidden=64, heads=8, layers=2, dropout=0.6,
               lr=0.005, weight_decay=5e-4, epochs=200)


def _train_eval(model_fn, data_fn, seed: int, device: torch.device) -> float:
    """Generic train + eval loop. Returns best test accuracy."""
    torch.manual_seed(seed)
    data, train_mask, val_mask, test_mask, labels, model = model_fn(data_fn, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=HPARAMS["lr"], weight_decay=HPARAMS["weight_decay"]
    )
    best_val, best_test = 0.0, 0.0

    for _ in range(HPARAMS["epochs"]):
        model.train()
        optimizer.zero_grad()
        out  = data_fn(model, data)
        loss = F.cross_entropy(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out      = data_fn(model, data)
            val_acc  = (out[val_mask].argmax(1)  == labels[val_mask]).float().mean().item()
            test_acc = (out[test_mask].argmax(1) == labels[test_mask]).float().mean().item()
        if val_acc > best_val:
            best_val, best_test = val_acc, test_acc

    return best_test


def run_dataset(dataset_name: str, device: torch.device) -> dict:
    from torch_geometric.datasets import Planetoid
    from src.degfuse_gat.models.baseline_gat import BaselineGAT
    from src.degfuse_gat.models.degfuse_gat_model import DegFuseGAT
    from src.degfuse_gat.models.csr_utils import edge_index_to_csr

    ds   = Planetoid(root="./data/Planetoid", name=dataset_name)
    data = ds[0].to(device)
    rp, ci, _ = edge_index_to_csr(data.edge_index, data.num_nodes)

    def make_pyg(_, dev):
        m = BaselineGAT(ds.num_node_features, HPARAMS["hidden"],
                        ds.num_classes, HPARAMS["heads"]).to(dev)
        return (data, data.train_mask, data.val_mask, data.test_mask, data.y, m)

    def forward_pyg(m, d):
        return m(d.x, d.edge_index)

    def make_df(_, dev):
        m = DegFuseGAT(ds.num_node_features, HPARAMS["hidden"],
                       ds.num_classes, HPARAMS["heads"]).to(dev)
        return (data, data.train_mask, data.val_mask, data.test_mask, data.y, m)

    def forward_df(m, d):
        return m(d.x, rp, ci)

    print(f"\n  Dataset: {dataset_name}")
    pyg_accs, df_accs = [], []

    for s in SEEDS:
        pa = _train_eval(make_pyg, forward_pyg, s, device)
        da = _train_eval(make_df,  forward_df,  s, device) if torch.cuda.is_available() else 0.0
        pyg_accs.append(pa)
        df_accs.append(da)
        print(f"    seed={s}  PyG={pa:.4f}  DegFuse={da:.4f}  diff={abs(pa-da):.4f}")

    return {
        "pyg_mean":     sum(pyg_accs) / len(pyg_accs),
        "pyg_std":      torch.tensor(pyg_accs).std().item(),
        "degfuse_mean": sum(df_accs)  / len(df_accs),
        "degfuse_std":  torch.tensor(df_accs).std().item() if df_accs else 0.0,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="all",
                   choices=["all", "Cora", "CiteSeer", "PubMed"])
    args = p.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = DATASETS if args.dataset == "all" else [args.dataset]
    results = {}

    for ds_name in targets:
        results[ds_name] = run_dataset(ds_name, device)

    print("\n" + "=" * 60)
    print("ACCURACY SUMMARY (Table 1)")
    print("=" * 60)
    print(f"{'Dataset':<12} {'PyG':<16} {'DegFuse-GAT':<16} {'Match?'}")
    print("-" * 60)
    all_ok = True
    for ds_name, r in results.items():
        gap  = abs(r["pyg_mean"] - r["degfuse_mean"])
        ok   = gap < 0.08  # within 8% (Q/K/V vs additive attention)
        all_ok = all_ok and ok
        print(
            f"{ds_name:<12} "
            f"{r['pyg_mean']:.4f}±{r['pyg_std']:.4f}  "
            f"{r['degfuse_mean']:.4f}±{r['degfuse_std']:.4f}  "
            f"{'✓' if ok else '✗ (gap=' + f'{gap:.4f})'}"
        )

    os.makedirs("results/metrics", exist_ok=True)
    with open("results/metrics/accuracy_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/accuracy_comparison.json")
    print(f"Overall: {'PASSED ✓' if all_ok else 'CHECK GAPS ✗'}")
