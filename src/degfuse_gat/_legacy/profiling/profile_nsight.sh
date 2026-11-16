#!/bin/bash
# ============================================================
# Nsight Compute + Nsight Systems profiling commands
# ============================================================
# Usage: bash profiling/profile_nsight.sh [Cora|ogbn-arxiv]
# Output: profiling/reports/

DATASET=${1:-"Cora"}
mkdir -p profiling/reports

echo ">>> Profiling DegFuse-GAT kernel with Nsight Compute..."
ncu \
  --set full \
  --kernel-name "_degfuse_bucket_kernel" \
  --launch-count 5 \
  --export profiling/reports/degfuse_${DATASET} \
  python -m benchmarks.benchmark_forward --dataset ${DATASET} --iters 10

echo ""
echo ">>> Timeline profiling with Nsight Systems..."
nsys profile \
  --output profiling/reports/timeline_${DATASET} \
  --trace cuda,nvtx \
  --force-overwrite true \
  python -m benchmarks.benchmark_forward --dataset ${DATASET} --iters 50

echo ""
echo "Done! Open reports:"
echo "  ncu-ui profiling/reports/degfuse_${DATASET}.ncu-rep"
echo "  nsys-ui profiling/reports/timeline_${DATASET}.nsys-rep"
