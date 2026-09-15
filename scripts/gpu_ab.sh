#!/bin/bash
# CPU vs GPU A/B for PDLP's SpMV backend (module.txt section 24E).
#
#   wsl bash scripts/gpu_ab.sh                      # the default instance set
#   wsl bash scripts/gpu_ab.sh path/to/model.mps    # one or more of your own
#   ITERS=5000 wsl bash scripts/gpu_ab.sh           # longer runs
#
# Both paths run the SAME algorithm with the same iteration cap, so
# `matrix_products` should agree closely. A GPU path that quietly took a
# different number of steps would look fast for the wrong reason, so the counts
# are compared and any gap is reported.
#
# They are NOT expected to match exactly, and that is not a defect. cuSPARSE
# accumulates each row in a different order from our sequential CSR loop, so the
# two disagree in the last bits; PDLP's ADAPTIVE step size then accepts or
# rejects a borderline trial differently and the product counts drift apart by a
# handful. Measured: 1 in 6415 on 25fv47, 11 in 6481 on fit2d. A drift of more
# than 1% means something else is wrong and the run is flagged.
#
# Reports wall time only. The kernel-vs-transfer split needs two syncs per
# product and so inflates the very total it is attributing, which is why
# `--gpu-spmv-timing=1` is a separate opt-in run and never combined with this.
set -u

export PATH=/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.." || exit 1

SOLVE=./build-cuda/tools/solve/solve
ITERS=${ITERS:-2000}
TIMEOUT=${TIMEOUT:-600}

if [ ! -x "$SOLVE" ]; then
  echo "no CUDA build found at $SOLVE"
  echo "build it with:  cmake --preset cuda && cmake --build build-cuda"
  exit 1
fi

if [ "$#" -gt 0 ]; then
  FILES=("$@")
else
  FILES=(
    tests/data/netlib/afiro.mps      # tiny   -- expect the GPU to lose badly
    tests/data/netlib/25fv47.mps     # small
    tests/data/maros-r7.mps          # medium
    tests/data/datt256.mps           # large  -- expect the GPU to win
  )
fi

field() { grep -E "^$2=" <<<"$1" | cut -d= -f2; }

printf '%-34s %10s %10s %10s   %s\n' instance size CPU GPU verdict
printf '%s\n' "---------------------------------------------------------------------------"

for f in "${FILES[@]}"; do
  [ -f "$f" ] || { printf '%-34s %10s   (missing)\n' "$(basename "$f")" "-"; continue; }
  size=$(du -h "$f" | cut -f1)

  cpu_out=$(timeout "$TIMEOUT" $SOLVE "$f" --method=pdlp --pdlp-max-iter="$ITERS" 2>&1)
  gpu_out=$(timeout "$TIMEOUT" $SOLVE "$f" --method=pdlp --pdlp-max-iter="$ITERS" --gpu-spmv=1 2>&1)

  cpu_t=$(field "$cpu_out" solve_time_seconds)
  gpu_t=$(field "$gpu_out" solve_time_seconds)
  cpu_p=$(field "$cpu_out" matrix_products)
  gpu_p=$(field "$gpu_out" matrix_products)

  if [ -z "$cpu_t" ] || [ -z "$gpu_t" ]; then
    printf '%-34s %10s %10s %10s   FAILED (timeout or load error)\n' \
      "$(basename "$f")" "$size" "${cpu_t:--}" "${gpu_t:--}"
    continue
  fi

  note=$(awk -v c="$cpu_t" -v g="$gpu_t" 'BEGIN{
    r = c / g;
    if (r >= 1) printf "GPU %.2fx faster", r; else printf "GPU %.2fx slower", 1/r }')
  drift=$(awk -v a="$cpu_p" -v b="$gpu_p" 'BEGIN{
    if (a == 0) { print "?"; exit }
    d = (a > b ? a - b : b - a) / a;
    if (d > 0.01) printf "SUSPECT: %d vs %d products", a, b;
    else if (a != b) printf "(%+d products, rounding)", b - a }')
  [ -n "$drift" ] && note="$note  $drift"
  printf '%-34s %10s %10s %10s   %s\n' "$(basename "$f")" "$size" "$cpu_t" "$gpu_t" "$note"
done

echo
echo "iterations capped at $ITERS (ITERS=N to change)."
echo "attribution, separate run:  $SOLVE <file> --method=pdlp --gpu-spmv=1 --gpu-spmv-timing=1"
