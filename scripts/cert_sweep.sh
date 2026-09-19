#!/bin/bash
# Infeasibility-certificate tolerance sweep (module.txt section 24D/24F).
#
#   bash scripts/cert_sweep.sh <solve-binary> [iterations]
#
# Every tracked corpus instance is FEASIBLE except gas11, which is unbounded.
# So any `Infeasible` verdict, or an `Unbounded` on anything but gas11, is a
# FALSE certificate. gas11 is run with presolve OFF, because presolve catches it
# by inspection and would otherwise hide whether PDLP's own detector works.
#
# Run it with each platform's binary. The reason this script exists: the
# earlier "1e-6 gives zero false verdicts" was measured on the Windows build
# only, and the Linux build turned out to disagree.
set -u
SOLVE=${1:?usage: cert_sweep.sh <solve-binary> [iterations]}
ITERS=${2:-30000}
cd "$(dirname "$0")/.." || exit 1

for tol in 1e-4 1e-6 1e-8; do
  false_verdicts=0
  detail=""
  for f in tests/data/netlib/*.mps; do
    name=$(basename "$f" .mps)
    extra=""
    [ "$name" = gas11 ] && extra="--presolve=0"
    st=$("$SOLVE" "$f" --method=pdlp --pdlp-max-iter="$ITERS" --pdlp-cert-tol="$tol" $extra 2>&1 |
         grep -E '^status=' | cut -d= -f2)
    if [ "$name" = gas11 ]; then
      [ "$st" = Unbounded ] || detail="$detail gas11:MISSED($st)"
    elif [ "$st" = Infeasible ] || [ "$st" = Unbounded ]; then
      false_verdicts=$((false_verdicts + 1))
      detail="$detail $name:$st"
    fi
  done
  echo "tol=$tol  false_verdicts=$false_verdicts ${detail:- (gas11 detected, no false verdicts)}"
done
