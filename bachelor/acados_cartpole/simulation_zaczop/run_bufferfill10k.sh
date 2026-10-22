#!/usr/bin/env bash
# Buffer-fill ablation, 10k env steps, one wandb group per variant (5 seeds each).
# Groups run one after another; the 5 seeds of a group run in parallel.
#
# Same everywhere: reward cos_bonus_spin3 (HARDWARE_REWARD in my_helpers.py),
# sim env defaults (dt = 0.01 s, 1000 steps/episode), 10k steps, seeds 0-4.
# Buffer fill: untrained SAC-ZOP actor fills steps 0..5000 (SWITCH_STEP of
# real_zopfill.py), pure SAC drives + learns from 5000..10000.
#
#   bash run_bufferfill10k.sh            # all groups
#   bash run_bufferfill10k.sh 05 06      # only groups whose name starts with 05 / 06
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY="$ROOT/.venv/bin/python"

# the venv's editable acados install still points at the old repo location
export ACADOS_SOURCE_DIR="$ROOT/external/acados"
export ACADOS_INSTALL_DIR="$ROOT/external/acados"
export LD_LIBRARY_PATH="$ROOT/external/acados/lib"
export PYTHONPATH="$ROOT/external/acados/interfaces/acados_template"

PROJECT="Sim-BufferFill-10k"
RUNS="$HERE/runs_bufferfill10k"
COMMON=(--rewards cos_bonus_spin3 --seeds 0 1 2 3 4 --steps 10000 --workers 5
        --torch-threads 1 --wandb-project "$PROJECT")
FILL=(--switch-step 5000)

# group name | launcher | extra args
VARIANTS=(
  "01_SAC_pure|run_parallel_sac.py|"
  "02_fill_full|run_parallel_zopfill.py|--planner full"
  "03_fill_fullcart|run_parallel_zopfill.py|--planner fullcart"
  "04_fill_cart|run_parallel_zopfill.py|--planner cart"
  "05_fill_du0_d100|run_parallel_zopfill.py|--planner du0 --d-max 100"
  "06_fill_du0_d50|run_parallel_zopfill.py|--planner du0 --d-max 50"
  "07_fill_full_noPosConstr|run_parallel_zopfill.py|--planner full --no-pos-constraint"
  "08_fill_cart_noPosConstr|run_parallel_zopfill.py|--planner cart --no-pos-constraint"
  "09_fill_random|run_parallel_zopfill.py|--buffer-fillers random"
)

cd "$HERE"
for entry in "${VARIANTS[@]}"; do
  IFS='|' read -r group launcher extra <<< "$entry"
  if [ $# -gt 0 ]; then
    match=0
    for sel in "$@"; do [[ "$group" == "$sel"* ]] && match=1; done
    [ $match -eq 1 ] || continue
  fi
  args=("${COMMON[@]}" --group "$group" --runs-dir "$RUNS/$group")
  [ "$launcher" = "run_parallel_zopfill.py" ] && args+=("${FILL[@]}")
  # shellcheck disable=SC2206
  [ -n "$extra" ] && args+=($extra)
  echo "=== $(date '+%F %T') start $group"
  "$PY" "$launcher" "${args[@]}"
  echo "=== $(date '+%F %T') end $group"
done
echo "=== $(date '+%F %T') all groups done"
