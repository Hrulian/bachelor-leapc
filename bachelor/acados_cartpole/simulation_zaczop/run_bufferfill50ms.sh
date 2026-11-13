#!/usr/bin/env bash
# Buffer-fill ablation at the 50 ms control rate, one wandb group per variant
# (5 seeds each). All groups run at the same time (5 seeds x 3 groups = 15 procs).
#
# Same everywhere: reward cos_bonus_spin3 (HARDWARE_REWARD in my_helpers.py),
# dt = 0.05 s, 200 steps/episode (10 s, as max_ep_steps in real_zopfill.py at
# 50 ms), 10k env steps, seeds 0-4.
# Buffer fill: untrained SAC-ZOP actor fills steps 0..1000, pure SAC drives
# from 1000..10000. In all groups SAC learns (UTD 20) from env step 200 on, i.e.
# the fill groups already train in the background while the filler drives.
#
#   bash run_bufferfill50ms.sh            # all groups
#   bash run_bufferfill50ms.sh 02 03      # only groups whose name starts with 02 / 03
set -u
# job control: background jobs would otherwise start with SIGINT ignored, and
# the launchers + their runs could not be shut down cleanly with Ctrl+C / kill -INT
set -m
export PYTHONUNBUFFERED=1

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY="$ROOT/.venv/bin/python"

# the venv's editable acados install still points at the old repo location
export ACADOS_SOURCE_DIR="$ROOT/external/acados"
export ACADOS_INSTALL_DIR="$ROOT/external/acados"
export LD_LIBRARY_PATH="$ROOT/external/acados/lib"
export PYTHONPATH="$ROOT/external/acados/interfaces/acados_template"

PROJECT="Sim-BufferFill-50ms"
RUNS="$HERE/runs_bufferfill50ms"
COMMON=(--rewards cos_bonus_spin3 --seeds 0 1 2 3 4 --steps 10000 --workers 5
        --torch-threads 1 --wandb-project "$PROJECT"
        --dt 0.05 --max-ep-steps 200)
LEARN_START=200
FILL=(--switch-step 1000 --fill-train-start "$LEARN_START")
SAC=(--train-start "$LEARN_START")

# group name | launcher | extra args
VARIANTS=(
  "01_SAC_pure_learn200|run_parallel_sac.py|"
  "02_fill_full_learn200|run_parallel_zopfill.py|--planner full"
  "03_fill_full_noPosConstr_learn200|run_parallel_zopfill.py|--planner full --no-pos-constraint"
)

cd "$HERE"
mkdir -p "$RUNS"
pids=()
for entry in "${VARIANTS[@]}"; do
  IFS='|' read -r group launcher extra <<< "$entry"
  if [ $# -gt 0 ]; then
    match=0
    for sel in "$@"; do [[ "$group" == "$sel"* ]] && match=1; done
    [ $match -eq 1 ] || continue
  fi
  args=("${COMMON[@]}" --group "$group" --runs-dir "$RUNS/$group")
  [ "$launcher" = "run_parallel_zopfill.py" ] && args+=("${FILL[@]}")
  [ "$launcher" = "run_parallel_sac.py" ] && args+=("${SAC[@]}")
  # shellcheck disable=SC2206
  [ -n "$extra" ] && args+=($extra)
  echo "=== $(date '+%F %T') start $group"
  "$PY" "$launcher" "${args[@]}" > "$RUNS/$group.log" 2>&1 &
  pids+=($!)
done
trap 'kill -INT "${pids[@]}" 2>/dev/null; wait' INT TERM
wait
echo "=== $(date '+%F %T') all groups done"
