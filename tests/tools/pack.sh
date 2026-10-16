#!/bin/bash
# pack.sh -- run a table of driver runs side by side on one node, each pinned to its own block of cores.
#
# usage: pack.sh TABLE OUTDIR DEFAULT_BIN "COMMON ARGS" [CORES_PER_RUN]
#   TABLE lines:  name|ENV=val ENV2=val|extra driver args[|binary]     ('#' lines and blank lines are skipped)
# Each run writes OUTDIR/<name>.{probe.csv,sens.csv,log,status}; status is "exit=<rc> wall_s=<s> slot=<k>".
# Runs whose status already says exit=0 are skipped, so a killed job can simply be resubmitted.
# Needs IN_DIR (the driver's working directory) in the environment.
set -uo pipefail
TABLE=$1; OUT=$2; DEFBIN=$3; ARGS=$4; CORES=${5:-8}
: "${IN_DIR:?IN_DIR not set}"
mkdir -p "$OUT"
NCPU=${SLURM_CPUS_ON_NODE:-$(nproc)}
NSLOT=$(( NCPU / CORES )); (( NSLOT < 1 )) && NSLOT=1
echo "[pack] $(grep -cv '^\s*\(#\|$\)' "$TABLE") runs, $NSLOT slots x $CORES cores, node $(hostname)"

run_one() {
  local slot=$1 line=$2 name envs args bin
  IFS='|' read -r name envs args bin <<< "$line"
  bin=${bin:-$DEFBIN}
  local lo=$(( slot * CORES )) hi=$(( slot * CORES + CORES - 1 )) t0=$SECONDS
  rm -f "$OUT/$name".probe.csv "$OUT/$name".sens.csv "$OUT/$name".sens.csv.hess.csv
  # shellcheck disable=SC2086  # envs / ARGS / args are intentionally word-split
  ( cd "$IN_DIR" && env OMP_NUM_THREADS=$CORES MKL_NUM_THREADS=$CORES $envs taskset -c $lo-$hi "$bin" $ARGS $args \
      -pcsv "$OUT/$name.probe.csv" -scsv "$OUT/$name.sens.csv" ) > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "exit=$rc wall_s=$(( SECONDS - t0 )) slot=$slot" > "$OUT/$name.status"
  echo "[pack] done  $name exit=$rc ($(( SECONDS - t0 )) s)"
}

declare -a PID
while IFS= read -r line || [ -n "$line" ]; do
  [[ -z "${line// }" || "$line" == \#* ]] && continue
  name=${line%%|*}
  if grep -qs '^exit=0' "$OUT/$name.status"; then echo "[pack] skip  $name (already done)"; continue; fi
  while :; do
    for (( s = 0; s < NSLOT; s++ )); do
      if [ -z "${PID[$s]:-}" ] || ! kill -0 "${PID[$s]}" 2>/dev/null; then
        run_one "$s" "$line" < /dev/null &      # never let a run read the table from the loop's stdin
        PID[$s]=$!
        echo "[pack] start $name (slot $s)"
        continue 3
      fi
    done
    sleep 2
  done
done < "$TABLE"
wait
nbad=0
while IFS= read -r line || [ -n "$line" ]; do
  [[ -z "${line// }" || "$line" == \#* ]] && continue
  name=${line%%|*}
  grep -qs '^exit=0' "$OUT/$name.status" || { echo "[pack] FAILED $name: $(cat "$OUT/$name.status" 2>/dev/null || echo 'no status')"; nbad=$(( nbad + 1 )); }
done < "$TABLE"
echo "[pack] finished, $nbad failed run(s)"
exit $(( nbad > 0 ))
