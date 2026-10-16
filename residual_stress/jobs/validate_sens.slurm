#!/bin/bash
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=rs_valsens
#SBATCH --time=06:00:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/residual_stress/jobs/valsens_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/residual_stress/jobs/valsens_%j.err
# Validation of the Enzyme first/second-order stress sensitivities against central finite differences of nominal runs
# (first 6 increments = the window where plasticity starts). Two streams of 16 threads.
R=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme
BIN=$R/residual_stress/mech/src/rs_mech; SNAP=$R/residual_stress/results/thermal_nominal/snap; V=$R/residual_stress/results/val; mkdir -p $V
cd $R/inputs
H=1e-3; NINC=6
run() { name=$1; shift; mkdir -p $V/$name
  env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 $BIN -m full_bridge.mesh -tie tie_weights_full.txt -snap $SNAP -o $V/$name -ninc $NINC "$@" > $V/$name/run.log 2>&1; echo "$name done $(date +%T)"; }
OPT=(-sE -snu -sa -ssy -sH)
sc() { # build the 5 scale args from a +/- pattern: sc i s_i [j s_j]
  local a=(1 1 1 1 1); a[$1]=$2; [ -n "${3:-}" ] && a[$3]=$4
  echo "-sE ${a[0]} -snu ${a[1]} -sa ${a[2]} -ssy ${a[3]} -sH ${a[4]}"; }
P=$(python3 -c "print(1+$H)"); M=$(python3 -c "print(1-$H)")
streamA() { run ad_d2 -dorder 2; run nominal -dorder 0
  for i in 0 1 2; do run p${i}_plus -dorder 0 $(sc $i $P); run p${i}_minus -dorder 0 $(sc $i $M); done
  for pr in "0 3" "1 2"; do set -- $pr; run m$1$2_pp -dorder 0 $(sc $1 $P $2 $P); run m$1$2_pm -dorder 0 $(sc $1 $P $2 $M); run m$1$2_mp -dorder 0 $(sc $1 $M $2 $P); run m$1$2_mm -dorder 0 $(sc $1 $M $2 $M); done; }
streamB() { for i in 3 4; do run p${i}_plus -dorder 0 $(sc $i $P); run p${i}_minus -dorder 0 $(sc $i $M); done
  for pr in "3 4" "0 4"; do set -- $pr; run m$1$2_pp -dorder 0 $(sc $1 $P $2 $P); run m$1$2_pm -dorder 0 $(sc $1 $P $2 $M); run m$1$2_mp -dorder 0 $(sc $1 $M $2 $P); run m$1$2_mm -dorder 0 $(sc $1 $M $2 $M); done; }
streamA & streamB & wait
echo VALIDATION_RUNS_DONE
