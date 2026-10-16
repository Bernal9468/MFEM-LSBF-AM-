#!/bin/bash
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=rs_fdfull
#SBATCH --time=1-18:00:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/residual_stress/jobs/fdfull_%x_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/residual_stress/jobs/fdfull_%x_%j.err
# Central finite differences of the NOMINAL thermo-mechanical solve over the FULL 286-increment history (h = 1e-4 in scale),
# to check the Enzyme first/second-order derivatives of the whole final stress field. Usage: sbatch fd_full.slurm A|B
R=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme
BIN=$R/residual_stress/mech/src/rs_mech; SNAP=$R/residual_stress/results/thermal_nominal/snap; V=$R/residual_stress/results/fd_full; mkdir -p $V
cd $R/inputs
H=1e-4; P=$(python3 -c "print(1+$H)"); M=$(python3 -c "print(1-$H)")
run() { name=$1; shift; [ -f $V/$name/DONE ] && return; mkdir -p $V/$name
  env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 $BIN -m full_bridge.mesh -tie tie_weights_full.txt -snap $SNAP -o $V/$name -dorder 0 -px 2.5 -py 0.05 -pz 0.18 -ckpt-every 1000 "$@" > $V/$name/run.log 2>&1 && touch $V/$name/DONE; echo "$name done $(date +%T)"; }
sc() { local a=(1 1 1 1 1); a[$1]=$2; [ -n "${3:-}" ] && a[$3]=$4; echo "-sE ${a[0]} -snu ${a[1]} -sa ${a[2]} -ssy ${a[3]} -sH ${a[4]}"; }
pm() { run p$1_plus $(sc $1 $P); run p$1_minus $(sc $1 $M); }
mx() { for s in "pp $P $P" "pm $P $M" "mp $M $P" "mm $M $M"; do set -- $s $1 $2; done; }
mixed() { i=$1; j=$2; run m${i}${j}_pp $(sc $i $P $j $P); run m${i}${j}_pm $(sc $i $P $j $M); run m${i}${j}_mp $(sc $i $M $j $P); run m${i}${j}_mm $(sc $i $M $j $M); }
case "$1" in
 A) ( pm 0; pm 1; mixed 0 3 ) & ( pm 2; mixed 1 2 ) & wait ;;
 B) ( pm 3; mixed 3 4 ) & ( pm 4; mixed 0 4 ) & wait ;;
esac
echo FD_FULL_$1_DONE
