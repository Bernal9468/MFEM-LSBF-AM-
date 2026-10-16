#!/bin/bash
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=rs_fdad
#SBATCH --time=1-18:00:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/residual_stress/jobs/fdad_%x_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/residual_stress/jobs/fdad_%x_%j.err
# Second-order check without second differences: run the FIRST-ORDER Enzyme solve at p_i(1 +/- h) and difference the
# first-order stress derivatives:  d2 sigma/dp_i dp_j  ~=  [d sigma/dp_j (p_i+h) - d sigma/dp_j (p_i-h)] / (2h).
# usage: sbatch fd_of_ad.slurm <param index 0..4>     (0 E, 1 nu, 2 CTE, 3 sigma_y, 4 H)
R=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme
BIN=$R/residual_stress/mech/src/rs_mech; SNAP=$R/residual_stress/results/thermal_nominal/snap; V=$R/residual_stress/results/fd_of_ad; mkdir -p $V
cd $R/inputs
H=1e-4; P=$(python3 -c "print(1+$H)"); M=$(python3 -c "print(1-$H)"); i=$1
sc() { local a=(1 1 1 1 1); a[$1]=$2; echo "-sE ${a[0]} -snu ${a[1]} -sa ${a[2]} -ssy ${a[3]} -sH ${a[4]}"; }
run() { name=$1; shift; mkdir -p $V/$name
  env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 $BIN -m full_bridge.mesh -tie tie_weights_full.txt -snap $SNAP -o $V/$name -dorder 1 -px 2.5 -py 0.05 -pz 0.18 -tolr 1e-12 -tola 1e-9 -ckpt-every 1000 "$@" > $V/$name/run.log 2>&1 && touch $V/$name/DONE; echo "$name done $(date +%T)"; }
run p${i}_plus $(sc $i $P) & run p${i}_minus $(sc $i $M) & wait
echo FDAD_${i}_DONE
