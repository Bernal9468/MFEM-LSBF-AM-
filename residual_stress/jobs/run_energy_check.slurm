#!/bin/bash
#SBATCH --partition=compute1,compute2,compute3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=rs_energy
#SBATCH --time=06:00:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/residual_stress/jobs/energy_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/residual_stress/jobs/energy_%j.err
# MFEM, NO powder, full 4400-step window, final temperature field saved: (a) nominal absorptivity, (b) absorptivity scale 0.5.
R=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme
IN=$R/inputs; V=$R/residual_stress/results/energy_check; mkdir -p $V
export OMP_PROC_BIND=false NEWTON_LS=auto
cd $IN
run() { name=$1; shift; d=$V/$name; mkdir -p $d
  env OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 $R/residual_stress/src/fbke_enzyme -m full_bridge.mesh -tie tie_weights_full.txt -nseg 20 -nmax 200 -solver pardiso -probex 2.5 -probey 0.0 -pf 400 \
     -spread 2 -gaf 0.1 -gar 0.1 -gb 0.1 -gc 0.1 -gff 1.0 -gfr 1.0 -fdt 1e-6 -ts 1.2522 -zshift 0.18 -tf 1.2566 -dorder 0 -snapdir $d/snap -snap-every 4400 -pcsv $d/probe.csv "$@" > $d/run.log 2>&1; echo "$name done"; }
run eta1.0 -etascale 1.0 & run eta0.5 -etascale 0.5 & wait; echo ENERGY_CHECK_DONE
