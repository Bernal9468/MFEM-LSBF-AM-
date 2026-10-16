#!/bin/bash
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --job-name=oti_d1
#SBATCH --time=2-12:00:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/otid1_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/otid1_%j.err
# OTI first order (20 params) on the full window in ONE pass (no checkpoint restart), same source as the Enzyme build
# (Tw IC seed included) so Tw is comparable.  Output: results/oti_first_order/
set -uo pipefail
R=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme
IN=$R/inputs; D=$R/results/oti_first_order; mkdir -p $D
export OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=40 MKL_NUM_THREADS=40 NEWTON_LS=auto
cp $R/src/fbko_blend_order1 $D/bin; cd $IN
C="-m full_bridge.mesh -tie tie_weights_full.txt -nseg 20 -nmax 200 -solver pardiso -probex 2.5 -probey 0.0 -pf 1 \
   -spread 2 -gaf 0.1 -gar 0.1 -gb 0.1 -gc 0.1 -gff 1.0 -gfr 1.0 -fdt 1e-6 -ts 1.2522 -tf 1.25660 -powderstart -zshift 0.18"
$D/bin $C -dorder 1 -pcsv $D/d1_probe.csv -scsv $D/d1_sens.csv > $D/d1.log 2>&1
tail -n 12 $D/d1.log | grep -q "Wall time" && touch $D/DONE
