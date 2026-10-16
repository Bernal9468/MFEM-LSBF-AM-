#!/bin/bash
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=fbe_smoked
#SBATCH --time=00:40:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/smoke_dump_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/smoke_dump_%j.err
R=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme; IN=$R/inputs; O=$R/results/smoke_dump; mkdir -p $O/dump; cd $IN; export OMP_STACKSIZE=512M
CORES=$(/work/gyj329/MFEM/implementations/bare_plate_enzyme_mechanical/jobs/alloc_cores.sh | tr ' ' ',')
C="-m full_bridge.mesh -tie tie_weights_full.txt -nseg 20 -nmax 200 -solver pardiso -probex 2.5 -probey 0.0 -pf 1 -spread 2 -gaf 0.1 -gar 0.1 -gb 0.1 -gc 0.1 -gff 1.0 -gfr 1.0 -fdt 1e-6 -ts 1.2522 -tf 1.252215 -powderstart -zshift 0.18"
OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 NEWTON_LS=auto FIELD_DUMP_DIR=$O/dump FIELD_DUMP_EVERY=5 taskset -c $CORES $R/src_dump/fbke_enzyme $C -dorder 1 -pcsv $O/p.csv -scsv $O/s.csv > $O/log.txt 2>&1
tail -n 3 $O/log.txt | cut -c1-160; ls -la $O/dump
