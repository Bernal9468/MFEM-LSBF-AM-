#!/bin/bash
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=fbe_buildviz
#SBATCH --time=01:00:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/buildviz_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/buildviz_%j.err
cd /work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/src
cp fbke_enzyme /tmp/fbke_enzyme.keep_$$ 2>/dev/null
make -j8 enzyme 2>&1 | grep -v "^g++\|^/usr" | tail -40
cp fbke_enzyme fbke_enzyme_viz
cp /tmp/fbke_enzyme.keep_$$ fbke_enzyme 2>/dev/null   # restore the validated binary
ls -la fbke_enzyme fbke_enzyme_viz
