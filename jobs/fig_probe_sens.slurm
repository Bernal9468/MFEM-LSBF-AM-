#!/bin/bash
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name=fbe_figps
#SBATCH --time=00:20:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/fig_probe_sens_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/fig_probe_sens_%j.err
/home/gyj329/anaconda3/bin/python /work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/figures/plot_probe_ranked_sensitivities.py
