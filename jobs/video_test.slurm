#!/bin/bash
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=fbe_vidtest
#SBATCH --time=00:40:00
#SBATCH --output=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/video_test_%j.out
#SBATCH --error=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme/jobs/video_test_%j.err
R=/work/gyj329/MFEM/implementations/full_bridge_20_param_single_enzyme; O=$R/results/smoke_dump; PY=/home/gyj329/anaconda3/bin/python
$PY $R/figures/make_bridge_video.py deriv $O/dump $O/test_deriv.mp4 10 4 1
$PY $R/figures/make_bridge_video.py temp  $O/dump $O/test_temp.mp4 10 4 1
ls -la $O/*.mp4
