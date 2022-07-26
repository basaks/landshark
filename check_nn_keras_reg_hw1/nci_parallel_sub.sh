#!/bin/bash
#PBS -q normal
#PBS -l ncpus=384
#PBS -l jobfs=100GB
#PBS -l walltime=04:00:00
#PBS -l mem=1550GB
#PBS -P ge3
#PBS -l storage=gdata/dg9+gdata/dz56+gdata/ge3
#PBS -l wd
#PBS -j oe

module load nci-parallel/1.0.0a
export ncores_per_task=1
export ncores_per_numanode=48

mpirun -np $((PBS_NCPUS/ncores_per_task)) --map-by ppr:$((ncores_per_numanode/ncores_per_task)):NUMA:PE=${ncores_per_task} nci-parallel --input-file cmds.txt --timeout 36000

# to create cmds.txt use the for loop like this:
# for i in {1..384}; do echo $i; echo ./pred_parts.sh $i 384 >> cmds.txt; done
# generate merge statement
# merge statement
# echo gdal_merge.py \\  > merge.txt
# for i in {1..384}; do echo nn_regression_keras_global_local_model_6of10/predictions_log_cond_1_${i}of384.tif >> merge.txt \\ ; done
# bash merge.txt
# echo -o predictions_log_cond_1.tif >> merge.txt

