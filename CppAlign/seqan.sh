#!/bin/bash
#SBATCH --qos=general-compute
#SBATCH --cluster=ub-hpc
#SBATCH --time=48:00:00
#SBATCH --partition=general-compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=12
#SBATCH --mem-per-cpu=4000
#SBATCH --mail-user=jchen378@buffalo.edu
#SBATCH --mail-type=END
#SBATCH --job-name=seqan_nw
#SBATCH --output=pnw.out
#SBATCH --error=pnw.err

tic=`date +%s`
echo "Start Time = "`date`

cd $SLURM_SUBMIT_DIR
echo "working directory = "$SLURM_SUBMIT_DIR
ulimit -s unlimited

cd /projects/academic/yijunsun/jian/CppKmer
./align /projects/academic/yijunsun/jian/ITS/qiime/chosen_500_wide_0.txt 0

echo "All Done!"

echo "End Time = "`date`
toc=`date +%s`

elapsedTime=`expr $toc - $tic`
echo "Elapsed Time = $elapsedTime seconds"
