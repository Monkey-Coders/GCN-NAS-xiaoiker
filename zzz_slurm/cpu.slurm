#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=01:15:00
#SBATCH --mem=40G
#SBATCH --job-name=CPU
#SBATCH --output=CPU.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mail-user=zuimran@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo “we are running from this directory: $SLURM_SUBMIT_DIR”
echo ” the name of the job is: $SLURM_JOB_NAME”
echo “Th job ID is $SLURM_JOB_ID”
echo “The job was run on these nodes: $SLURM_JOB_NODELIST”
echo “Number of nodes: $SLURM_JOB_NUM_NODES”
echo “We are using $SLURM_CPUS_ON_NODE cores”
echo “We are using $SLURM_CPUS_ON_NODE cores per node”
echo “Total of $SLURM_NTASKS cores”

# module purge
# module load Python/3.8.6-GCCcore-10.2.0
# pip install -r requirements.txt --user -q


# zip -r run.zip experiment/run
# python3 utils/move.py
# python3 utils/weighted_average.py
# python3 ZeroCostFramework/calculate_correlation_of_epochs.py
# python3 utils/plot.py
# python3 ZeroCostFramework/calculate_vote.py

uname -a