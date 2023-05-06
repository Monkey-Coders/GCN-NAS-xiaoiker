#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="V100|P100"
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=JOB
#SBATCH --output=JOB.out
#SBATCH --mail-user=zuimran@stud.ntnu.no
#SBATCH --mail-type=NONE
#SBATCH --export=model_hash,path

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

module purge
module load Python/3.9.6-GCCcore-11.2.0

echo "######################"
echo "Copying data"
cp -r data model_data/$model_hash
echo "######################"

nvidia-smi

echo "Running warmup with hash $model_hash"
# python3 zaim.py --hash $model_hash --path $path --start_epoch $start_epoch --weights $weights
python3 warmup.py --hash $model_hash --path $path
python3 warmup.py --hash $model_hash --path $path --overide "epe_nas|grad_sign|grasp"

echo "######################"
echo "Deleting data"
rm -rf model_data/$model_hash
uname -a
