#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=48:20:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100"
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=JOB
#SBATCH --output=#JOB.out
#SBATCH --mail-user=zuimran@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --export=model_hash,weights,start_epoch,path

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
echo "Pip install requirements"
pip install -r requirements.txt --user -q
echo "######################"
echo "Installing thop"
pip install thop --user -q
echo "######################"
echo "Installing PyTorch"
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html --user -q

echo "######################"
echo "Wandb login"
wandb login a049ba80f44a62c306d6bed8249174b30801d69e

echo $model_hash
echo $start_epoch
echo $weights

echo "######################"
echo "Copying data"
cp -r data model_data/$model_hash
echo "######################"
echo "Running zaim with hash $model_hash and start epoch $start_epoch and weights $weights"
python3 zaim.py --hash $model_hash --path $path --start_epoch $start_epoch --weights $weights
echo "######################"
echo "Deleting data"
rm -rf model_data/$model_hash
uname -a
