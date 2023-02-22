#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|V100"
#SBATCH --mem=40G
#SBATCH --job-name=TRAIN
#SBATCH --output=TRAIN.out
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

module purge
module load Anaconda3/2020.07
module load Python/3.8.6-GCCcore-10.2.0

echo "Creating conda environment"
conda env create -n promp -f environment.yml
echo "Activating conda environment"
conda activate max
echo "Pip install requirements"
pip install -r requirements.txt --user -q
pip install thop --user -q
echo "Installing PyTorch"
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html --user -q
# echo "Installing wandb"
# pip install wandb --user -q

# echo "Logging in to wandb"
# wandb login a049ba80f44a62c306d6bed8249174b30801d69e
# echo "Finished logging in to wandb"

python3 train_random_architectures.py --start 70 --end 101

uname -a