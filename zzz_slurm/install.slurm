#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|V100|P100"
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=INSTALL
#SBATCH --output=INSTALL.out
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
module load Python/3.9.6-GCCcore-11.2.0

# echo "Creating conda environment"
# conda env create -n promp
# echo "Activating conda environment"
# eval "$(conda shell.bash hook)"
# conda activate promp
# conda update -n base -c defaults conda
# conda update --all
echo "Pip install requirements"
pip install -r requirements.txt --user -q
pip install thop --user -q
echo "Installing PyTorch"
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html --user -q

wandb login a049ba80f44a62c306d6bed8249174b30801d69e

uname -a
