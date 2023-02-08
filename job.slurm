#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=12:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:V100:2
#SBATCH --mem=20G
#SBATCH --job-name=GCN-NAS
#SBATCH --output=GCN-NAS.out
#SBATCH --mail-user=maxts@stud.ntnu.no
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

echo "Creating conda environment"
conda env create -n max -f environment.yml
echo "Activating conda environment"
conda activate max
echo "Pip install requirements"
pip install -r requirements.txt --user
echo "Installing PyTorch"
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html --user
echo "Installing wandb"

module load Python/3.8.6-GCCcore-10.2.0
pip install wandb --user -q

echo "Logging in to wandb"
wandb login "API_KEY"
echo "Finished logging in to wandb"

python3 gcn_search.py

uname -a