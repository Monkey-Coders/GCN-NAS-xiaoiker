#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100"
#SBATCH --mem=30G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=ZC
#SBATCH --output=ZC.out
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
conda activate zaim
echo "Pip install requirements"
pip install -r requirements.txt --user -q
pip install thop --user -q
echo "Installing PyTorch"
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html --user -q

python3 merge_architectures.py
# python3 calculate_zc_scores.py --start 0 --end 500
# python3 zero_cost_proxies/calculate_correlation.py
# python3 zero_cost_proxies/calculate_vote.py

uname -a
