conda env create -n max -f environment.yml
conda activate max
pip install --upgrade pip
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html

echo "Logging in to wandb"
python3 -m wandb login "API-KEY"
echo "Finished logging in to wandb"