search:
	python gcn_search.py

create:
	conda env create -f environment.yml

env:
	echo "conda activate zaim"

exit:
	echo "conda deactivate"

tens:
	tensorboard --logdir Run\ 2/runs

refresh:
	cd ../../anaconda3/ && rm -r envs/

gpu:
	nvidia-smi

pip:
	pip install tqdm tensorboard tensorboardX pyyaml==5.3 scipy torchprofile

conda:
	conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

hop:
	chmod +x ./gcn_search.py && nohup ./gcn_search.py &

find:
	ps ax | grep ./gcn_search.py