r:
	chmod u+x job.slurm && sbatch job.slurm

z:
	chmod u+x zc.slurm && sbatch zc.slurm

q:
	squeue -u zuimran

s:
	scancel $(id)

sa:
	scancel -u zuimran

t:
	tail -f -n 1 GCN-NAS.out 

gpu:
	nvidia-smi
