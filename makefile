r:
	chmod u+x job.slurm && sbatch job.slurm

q:
	squeue -u maxts

s:
	scancel $(id)

sa:
	scancel -u maxts

t:
	tail -f -n 1 GCN-NAS.out 

gpu:
	nvidia-smi
