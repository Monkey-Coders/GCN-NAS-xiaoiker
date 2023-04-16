q:
	squeue -u zuimran
	
qs:
	squeue --start -u zuimran
	
r:
	chmod u+x zzz_slurm/job.slurm && sbatch zzz_slurm/job.slurm

z:
	chmod u+x zzz_slurm/zc.slurm && sbatch zzz_slurm/zc.slurm

c:
	chmod u+x zzz_slurm/cpu.slurm && sbatch zzz_slurm/cpu.slurm

s:
	scancel $(id)

sa:
	scancel -u zuimran -p GPUQ

t:
	tail -f -n 1 GCN-NAS.out 

gpu:
	nvidia-smi
