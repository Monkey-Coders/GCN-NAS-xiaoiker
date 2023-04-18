q:
	squeue -u zuimran

qg:
	squeue -u zuimran -p GPUQ

qc:
	squeue -u zuimran -p CPUQ
	
qs:
	squeue --start -u zuimran
	
r:
	chmod u+x zzz_slurm/job.slurm && sbatch zzz_slurm/job.slurm

t:
	chmod u+x zzz_slurm/task.slurm && sbatch zzz_slurm/task.slurm

z:
	chmod u+x zzz_slurm/zc.slurm && sbatch zzz_slurm/zc.slurm

c:
	chmod u+x zzz_slurm/cpu.slurm && sbatch zzz_slurm/cpu.slurm

s:
	scancel $(id)

sa:
	scancel -u zuimran -p GPUQ

gpu:
	nvidia-smi
