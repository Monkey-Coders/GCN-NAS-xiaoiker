q:
	squeue -u maxts

qg:
	squeue -u maxts -p GPUQ

qc:
	squeue -u maxts -p CPUQ
	
qs:
	squeue --start -u maxts
	
r:
	sbatch zzz_slurm/job.slurm

t:
	sbatch zzz_slurm/task.slurm

z:
	sbatch zzz_slurm/zc.slurm

c:
	sbatch zzz_slurm/cpu.slurm

zip:
	sbatch zzz_slurm/zip.slurm

s:
	scancel $(id)

sa:
	scancel -u maxts -p GPUQ

gpu:
	nvidia-smi

zsa:
	scancel -u zuimran -p GPUQ

zq:
	squeue -u zuimran

zqg:
	squeue -u zuimran -p GPUQ

zqc:
	squeue -u zuimran -p CPUQ
	
zqs:
	squeue --start -u zuimran

niko:
	squeue -u nikolard

nikos:
	squeue --start -u nikolard

ian:
	squeue -u iaevange

salar:
	squeue -u salara