q:
	squeue -u zuimran

qg:
	squeue -u zuimran -p GPUQ

qc:
	squeue -u zuimran -p CPUQ
	
qs:
	squeue --start -u zuimran
	
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
	scancel -u zuimran

gpu:
	nvidia-smi

max:
	squeue -u maxts

niko:
	squeue -u nikolard

nikos:
	squeue --start -u nikolard

ian:
	squeue -u iaevange

salar:
	squeue -u salara

mathias:
	squeue -u mathiaoh

pal:
	squeue -u paalamo

check:
	squeue -u maxts -t R > a.txt && squeue -u zuimran -h -t R >> a.txt && squeue -u nikolard -h -t R >> a.txt && squeue -u iaevange -h -t R >> a.txt && squeue -u salara -h -t R >> a.txt && squeue -u mathiaoh -h -t R >> a.txt && squeue -u paalamo -h -t R >> a.txt
	squeue -u maxts -t PD > b.txt && squeue -u zuimran -h -t PD >> b.txt && squeue -u nikolard -h -t PD >> b.txt && squeue -u iaevange -h -t PD >> b.txt && squeue -u salara -h -t PD >> b.txt && squeue -u mathiaoh -h -t PD >> b.txt && squeue -u paalamo -h -t PD >> b.txt
	python3 utils/retrain_diff.py