import subprocess

hashes = [
    {"model_hash": "7577dd5ac267937420b307bc986c87653c617495799adb20b435a7766fc9aceb", "max_epoch": 7},
    # {"model_hash": "14670fcd61c44b030094d84e5b0e0ee3f6af9415f667e63f2608c8812d5ae307", "max_epoch": 4},
    # {"model_hash": "2ecc95faf6febc1db5ca5b2a06eed301a46289493dbe6fae5a94d5228e806432", "max_epoch": 5},
    # {"model_hash": "53e547b2366bd4d35f29ef8a54e421f41af141a665985e4faaa65cfe3bfec65c", "max_epoch": 8},
    # {"model_hash": "426ef50bcf50cec9b716771e4a38e083e27cdf75469d09e96d9cb1dce5cddcec", "max_epoch": 8}, 
    # {"model_hash": "0bdd428eda03fa10d2e8525a316a7123a4234f7ebb90312b6805d9b57ba57475", "max_epoch": 8}, 
    # {"model_hash": "c319902d9a61afd0f94767b51d2509fa76bd03059a14e35f3ed7bb535c9b3c00", "max_epoch": 8}, 
    # {"model_hash": "b54c72aa2a3ee8127276687cf82f4758651223b0ffe4cdb767fa82991559d739", "max_epoch": 8}, 
    # {"model_hash": "7d5d256beafd15cb88d57fdb6548ed30112e9f68880640d9f02d9cf49471ef6e", "max_epoch": 1},
    # {"model_hash": "ca0aa440154ef9d53cc8f36f7013508024e0d006395d526fbb0cf6653a1bdcac", "max_epoch": 2}, 
    # {"model_hash": "8c8a7607a030b03fd81780a7f174266c1971e974939f837c56b80acb501d35fb", "max_epoch": 4}
]

for hash_value in hashes:
    # Define the command to run the sbatch job with the hash value
    sbatch_cmd = f"sbatch --export=hash_value={hash_value['model_hash']} --export=start_epoch={hash_value['max_epoch']} --job-name={hash_value['model_hash']} --output=output/{hash_value['model_hash']}.out zzz_slurm/job.slurm"

    # Submit the sbatch job using subprocess
    subprocess.call(sbatch_cmd.split())
