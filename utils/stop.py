import subprocess

hashes = [
"18037383",
"18037382",
"18037381",
"18037380",
"18037379",
"18037378",
"18037377",
"18037376",
"18037375",
"18037374",
]

for hash_value in hashes:
    # Define the command to run the sbatch job with the hash value
    sbatch_cmd = f"scancel {hash_value}"

    # Submit the sbatch job using subprocess
    subprocess.call(sbatch_cmd.split())
