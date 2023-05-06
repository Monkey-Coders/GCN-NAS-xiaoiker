import subprocess

hashes = [
"18145113",
"18145112",
"18145111",
"18145110",
"18145109",
"18145108",
"18145107",
"18145106",
"18145105",
"18145104",
"18145103",
"18145102",
"18145101",
"18145100",
"18145099",
"18145098",
"18145097",
"18145096",
"18145095",
"18145094",
"18145093",
"18145092",
"18145083",
"18145084",
"18145085",
"18145086",
"18145087",
"18145088",
"18145089",
"18145090",
"18145091",
]

for hash_value in hashes:
    # Define the command to run the sbatch job with the hash value
    sbatch_cmd = f"scancel {hash_value}"

    # Submit the sbatch job using subprocess
    subprocess.call(sbatch_cmd.split())
