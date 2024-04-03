# Generate a file with 20000 random seeds for Stable Diffusion

import random

# Define the number of seeds to generate
num_seeds = 20000

# Generate random seeds in the range typically used for Stable Diffusion (1 to 2**32 - 1)
seeds = [str(random.randint(1, 2**32 - 1)) for _ in range(num_seeds)]

# Path to the file where seeds will be saved
file_path = 'random_seeds.txt'

# Write seeds to the file, each on a new line
with open(file_path, 'w') as file:
    file.write('\n'.join(seeds))
