import os
import glob
import random

original_folder = 'official_test'
new_folder = 'demo_test'

# create a new folder
if not os.path.exists(new_folder):
    os.makedirs(new_folder)


file_list = glob.glob(f'{original_folder}/*.txt')
file_list.sort()

for file in file_list:
    file_name = os.path.basename(file)
    print(file_name)
    with open(file, 'r') as f:
        lines = f.readlines()
        # first line consists of multiple integers
        # read the integers
        weights = list(map(int, lines[0].split()))
        size = len(weights)
        # create new random weight values
        new_weights = [random.randint(1, 100) for _ in range(size)]
        # create a new file in the new folder
        with open(os.path.join(new_folder, file_name), 'w') as f:
            f.write(' '.join(map(str, new_weights)))
            f.write('\n')
            for line in lines[1:]:
                f.write(line)