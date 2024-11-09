import os
import glob
import subprocess

def benchmark(folder_path):
    # get all the files in the folder
    files = glob.glob(os.path.join(folder_path, '*.txt'))
    files.sort()
    #types = ['BFS', 'A*', 'UCS']
    types = ['A*']
    transitions = ['ares', 'box']
    timeout = 60 * 5 # 5 minutes
    for file in files:
        if 'diff-rating' not in file:
            continue
        # call the search.py file with the input file
        for transition in transitions:
            for type in types:
                print(f'Running search.py with input file: {file}, transition: {transition}, type: {type}')
                try:
                    subprocess.run(f'python search.py --input {file} --transition {transition} --type {type} --timeout {timeout}', check=True)
                except subprocess.TimeoutExpired:
                    print('Timeout: The search is taking too long')
                except subprocess.CalledProcessError as e:
                    print('Error:', e)

if __name__ == '__main__':
    benchmark('official_test')