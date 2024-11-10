# read run-test.txt
# each line is a command
# read the command and run the command at L to R in subprocess

import subprocess
import os
import argparse

def run(L, R):
    with open('run_test_fix.txt', 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i >= L and i < R:
                # use subprocess to run the command
                print(f'Running command: {line}')
                subprocess.run(line, shell=True)
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--L', type=int, default=0)
    parser.add_argument('--R', type=int, default=0)
    args = parser.parse_args()
    run(args.L, args.R)