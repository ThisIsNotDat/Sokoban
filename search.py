import argparse
import os
import subprocess
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='input file', required=True)
    parser.add_argument('--transition', help='transition type', choices=['ares', 'box'], default='ares')
    parser.add_argument('--type', help='type of search', choices=['DFS', 'BFS', 'A*', 'UCS'], required=True)
    parser.add_argument('--output', help='output file', required=False, default=None)
    args = parser.parse_args()
    # call ares_move.py or box_move.py based on the transition type
    output  = args.output
    if output is not None:
        output = f'--output {output}'
    else:
        output = ' '
    if args.transition == 'ares':
        # use subprocess to call ares_move.py
        subprocess.run(f'python ares_move.py --input {args.input} --type {args.type} {output}', shell=True)
    else:
        # use subprocess to call box_move.py
        subprocess.run(f'python box_move.py --input {args.input} --type {args.type} {output}', shell=True)
    