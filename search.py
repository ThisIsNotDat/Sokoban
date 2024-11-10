import argparse
import os
import subprocess
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='input file', required=True)
    parser.add_argument('--transition', help='transition type',
                        choices=['ares', 'box'], default='ares')
    parser.add_argument('--type', help='type of search',
                        choices=['DFS', 'BFS', 'A*', 'UCS'], required=True)
    parser.add_argument('--output', help='output file',
                        required=False, default=None)
    parser.add_argument('--timeout', help='timeout',
                        required=False, default=120, type=int)
    args = parser.parse_args()
    # call ares_move.py or box_move.py based on the transition type
    output = args.output
    if output is not None:
        output = f'--output {output}'
    else:
        output = ' '
    python_file = 'ares_move.py' if args.transition == 'ares' else 'box_move.py'

    # call the python file with the given arguments
    try:
        subprocess.run(f'python {python_file} --input {args.input} --type {
                       args.type}  {output}', timeout=args.timeout, check=True)
    except subprocess.TimeoutExpired:
        print('Timeout: The search is taking too long')
    except subprocess.CalledProcessError as e:
        print('Error:', e)
