import argparse
import os
import subprocess
import glob
import json
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='input file', required=True)
    parser.add_argument('--transition', help='transition type',
                        choices=['ares', 'box'], default='ares')
    parser.add_argument('--timeout', help='timeout',
                        required=False, default=120, type=int)
    args = parser.parse_args()
    # call ares_move.py or box_move.py based on the transition type
    python_file = 'ares_move.py' if args.transition == 'ares' else 'box_move.py'

    # call the python file with the given arguments
    for type in ['DFS', 'BFS', 'A*', 'UCS']:
        try:
            print(f'Running {type} search')
            subprocess.run(f'python {python_file} --input {args.input} --type {
                        type}', timeout=args.timeout, check=True)
        except subprocess.TimeoutExpired:
            print('Timeout: The search is taking too long')
        except subprocess.CalledProcessError as e:
            print('Error:', e)
    
    # read the json output files

    current_dir = os.path.dirname(args.input)
    output_dir = os.path.join(current_dir, 'output')
    test_name = os.path.basename(args.input).split('.')[0]
    print(f'Output directory: {output_dir}')
    text_output_file = os.path.join(current_dir, 'output', f'{test_name}_output.txt')
    with open(text_output_file, 'w') as f:
        output_files = glob.glob(os.path.join(output_dir, '*.json'))
        output_files.sort()
        for file in output_files:
            if test_name not in file:
                continue
            algorithm_type = file.split('_')[-1].split('.')[0]
            algorithm_type = algorithm_type.replace('star', '*')
            f.write(f'{algorithm_type}\n')
            with open(file, 'r') as json_file:
                json_data = json.load(json_file)
                path = json_data['node']
                num_steps = len(path)
                weight = json_data['cost']
                node = json_data['node_number']
                time = json_data['time'] # time is currently in seconds
                mem = json_data['peak_memory'] # memory is in bytes
                # Should output the following format
                # Steps: 16, Weight: 695, Node: 4321, Time (ms): 58.12, Memory (MB): 12.56 
                # uLulDrrRRRRRRurD
                f.write(f'Steps: {num_steps}, Weight: {weight}, Node: {node}, Time (ms): {time*1000:.2f}, Memory (MB): {mem/1024/1024:.2f}\n')
                f.write(f'{path}\n')





