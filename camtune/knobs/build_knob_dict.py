import os
import json
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def update_json(target_list, source_file_path):
    target_list_path = os.path.join(BASE_DIR, 'selection_lists', f'{target_list}.txt')

    with open(target_list_path, 'r') as file:
        knob_list = file.read().splitlines()
    knob_list = [line.strip() for line in knob_list if len(line.strip()) > 0]
    target_knob_dict = {}
    
    # Load the source file from which to take updates
    with open(source_file_path, 'r') as file:
        source_data = json.load(file)
    
    # Update target_data with values from source_data based on matching keys
    for knob in knob_list:
        if knob in source_data:
            target_knob_dict[knob] = source_data[knob]
        else:
            print(f'Knob {knob} not found in {source_file_path}')

    # Write the updated data back to the target file
    target_file_path = os.path.join(BASE_DIR, f'pg_{target_list}.json')
    with open(target_file_path, 'w') as file:
        json.dump(target_knob_dict, file, indent=4, sort_keys=True)

    print(f"Build {target_file_path} using data from {target_list_path} and {source_file_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--knob_list', '-l', type=str)
    parser.add_argument('--source_name', '-s', type=str, default='llamatune_90')
    args = parser.parse_args()

    
    source_file_path = os.path.join(BASE_DIR, f'{args.source_name}.json')
    update_json(args.knob_list, source_file_path)