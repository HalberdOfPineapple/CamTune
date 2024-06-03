import os
import json
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def update_json(target_file_path, source_file_path):
    # Load the target file which we will update
    with open(target_file_path, 'r') as file:
        target_data = json.load(file)
    
    # Load the source file from which to take updates
    with open(source_file_path, 'r') as file:
        source_data = json.load(file)
    
    # Update target_data with values from source_data based on matching keys
    for key in target_data.keys():
        if key in source_data:
            target_data[key] = source_data[key]
    
    # Write the updated data back to the target file
    with open(target_file_path, 'w') as file:
        json.dump(target_data, file, indent=4)

    print(f"Updated {target_file_path} using data from {source_file_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_name', '-t', type=str)
    parser.add_argument('--source_name', '-s', type=str, default='llamatune_90')
    args = parser.parse_args()

    target_file_path = os.path.join(BASE_DIR, f'{args.target_name}.json')
    source_file_path = os.path.join(BASE_DIR, f'{args.source_name}.json')

    update_json(target_file_path, source_file_path)