import os
import json
import argparse

from camtune.utils import KNOB_DIR

def extract_keys_from_json(keys_file_path, json_file_path, output_file_path):
    # Read the list of keys from the txt file
    with open(keys_file_path, 'r') as file:
        print(f'Reading keys from {keys_file_path}...')
        keys = file.read().splitlines()
    
    # Read the json file
    with open(json_file_path, 'r') as file:
        print(f'Reading data from {json_file_path}...')
        data = json.load(file)
    
    # Extract items based on the keys
    extracted_data = {key: data[key] for key in keys if key in data}
    
    # Save the extracted items to a new json file
    with open(output_file_path, 'w') as file:
        print(f'Saving extracted data to {output_file_path}...')
        json.dump(extracted_data, file, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--keys_file_name', '-k', type=str)
    parser.add_argument('--json_file_name', '-j', type=str, default='postgres_all')
    parser.add_argument('--output_file_name', '-o', type=str)
    args = parser.parse_args()

    keys_file_path = os.path.join(KNOB_DIR, 'selection_lists', f'{args.keys_file_name}.txt')
    json_file_path = os.path.join(KNOB_DIR, f'{args.json_file_name}.json')
    output_file_path = os.path.join(KNOB_DIR, f'{args.output_file_name}.json')

    extract_keys_from_json(keys_file_path, json_file_path, output_file_path)