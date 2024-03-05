import argparse
import collections
import os
import json
import numpy as np

def load_result_json_files(directory_path):
    all_directories = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
    result_directories = [d for d in all_directories if d != 'log']

    result_list = []
    for result_directory in result_directories:
        result_directory_path = os.path.join(directory_path, result_directory)
        json_files = [f for f in os.listdir(result_directory_path) if f.endswith('.json')]

        for json_file in json_files:
            json_file_path = os.path.join(result_directory_path, json_file)

            if os.path.exists(json_file_path):
                with open(json_file_path, 'r') as json_file:
                    result_data = json.load(json_file)
                    result_list.extend(result_data)
            else:
                print(f"Warning: JSON file not found for {json_file_path}")
    return result_list

def main(args):
    result_list = load_result_json_files(args.saved_result_path)
    cur_sum = collections.defaultdict(lambda: []) 

    for data in result_list:
        for key in ["ngram_entropy", "reference_score"]:
            if key in data:
               cur_sum[f"{key}"].append(data[key])
    cur_sum = {k: (np.mean(v), np.std(v)) for k, v in cur_sum.items()}
    for k, v in cur_sum.items():
        if all(exclude not in k for exclude in ["essence_score", "time"]):
            cur_sum[k] = tuple(np.around(z * 100, 2) for z in v)
    print("cur_sum: ", cur_sum)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_result_path", type=str, required=True)
    args = parser.parse_args()
    main(args)

