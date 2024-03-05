import os
import json

def load_json(path):
    with open(path) as fp:
        data = json.load(fp)
    return data

def save_json(data, path):
    with open(path, "w") as fp:
        json.dump(data, fp)
    print("File saved to: ", path)

def create_directory(base_directory, additional_directory_name):
    new_directory = os.path.join(base_directory, additional_directory_name)
    os.makedirs(new_directory)
    return new_directory

def create_incremented_directory(base_path):
    counter = 0
    while True:
        dir_name = f"run_{counter}"
        dir_path = os.path.join(base_path, dir_name)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Directory '{dir_name}' created.")
            break
        else:
            counter += 1
    return dir_path
