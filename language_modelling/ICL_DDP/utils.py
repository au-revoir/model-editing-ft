import os
import json

def load_json(path):
    with open(path) as fp:
        data = json.load(fp)
    return data

def save_json(path, data):
    with open(path, "w") as fp:
        json.dump(data, fp)
    print("File saved to: {}".format(path))

def create_numbered_directory(base_dir, model_name, dir_prefix="inference_run_"):
    """
    Creates a new numbered directory inside the given base directory.

    Parameters:
        base_dir (str): The base directory where the new directories will be created.
        dir_prefix (str, optional): Prefix for the directory name. Default is "inference_".

    Returns:
        str: The path of the newly created directory.
    """
    #base_dir = os.path.join(base_dir, "/", model_name, "/")
    new_base_dir = f"{base_dir}/{model_name}/"
    if not os.path.exists(new_base_dir):
        os.makedirs(new_base_dir)
    print("new_base_dir: ", new_base_dir)
    dir_count = 0
    while True:
        new_dir_name = f"{dir_prefix}{dir_count}/"
        new_dir_path = os.path.join(new_base_dir, new_dir_name)
        print(new_dir_path)
        if not os.path.exists(new_dir_path):
            os.makedirs(new_dir_path)
            print("Output save path: ", new_dir_path)
            return new_dir_path
        dir_count += 1
