import os


def get_dir_files(dir_path):
    files = [file for file in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, file))]
    return sorted(files)


def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def remove_dir(dir_path):
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        if os.path.isfile(file_path):
            os.unlink(file_path)
    os.rmdir(dir_path)
