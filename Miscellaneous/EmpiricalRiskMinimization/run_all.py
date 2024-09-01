import os
import threading
import tensorflow as tf


base_location = os.getcwd()


def run_python_files(files, gpu_device):
    for file in files:
        try:
            os.chdir(base_location+os.path.dirname(file).replace('.', ''))
            os.system(
                'python3 {} --gpu {}'.format(os.path.basename(file), gpu_device))
        finally:
            os.chdir(base_location)


def main_threaded():
    device_count = len(tf.config.experimental.list_physical_devices('GPU'))
    files = []
    not_files = ['params.py', 'train.py', 'dataset.py',
                 'model.py', 'train_and_inference.py', 'run_all.py']
    for file in find_python_files():
        found = False
        for check in not_files:
            if file.find(check) != -1:
                found = True
                break
        if not found:
            files.append(file)

    dividings = [[] for _ in range(device_count)]

    for i, file in enumerate(files):
        dividings[i % device_count].append(file)

    threads = [threading.Thread(target=run_python_files, args=(
        dividings[i], i)) for i in range(device_count)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


def find_python_files(root_dir='./'):
    python_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))

    return python_files


if __name__ == "__main__":
    print(find_python_files())
    main_threaded()
