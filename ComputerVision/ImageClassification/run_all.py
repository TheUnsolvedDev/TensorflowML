import os
import threading
import tensorflow as tf


base_location = os.getcwd()
def run_python_files(files, gpu_device):
    for file in files:
        if file.find('run_all') != -1:
            continue
        try:
            os.chdir(base_location+os.path.dirname(file).replace('.',''))
            print()
            print('Currently running {}'.format(os.path.basename(file)))
            print()
            os.system('python3 {} --gpu {}'.format(os.path.basename(file), gpu_device))
        finally:
            os.chdir(base_location)


def main_threaded():
    device_count = len(tf.config.experimental.list_physical_devices('GPU'))
    not_files = 'run_all.py'
    files = sorted([file for file in find_python_files()
                   if file.find(not_files) == -1])
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
    # main_threaded()
    files = find_python_files()
    run_python_files(files, '0')
