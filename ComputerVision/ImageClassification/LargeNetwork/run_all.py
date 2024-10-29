import os
import threading
import tensorflow as tf
import pyfiglet


base_location = os.getcwd()
def run_shell_files(files):
    for file in files:
        print(file)
        if file.find('run_all') != -1:
            continue
        try:
            os.chdir(base_location+os.path.dirname(file).replace('.',''))
            print()
            pyfiglet.print_figlet('{}'.format(file.split('/')[1]))
            print('Currently running {}'.format(file))
            print()
            for i in ['mnist','cifar10','cifar100','fashion_mnist']:
                os.system('python3 {} --type {} --gpu 0'.format(os.path.basename(file),i))
            # os.system('sleep 10')
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


def find_files(root_dir='./',type='train_and_test.py'):
    python_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(type):
                python_files.append(os.path.join(root, file))

    return python_files


if __name__ == "__main__":
    # print(find_files())
    # main_threaded()
    files = sorted(find_files())
    run_shell_files(files)
    # print(files)
