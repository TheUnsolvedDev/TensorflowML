import os
import threading
import tensorflow as tf
import pyfiglet


base_location = os.getcwd()
def run_shell_files(files): # 'AlexNet','DenseNet121','DenseNet169', 'DenseNet201','DenseNet264','InceptionV1','ResNet18','ResNet34'
    files_to_run = ['VGG11','VGG11_LRN','VGG13','VGG16C','VGG16D','VGG19']
    for file in files:
        if file.find('run_all') != -1:
            continue
        try:
            os.chdir(base_location+os.path.dirname(file).replace('.',''))
            # print()
            file_name = file.split('/')[-2]
            if file_name not in files_to_run:
                continue
            pyfiglet.print_figlet('{}'.format(file_name), font='digital')
            # print('Currently running {}'.format(file))
            # print()
            # for i in ['mnist','cifar10','cifar100','fashion_mnist']:
            #     print('python3 {} --type {} --gpu 0'.format(os.path.basename(file),i))
            os.system('bash run.sh')
            os.system('sleep 10')
        finally:
            os.chdir(base_location)


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
