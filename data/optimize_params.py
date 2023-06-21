# Write a python file to run a python file using subprocess


#### NEED TO SORT THE WRITING OF OUTPUTS TO LOG FILES ####
import numpy as np
import os
import sys
import logging
import subprocess
import torch
import gc

def main(args):

    # Create a logs directory
    log_path = "logs/"
    log_file_path = str()
    os.makedirs(log_path, exist_ok=True)

    arg = args[1]
    device_ids = args[2]

    if arg == 'lr':
        log_file_path = log_path + "lr.txt"

        with (open(log_file_path, 'a')) as f:

            # Fine tune based on learning rate
            for lr in np.arange(1e-3, 6e-3, 1e-3):
                subprocess.run(['python', 'main.py', '--batch_size', '64', '--num_triplets', '1000', '--device', device_ids, '--lr', str(lr) , '--num_epochs', '10', '--log_path', log_file_path, '--arg', arg])
                torch.cuda.empty_cache()
                gc.collect()

    elif arg == 'bs':
        log_file_path = log_path + "bs.txt"

        with (open(log_file_path, 'a')) as f:

            # Fine tune based on batch size
            for bs in [32, 64, 128]:
                print(f"\n\n##### BATCH SIZE : {bs} ######\n")
                subprocess.run(['python', 'main.py', '--num_triplets', '5', '--device', device_ids, '--batch_size', str(bs), '--num_epochs', '2', '--log_path', log_file_path, '--arg', arg])
                torch.cuda.empty_cache()
                gc.collect()

    elif arg == 'num_epochs':
        log_file_path = log_path + "num_epochs.txt"

        with (open(log_file_path, 'a')) as f:

            # Fine tune based on number of epochs
            for ep in [5, 10, 15, 20]:
                print(f"\n\n##### NUMBER OF EPOCHS : {ep} ######\n")
                subprocess.run(['python', 'main.py', '--num_triplets', '5', '--device', device_ids, '--num_epochs', str(ep), '--log_path', log_file_path, '--arg', arg])
                torch.cuda.empty_cache()
                gc.collect()

    elif arg == 'num_triplets':
        log_file_path = log_path + "num_triplets.txt"

        with (open(log_file_path, 'a')) as f:

            # Fine tune based on number of triplets
            for tr in [100, 1000, 10000, 50000]:
                print(f"\n\n##### NUMBER OF TRIPLETS : {tr} ######\n")
                subprocess.run(['python', 'main.py', '--lr', '3e-5', '--num_triplets', str(tr), '--device', device_ids, '--num_epochs', '10', '--log_path', log_file_path, '--arg', arg])
                torch.cuda.empty_cache()
                gc.collect()


# Usage : python optimize_params.py {arg}
if __name__ == '__main__':
    main(sys.argv)