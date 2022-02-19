import pandas as pd
import numpy as np
import os
import sys
import pdb
import argparse

'''
Script to find the epoch that gives maximum validation set accuracy.
Used to choose best checkpoint
'''

def main(args):
    exp = args.exp_id

    log_dir = os.path.join(args.log_root,"exp_{}".format(exp))
    df = pd.read_csv(os.path.join(log_dir,'output.log'), delimiter=' ')
    epochs = df['Epoch'].values
    val_acc = df['Validation_accuracy'].values

    best_idx = np.argmax(val_acc)
    best_ep = epochs[best_idx]

    if 'Validation_accuracy_bias' in df.columns:
        val_acc_bias = df['Validation_accuracy_bias'].values[best_idx]
    else:
        val_acc_bias = 0
    print("best epoch for exp = {} is {}".format(exp,best_ep))
    print("Val spk acc = {}. Val bias acc = {}".format(df['Validation_accuracy'].values[best_idx], val_acc_bias))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str, required=True)
    parser.add_argument('--log_root', type=str, required=True,
                        help='Directory where logs are saved')

    args = parser.parse_args()
    main(args)
