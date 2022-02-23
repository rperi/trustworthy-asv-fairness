import os
import torch
import numpy as np
import argparse
from models.model import Xvector_UAI
from models.model_unifai import Xvector_UnifAI
from datasets.xvectors import Xvectors_test
from datasets.fairvoice import FairVoice_embeddings_test
import configparser
from torch.utils.data import DataLoader
import pdb
import sys
from utils.basic_utils import *

def create_dirs(directories):
    for direc in directories:
        if not os.path.exists(direc):
            os.makedirs(direc)

def save(out_dir, var):
    np.save(os.path.join(out_dir,'emb1'), var[0])
    np.save(os.path.join(out_dir,'emb2'), var[1])
    np.save(os.path.join(out_dir,'spk_pred'), var[2])

def save_bias(out_dir, var):
    np.save(os.path.join(out_dir,'bias_pred'), var)

test_gen_params = {
    'batch_size': 32,
    'shuffle': False,
    'num_workers': 1
}

def main(args):
    # Check existence of required files
    if not os.path.exists(args.inp_feats):
        print("Input file provided doesn't exist in {}. Exiting".format(args.inp_feats))
        sys.exit(1)
    if not os.path.exists(args.checkpoint_file):
        print("Model checkpoint file provided doesn't exist in {}. Exiting".format(args.checkpoint_file)) 
        sys.exit(1)
    device = get_device()  # Function defined in utils/basic_utils.py

    
    # Load configuration file
    config = configparser.ConfigParser()
    config.read(args.config_path)
    model_config = config['model_params']
    test_config = config['test_params']
    test_gen_params['batch_size'] = int(test_config['batch_size'])

    os.environ['CUDA_VISIBLE_DEVICES'] = test_config['visible_devices'] 

    # Load data
    #feats = torch.from_numpy(np.load(args.inp_feats)).type(torch.FloatTensor).to(device)
    if args.mode == 'UAI':
        test_data = Xvectors_test(args.inp_feats)
    else:
        test_data = FairVoice_embeddings_test(args.inp_feats)
    test_data.data, num_pad = pad_data(test_data.data, int(test_config['batch_size']))

    # Initialize and load model
    if args.mode == 'UAI':
        model = Xvector_UAI()
    else:
        model = Xvector_UnifAI()
    model.x_shape = (test_data.feat_dim,)
    checkpoint = torch.load(args.checkpoint_file, map_location=device)
    model.n_spk = checkpoint['state_dict']['pred_final.0.bias'].shape[0]
    if args.mode != 'UAI':
        model.n_bias = checkpoint['state_dict']['pred_bias_final.0.bias'].shape[0]
    model.init_model(model_config)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    # Evaluate model on test input
    y_hat, e1, e2 = forward_pass(model, test_data)
    if args.mode != 'UAI':
        b_hat = forward_pass_bias(model, e1)

    # Save predictions and embeddings
    out_dir = args.out_dir
    if num_pad!=0:
        save(out_dir, (e1[0:-num_pad,:], e2[0:-num_pad,:], y_hat[0:-num_pad]))
        if args.mode != 'UAI':
            save_bias(out_dir, b_hat[0:-num_pad])
    else:
        save(out_dir, (e1, e2, y_hat))
        if args.mode != 'UAI':
            save_bias(out_dir, b_hat)

    print("Done predicting. Saved predictions in {}".format(out_dir))


def pad_data(feats, batch_size):
    num_samples = feats.shape[0]
    if num_samples%batch_size == 0:
        return feats, 0
    else:
        num_extra = (num_samples//batch_size + 1)*batch_size - num_samples
        feats_new = torch.empty((num_samples//batch_size + 1)*batch_size,feats.shape[1])
        feats_new[0:num_samples,:] = feats[:,:]
        feats_new[num_samples:,:] = feats[-num_extra:,:]
        return feats_new, num_extra

def forward_pass(model, test_data):
    device = get_device()
    test_loader = DataLoader(test_data, **test_gen_params)
    e1_all = []
    e2_all = []
    spk_pred = []
    for batch_idx, batch_data in enumerate(test_loader):
        batch_data = batch_data.to(device)
        y_hat, x_hat, e1, e2 = model.forward_prim(batch_data)
        e1_all.append(e1.detach().cpu().numpy())
        e2_all.append(e2.detach().cpu().numpy())
        spk_pred.append(torch.argmax(y_hat, axis=1).detach().cpu().numpy())
    return np.concatenate(spk_pred, axis=0), np.concatenate(e1_all,axis=0), np.concatenate(e2_all, axis=0)

def forward_pass_bias(model, e1):
    device = get_device()
    bias_pred = []
    for sample_idx, sample_data in enumerate(torch.from_numpy(e1)):
        sample_data = sample_data.to(device).unsqueeze(0)
        b_hat = model.forward_bias(sample_data)
        bias_pred.append(torch.argmax(b_hat, axis=1).detach().cpu().numpy())
    return np.concatenate(bias_pred, axis=0)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--inp_feats', type=str, required=True, 
                        help='Input features (x-vectors) as numpy array to do prediction on')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Directory where outut predictions and embeddings will be saved')
    parser.add_argument('--checkpoint_file', type=str, required=True)

    args = parser.parse_args()
    main(args)
