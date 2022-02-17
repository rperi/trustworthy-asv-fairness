import os, shutil
import torch
import numpy as np
import argparse
from models.model_unifai import Xvector_UnifAI
from models.model import Xvector_UAI
from datasets.fairvoice import FairVoice_embeddings
from datasets.xvectors import Xvectors
import configparser
from scripts.trainer import Trainer
import pdb

def seed_random():
    torch.manual_seed(10101)
    np.random.seed(10101)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_dirs(directories, clean_flag):
    for direc in directories:
        if not os.path.exists(direc):
            os.makedirs(direc)
        elif clean_flag:
            shutil.rmtree(direc)
            os.makedirs(direc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True,
                        help='Should contain train_data.npy, val_data.npy,'
                             'train_labels.npy and val_labels.npy')
    parser.add_argument('--out_data_root', type=str, default=None,
                        help='If not provided, it defaults to args.data_root/out_exp_<exp_id>')
    parser.add_argument('--weights_root', type=str, default='./saved_models/')
    parser.add_argument('--logs_root', type=str, default='./logs/')
    parser.add_argument('--clean_flag', type=bool, default=False)

    args = parser.parse_args()
    
    # Load configuration file
    config = configparser.ConfigParser()
    config.read(args.config_path)
    model_config = config['model_params']
    train_config = config['train_params']
    loss_config = config['loss_params']
    
    os.environ['CUDA_VISIBLE_DEVICES'] = train_config['visible_devices']

    exp_id = args.config_path.split('/')[-1].split('.cfg')[0]
 
    log_dir = "{}/{}".format(args.logs_root, exp_id)
    model_dir = "{}/{}".format(args.weights_root, exp_id)
    if args.out_data_root:
        out_dir = os.path.join(args.out_data_root,'out_{}'.format(exp_id))
    else:
        out_dir = os.path.join(args.data_root,'out_{}'.format(exp_id))

    # Create necessary directories if they do not exist
    create_dirs([log_dir, model_dir, out_dir], clean_flag=args.clean_flag)

    # Seed all randomization
    seed_random()
    # Load data
    if train_config['mode'] != 'NLDR' and train_config['mode'] != 'UAI':
    #train_config.getboolean('bias_FLAG') or model_config.getboolean('indirect_bias_FLAG') or train_config.getboolean('bias_only_FLAG'):
        train_data = FairVoice_embeddings(args.data_root, bias_type=train_config['bias_type'])
        val_data = FairVoice_embeddings(args.data_root, bias_type=train_config['bias_type'], valFlag=True)
    else: # Not using bias labels
        train_data = Xvectors(args.data_root)
        val_data = Xvectors(args.data_root, valFlag=True)
         
    # Initialize model
    if train_config['mode'] != 'NLDR' and train_config['mode'] != 'UAI':
    #train_config.getboolean('bias_FLAG') or model_config.getboolean('indirect_bias_FLAG') or train_config.getboolean('bias_only_FLAG'):
        model_UAI = Xvector_UnifAI()
        model_UAI.n_bias = train_data.num_bias
    else:
        model_UAI = Xvector_UAI()
    model_UAI.n_spk = train_data.num_spk
    model_UAI.x_shape = (train_data.feat_dim,)
    model_UAI.init_model(model_config)
    if train_config['mode'] == 'UAI':
    #not (train_config.getboolean('bias_FLAG') or model_config.getboolean('indirect_bias_FLAG') or train_config.getboolean('bias_only_FLAG')):
        model_UAI.init_model_weights()
     
    trainer = Trainer(model_UAI, train_config=train_config, loss_config=loss_config, model_config=model_config,
                      log_dir=log_dir, model_dir=model_dir, out_dir=out_dir)
    
    # Train model
    if train_config['mode'] != 'NLDR' and train_config['mode'] != 'UAI':
    #train_config.getboolean('bias_FLAG') or train_config.getboolean('bias_only_FLAG'):
        trainer.prob_bias = train_data.prob_bias
    trainer.train_model(train_data, val_data)
    
    trainer.writer.flush()
    trainer.writer.close()


if __name__=='__main__':
    main()
