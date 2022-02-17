import numpy as np
import os
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from utils.basic_utils import *
import torch
from collections import Counter, defaultdict
from sklearn.metrics import pairwise_distances
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import pdb
import time

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class Trainer():
    def __init__(self, model, train_config, loss_config, model_config, log_dir, model_dir, out_dir, model_spk_embed=None):
        # model refers to the uai model after speaker embeddings

        self.mode = 'train'  # 'val'

        self.ep_idx = 0
        self.pre_trained = False
        self.idx = 0
        self.train_size = 0
        self.val_size = 0

        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_file = os.path.join(log_dir, 'output.log')
        self.model_dir = model_dir
        self.out_dir = out_dir
        self.device = get_device()

        self.model = model.to(self.device)
 
        self.train_mode = train_config['mode']
        if train_config['batch_size'] == 'None':
            self.batch_size = None
        else:
            self.batch_size = int(train_config['batch_size'])
        self.num_epochs = int(train_config['num_epochs'])
        self.num_sec_updates = int(train_config['num_sec_updates'])
        self.weight_pred = float(loss_config['weight_pred'])
        self.weight_recon = float(loss_config['weight_recon'])
        if self.train_mode != 'UAI' and self.train_mode != 'NLDR':
            self.weight_bias = float(loss_config['weight_bias'])
        if self.train_mode != 'NLDR':
            self.weight_secondary = float(loss_config['weight_secondary'])

        self.save_val_embeddings_FLAG = train_config.getboolean('save_val_embeddings_FLAG')
        
        self.sec_update = False
        self.train_gen_params = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': train_config.getint('num_workers_train'),
            'worker_init_fn': worker_init_fn,
            'pin_memory': True
        }
        self.val_gen_params = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': train_config.getint('num_workers_val'),
            'worker_init_fn': worker_init_fn,
            'pin_memory': False
        }
        if self.batch_size is not None:
            self.train_gen_params['drop_last'] = True
            self.val_gen_params['drop_last'] = True
        
        #params = self.model.parameters()  # All model parameters
        # Primary branch parameters
        params_prim = \
                list(self.model.enc_fc1.parameters()) +\
                list(self.model.enc_fc2.parameters()) +\
                list(self.model.pred_fc1.parameters()) +\
                list(self.model.pred_fc2.parameters()) +\
                list(self.model.emb1_layer.parameters())
        if self.train_mode != 'NLDR' and self.train_mode != 'AT' and self.train_mode != 'MTL':
            params_prim += \
                list(self.model.emb2_layer.parameters()) +\
                list(self.model.dec_fc1.parameters()) +\
                list(self.model.dec_fc2.parameters()) +\
                list(self.model.dec_fc3.parameters())
        self.optimizer = optim.Adam(params_prim, lr=train_config.getfloat('prim_lr'), weight_decay=train_config.getfloat('prim_decay'))

        params_pred_final = list(self.model.pred_final.parameters())
        self.optimizer_pred_final = optim.Adam(params_pred_final, lr=1e-3, weight_decay=train_config.getfloat('prim_decay'))

        # Also create an optimizer for disentangler and bias predictor only updates
        params_sec = []
        if self.train_mode != 'NLDR' and self.train_mode != 'AT' and self.train_mode != 'MTL':
            params_sec += list(self.model.dis_1to2.parameters()) + list(self.model.dis_2to1.parameters())
        if self.train_mode != 'NLDR' and self.train_mode != 'UAI':
            params_sec += list(self.model.pred_bias_fc1.parameters()) + list(self.model.pred_bias_final.parameters())
        if self.train_mode != 'NLDR': 
            self.optimizer_secondary = optim.Adam(params_sec, lr=train_config.getfloat('sec_lr'), weight_decay=train_config.getfloat('sec_decay'))
                
        self.lr_decay_FLAG = train_config.getboolean('lr_decay_FLAG')
        if self.lr_decay_FLAG:
            self.scheduler = StepLR(self.optimizer, step_size=train_config.getfloat('lr_scheduler_step'), gamma=train_config.getfloat('lr_scheduler_gamma')) # halve the learning rate every 10 epochs
            self.scheduler_secondary = StepLR(self.optimizer_secondary, step_size=train_config.getfloat('lr_scheduler_step'), gamma=train_config.getfloat('lr_scheduler_gamma')) # halve the learning rate every 10 epochs

        # Loss functions to be optimized
        self.criterion_pred = nn.CrossEntropyLoss()
        self.criterion_recon = nn.MSELoss()
        self.criterion_dis = nn.MSELoss()
        self.criterion_bias = nn.CrossEntropyLoss()

    def train_model(self, train_data, val_data):
        train_loader = DataLoader(train_data, **self.train_gen_params)
        val_loader = DataLoader(val_data, **self.val_gen_params)

        self.train_size = len(train_loader)
        self.val_size = len(val_loader)
        val_acc_best = 0.0
        if self.ep_idx >= self.num_epochs-1:
            print("Already trained for {} epochs. Exiting".format(self.num_epochs))
            sys.exit(1)
        best_ep = self.ep_idx
        if self.pre_trained:
            self.ep_idx += 1
        for e in range(self.ep_idx, self.num_epochs):
            # Training
            print("Training. Epoch {}".format(e))
            self.model = self.model.train()

            self.mode = 'train'
            self.ep_idx = e
            true_spk_labels = []
            true_bias_labels = []
            pred_spk_labels = []
            pred_bias_labels = []
            start = time.time()
            for batch_idx, batch in enumerate(tqdm(train_loader)):
                self.idx = batch_idx
                if self.train_mode != 'NLDR' and self.train_mode != 'UAI':
                    batch_data, spk_labels, bias_labels = batch
                    bias_labels = bias_labels.to(self.device)
                else:
                    batch_data, spk_labels = batch
                batch_data = batch_data.type(torch.FloatTensor).to(self.device)
                spk_labels = spk_labels.to(self.device)

                # Forward pass and loss computation of primary model
                loss_prim, y_hat, x_hat, e1, e2, x_inp = \
                    self.forward_pass_prim(batch_data, spk_labels)

                # Model updates
                self.optimizer.zero_grad()
                self.optimizer_pred_final.zero_grad()
                if self.train_mode != 'NLDR':
                    self.optimizer_secondary.zero_grad()
                if self.train_mode == 'NLDR': # Only in the case of NLDR, the secondary model is not required
                    loss_prim.backward()
                    self.optimizer.step()
                    self.optimizer_pred_final.step()
                else:
                    if batch_idx % (self.num_sec_updates+1) == 0:
                        # Update the primary model. Note that the loss computation involves primary and secondary model
                        # However, only the weights corresponding to modules in primary model are updated
                        self.sec_update = False
                        # Forward pass and loss computation of secondary model
                        if self.train_mode != 'UAI':
                            loss_secondary, b_hat = self.forward_pass_secondary(e1, e2, bias_labels, self.sec_update)
                        else:
                            loss_secondary = self.forward_pass_secondary(e1, e2, self.sec_update)
                        
                        loss_overall = loss_prim + \
                                       self.weight_secondary * loss_secondary
                        loss_overall.backward()
                        self.optimizer.step()   # Update only the encoder, predictor and decoder
                        self.optimizer_pred_final.step()
                    else:
                        # Note that the loss computation involves only secondary branch
                        # And only the weights corresponding to modules in secondary branch are updated
                        self.sec_update = True
                        
                        # Forward pass and loss computation of secondary model
                        if self.train_mode != 'UAI':
                            loss_secondary, b_hat = self.forward_pass_secondary(e1, e2, bias_labels, self.sec_update)
                        else:
                            loss_secondary = self.forward_pass_secondary(e1, e2, self.sec_update)
                        
                        # Update only secondary branch
                        loss_secondary.backward()
                        self.optimizer_secondary.step()  # Update only the disentanglers and bias discriminator
                true_spk_labels.append(spk_labels.data.cpu().numpy().tolist())
                pred_spk_labels.append(torch.argmax(y_hat, axis=1).data.cpu().numpy().tolist())
                if self.train_mode != 'NLDR' and self.train_mode != 'UAI':
                    true_bias_labels.append(bias_labels.data.cpu().numpy().tolist())
                    pred_bias_labels.append(torch.argmax(b_hat, axis=1).data.cpu().numpy().tolist())
            true_spk_labels = [x for y in true_spk_labels for x in y]
            pred_spk_labels = [x for y in pred_spk_labels for x in y]
            if self.train_mode != 'NLDR' and self.train_mode != 'UAI':
                pred_bias_labels = [x for y in pred_bias_labels for x in y]
                true_bias_labels = [x for y in true_bias_labels for x in y]
                train_acc_bias = compute_spk_accuracy(true_bias_labels, pred_bias_labels)
            else:
                train_acc_bias = None
            train_acc = compute_spk_accuracy(true_spk_labels, pred_spk_labels)
            self.write_acc(train_acc, train_acc_bias) # For tensorboard
            
            # Learning rate scheduler step
            if self.lr_decay_FLAG:
                if self.ep_idx < 50:
                    self.scheduler.step()
                    self.scheduler_secondary.step()

            # Validation
            print("Validating")
            self.model = self.model.eval()
            self.mode = 'val'
            true_spk_labels = []
            true_bias_labels = []
            pred_spk_labels = []
            pred_bias_labels = []
            if self.save_val_embeddings_FLAG:
                val_inp = []
                val_e1 = []
                val_e2 = []
                val_x = []
            for batch_idx, batch in enumerate(tqdm(val_loader)):
                self.idx = batch_idx
                #if batch_idx % 100 == 0:
                #    print("{}/{}".format(batch_idx, len(val_loader)))
                if self.train_mode != 'NLDR' and self.train_mode != 'UAI':
                    batch_data, spk_labels, bias_labels = batch
                    bias_labels = bias_labels.to(self.device)
                else:
                    batch_data, spk_labels = batch
                batch_data = batch_data.type(torch.FloatTensor).to(self.device)
                spk_labels = spk_labels.to(self.device)
                # Forward pass and loss computation of primary model
                loss_prim, y_hat, x_hat, e1, e2, x_inp = \
                    self.forward_pass_prim(batch_data, spk_labels)

                if self.train_mode != 'NLDR':  # For NLDR, secondary loss doesn't exist
                    # Forward pass and loss computation of secondary model
                    if self.train_mode != 'UAI':  # i.e. if it is UAI-AT, UAI-MTL, AT or MTL
                        loss_secondary, b_hat = self.forward_pass_secondary(e1, e2, bias_labels)
                    else:
                        loss_secondary = self.forward_pass_secondary(e1, e2)

                true_spk_labels.append(spk_labels.data.cpu().numpy().tolist())
                pred_spk_labels.append(torch.argmax(y_hat, axis=1).data.cpu().numpy().tolist())
                if self.train_mode != 'NLDR' and self.train_mode != 'UAI':
                    true_bias_labels.append(bias_labels.data.cpu().numpy().tolist())
                    pred_bias_labels.append(torch.argmax(b_hat, axis=1).data.cpu().numpy().tolist())
                if self.save_val_embeddings_FLAG:
                    val_inp.append(x_inp.data.cpu().numpy().tolist())
                    val_e1.append(e1.data.cpu().numpy().tolist())
                    val_e2.append(e2.data.cpu().numpy().tolist())

            val_inp = np.array([x for y in val_inp for x in y], dtype=float)
            val_e1 = np.array([x for y in val_e1 for x in y], dtype=float)
            val_e2 = np.array([x for y in val_e2 for x in y], dtype=float)
            
            true_spk_labels = np.array([x for y in true_spk_labels for x in y])
            pred_spk_labels = np.array([x for y in pred_spk_labels for x in y])
            if self.train_mode != 'NLDR' and self.train_mode != 'UAI':
                pred_bias_labels = np.array([x for y in pred_bias_labels for x in y])
                true_bias_labels = np.array([x for y in true_bias_labels for x in y])
            # Compute accuracy of minority speakers (Speakers with only one sample)
            counts = Counter(true_spk_labels)
            keys = list(counts.keys())
            keys_minority = [k for k in keys if counts[k]==np.min(list(counts.values()))]
            keys_majority = [k for k in keys if counts[k]>=10]

            idx_minority = [idx for idx in range(true_spk_labels.shape[0]) if true_spk_labels[idx] in keys_minority]
            idx_majority = [idx for idx in range(true_spk_labels.shape[0]) if true_spk_labels[idx] in keys_majority]

            val_acc = compute_spk_accuracy(true_spk_labels, pred_spk_labels)
            if self.train_mode != 'NLDR' and self.train_mode != 'UAI':
                val_acc_bias = compute_spk_accuracy(true_bias_labels, pred_bias_labels)
            else:
                val_acc_bias = None
            val_acc_minority = compute_spk_accuracy(true_spk_labels[idx_minority], pred_spk_labels[idx_minority])
            val_acc_majority = compute_spk_accuracy(true_spk_labels[idx_majority], pred_spk_labels[idx_majority])
             
            self.write_acc(val_acc, val_acc_bias, val_acc_minority, val_acc_majority)  # For tensorboard

            # Evaluating the clustering of e1 embeddings w.r.t speaker labels
            speaker_indices = defaultdict(list)
            mean_e1 = defaultdict(list)
            for spk in np.unique(true_spk_labels):
                speaker_indices[spk] = np.where(true_spk_labels==spk)[0]
                mean_e1[spk] = np.mean(val_e1[speaker_indices[spk],:], axis=0)
            
            # Compute inter-speaker distance, i.e., distance between speaker centroid for every pair of speaker 
            ## The minimum distance for each speaker centroid is computed and mean of this minimum distance is taken over all speakers
            ## This essentially captures the distance to the most closest speaker centroid
            distances = pairwise_distances(np.array(list(mean_e1.values())))
            distances[distances==0]=1e10
            inter_spk_dist = np.mean(np.min(distances, axis=0))
            
            # Compute median intra-speaker distance, i.e., the average distance between the centroid and the embeddings for each speaker
            # And then report the mean over all speakers.
            # Median is used to be more robust to outliers (due to noise)
            intra_spk_dist = []
            for spk in mean_e1.keys():
                intra_spk_dist.append(np.median(pairwise_distances(mean_e1[spk].reshape(1,-1),val_e1[speaker_indices[spk],:])))
            intra_spk_dist = np.mean(intra_spk_dist)
            
            self.write_distances(intra_spk_dist, inter_spk_dist)  # For tensorboard
            if self.save_val_embeddings_FLAG:
                if val_acc > val_acc_best:
                    np.save(os.path.join(self.out_dir, 'val_inp'), val_inp)
                    np.save(os.path.join(self.out_dir, 'val_e1'), val_e1)
                    np.save(os.path.join(self.out_dir, 'val_e2'), val_e2)
                    np.save(os.path.join(self.out_dir, 'val_spk_labels'), true_spk_labels)
                    if self.train_mode != 'NLDR' and self.train_mode != 'UAI': 
                        np.save(os.path.join(self.out_dir, 'val_bias_labels'), true_bias_labels)
                    with open(os.path.join(self.out_dir, 'best_epoch.txt'), 'w') as o:
                        o.write("Best Epoch = {}".format(e))

            best_epoch_FLAG = False
            if val_acc > val_acc_best:
                val_acc_best = val_acc
                if os.path.exists(os.path.join(self.model_dir, 'Epoch_{}.pt'.format(best_ep))) and self.ep_idx%50 != 0:
                    os.remove(os.path.join(self.model_dir, 'Epoch_{}.pt'.format(best_ep)))
                best_ep = e
                best_epoch_FLAG = True
            # Save models
            if self.ep_idx%10 == 0 or best_epoch_FLAG:
                model_ckpt = os.path.join(self.model_dir, 'Epoch_{}.pt'.format(e))
                self.save_model(model_ckpt)

            # Printing metrics
            if self.train_mode != 'NLDR' and self.train_mode != 'UAI':
                print("Epoch:{}, Train accuracy:{}, Val accuracy:{} Train bias accuracy:{} Val bias accuracy:{}".format(e, train_acc, val_acc, train_acc_bias, val_acc_bias))
            else:
                print("Epoch:{}, Train accuracy:{}, Val accuracy:{}".format(e, train_acc, val_acc))

            # Logging metrics
            if self.train_mode != 'NLDR' and self.train_mode != 'UAI':
                self.write_log(train_acc, val_acc, val_acc_minority,val_acc_majority, intra_spk_dist, inter_spk_dist, val_acc_bias)
            else:
                self.write_log(train_acc, val_acc, val_acc_minority,val_acc_majority, intra_spk_dist, inter_spk_dist)

    def forward_pass_prim(self, batch_data, spk_labels, b=None):
        '''
        Performs forward pass of primary model on given batch
        :return: loss
        '''
        spk_embed = batch_data

        y_hat, x_hat, e1, e2 = self.model.forward_prim(spk_embed)
        loss_pred = self.criterion_pred(y_hat, spk_labels)
        if self.train_mode=='NLDR' or self.train_mode=='AT' or self.train_mode=='MTL': # STL, AT, MTL
        # Disable the decoder. Reconstrucion loss should not be considered. Only encoder and predictor should be active
            loss_recon = torch.tensor(0)
            loss_prim_overall = self.weight_pred*loss_pred
        else:
            loss_recon = self.criterion_recon(x_hat, spk_embed)
            loss_prim_overall = self.weight_pred * loss_pred + self.weight_recon * loss_recon
        

        self.write_loss_prim(loss_pred.data.cpu().numpy(),
                             loss_recon.data.cpu().numpy(),
                             loss_prim_overall.data.cpu().numpy())
        return loss_prim_overall, y_hat, x_hat, e1, e2, spk_embed

    def forward_pass_secondary(self, e1, e2, b=None, sec_update=True):
        '''
        Does forward pass of disentanglers and
        computes reconstruction loss of e1->e2 and e2->e1, 
        if applicable computes prediction loss of bias discriminator
        :return: secondary branch loss
        '''
        if self.train_mode != 'AT' and self.train_mode != 'MTL':
            e1_hat, e2_hat = self.model.forward_dis(e1, e2)
        if self.train_mode != 'UAI':
            b_hat = self.model.forward_bias(e1)
        if sec_update: # Secdondary branch update
            if self.train_mode != 'AT' and self.train_mode != 'MTL': # i.e. if it is either UAI,UAI-AT or UAI-MTL
                loss_dis_1to2 = self.criterion_dis(e2_hat, e2)
                loss_dis_2to1 = self.criterion_dis(e1_hat, e1)
            if self.train_mode != 'UAI':  # UAI, UAI-AT, UAI-MTL
                loss_bias = self.criterion_bias(b_hat, b)
        else: # Primary model update using disentangler and discriminator loss
            # Make predictions of disentangler close to uniform distribution
            if self.train_mode != 'AT' and self.train_mode != 'MTL':
                e1_rand = (-2 * torch.rand(e1.shape[0], e1.shape[1]) + 1).to(self.device)
                e2_rand = (-2 * torch.rand(e2.shape[0], e2.shape[1]) + 1).to(self.device)
                loss_dis_1to2 = self.criterion_dis(e2_hat, e2_rand)
                loss_dis_2to1 = self.criterion_dis(e1_hat, e1_rand)
            if self.train_mode == 'UAI-AT' or self.train_mode == 'AT':
                # Make predictions of bias discriminator close to uniform
                b_rand = torch.Tensor(e1.shape[0])
                for idx in range(e1.shape[0]):
                    b_rand[idx] = np.random.choice([key for key in self.prob_bias.keys()],p=[self.prob_bias[key] for key in self.prob_bias.keys()])
                b_rand = b_rand.type(torch.LongTensor).to(self.device)
                loss_bias = self.criterion_bias(b_hat, b_rand)
            elif self.train_mode == 'UAI-MTL' or self.train_mode == 'MTL':
                loss_bias = self.criterion_bias(b_hat, b)
        if self.train_mode == 'UAI':
            loss_secondary = loss_dis_1to2 + loss_dis_2to1
        elif self.train_mode == 'AT' or self.train_mode == 'MTL':
            loss_secondary = self.weight_bias*loss_bias
        else: #UAI-AT,UAI-MTL
            loss_secondary = loss_dis_1to2 + loss_dis_2to1 + self.weight_bias*loss_bias

        # Write losses to tensorboard
        if self.train_mode == 'UAI':
            self.write_loss_sec(self.criterion_dis(e2_hat, e2).data.cpu().numpy(),
                                self.criterion_dis(e1_hat, e1).data.cpu().numpy(),
                                (self.criterion_dis(e2_hat, e2)+ self.criterion_dis(e1_hat, e1)).data.cpu().numpy())
        elif self.train_mode == 'AT' or self.train_mode == 'MTL':
            self.write_loss_sec(0, 0, self.criterion_bias(b_hat, b).data.cpu().numpy(), self.criterion_bias(b_hat, b).data.cpu().numpy())
        else: #UAI-AT,UAI-MTL
            self.write_loss_sec(self.criterion_dis(e2_hat, e2).data.cpu().numpy(),
                                self.criterion_dis(e1_hat, e1).data.cpu().numpy(),
                                (self.criterion_dis(e2_hat, e2)+ self.criterion_dis(e1_hat, e1)+self.criterion_bias(b_hat, b)).data.cpu().numpy(),
                                self.criterion_bias(b_hat, b).data.cpu().numpy())
        if self.train_mode != 'UAI':
            return loss_secondary, b_hat
        else:
            return loss_secondary

    def save_model(self, ckpt):
        state = {'epoch': self.ep_idx,
                 'state_dict': self.model.state_dict(),
                 'optimizer_prim': self.optimizer.state_dict(),
                 }
        if self.train_mode != 'NLDR':
            state['optimizer_sec'] = self.optimizer_secondary.state_dict()
            
        torch.save(state, ckpt)

    def write_loss_prim(self, loss_pred, loss_recon, loss_overall):
        if self.mode == 'train':
            idx = self.idx + self.train_size * self.ep_idx
        else:
            idx = self.idx + self.val_size * self.ep_idx

        self.writer.add_scalar('Loss/Prediction/{}/'.format(self.mode),
                               loss_pred, idx)
        self.writer.add_scalar('Loss/Reconstruction/{}/'.format(self.mode),
                               loss_recon, idx)
        self.writer.add_scalar('Loss/Overall_prim/{}/'.format(self.mode),
                               loss_overall, idx)

    def write_loss_sec(self, loss_dis_1to2, loss_dis_2to1, loss_secondary, loss_bias=None):
        if self.mode == 'train':
            idx = self.idx + self.train_size * self.ep_idx
        else:
            idx = self.idx + self.val_size * self.ep_idx

        self.writer.add_scalar('Loss/Dis_1to2/{}/'.format(self.mode),
                               loss_dis_1to2, idx)
        self.writer.add_scalar('Loss/Dis_2to1/{}/'.format(self.mode),
                               loss_dis_2to1, idx)
        self.writer.add_scalar('Loss/Overall_sec/{}/'.format(self.mode),
                               loss_secondary, idx)
        if self.train_mode != 'NLDR' and self.train_mode != 'UAI':
            self.writer.add_scalar('Loss/Bias/{}/'.format(self.mode),
                                   loss_bias, idx)
        

    def write_acc(self, accuracy, accuracy_bias=None, accuracy_minority=None, accuracy_majority=None):
        self.writer.add_scalar('Accuracy/{}/'.format(self.mode),
                               accuracy, self.ep_idx)
        if accuracy_bias is not None:
            self.writer.add_scalar('Accuracy/{}_bias'.format(self.mode),
                                   accuracy_bias, self.ep_idx)
            
        if accuracy_minority:
            self.writer.add_scalar('Accuracy/{}_minority/'.format(self.mode),
                                   accuracy_minority, self.ep_idx)
        if accuracy_majority:
            self.writer.add_scalar('Accuracy/{}_majority/'.format(self.mode),
                                   accuracy_majority, self.ep_idx)

    def write_distances(self, intra_spk, inter_spk):
        self.writer.add_scalar('Distances/{}_intra_speaker/'.format(self.mode),
                               intra_spk, self.ep_idx) 
        self.writer.add_scalar('Distances/{}_inter_speaker/'.format(self.mode),
                               inter_spk, self.ep_idx)

    def write_log(self, train_acc, val_acc, val_acc_minority, val_acc_majority, intra_spk, inter_spk, val_acc_bias=None):
        with open(self.log_file, 'a') as o:
            if self.ep_idx == 0:
                if not val_acc_bias:
                    o.write("Epoch Train_accuracy Validation_accuracy Minority_accuracy Majority_accuracy Intra_speaker_distance Inter_speaker_distance\n")
                else:
                    o.write("Epoch Train_accuracy Validation_accuracy Minority_accuracy Majority_accuracy Intra_speaker_distance Inter_speaker_distance Validation_accuracy_bias\n")

            if not val_acc_bias:
                o.write("{} {} {} {} {} {} {}\n".format(self.ep_idx, train_acc, val_acc, val_acc_minority, val_acc_majority, intra_spk, inter_spk))
            else:
                o.write("{} {} {} {} {} {} {} {}\n".format(self.ep_idx, train_acc, val_acc, val_acc_minority, val_acc_majority, intra_spk, inter_spk, val_acc_bias))

