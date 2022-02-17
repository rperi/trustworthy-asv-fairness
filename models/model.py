import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Parameter
import pdb

'''
This code was adapted from https://github.com/isi-vista/Unified-Adversarial-Invariance
and re-implemented in pytorch to use speaker embeddings as input
'''

class Xvector_UAI(nn.Module):

    def __init__(self):

        super().__init__()
        self.n_spk = 1000
        self.x_shape = (512,)
        self.embedding_dim_1 = 128
        self.embedding_dim_2 = 128
        self.noisy_drp_rate = 0.2

    def init_model(self, model_config):
        self.embedding_dim_1 = int(model_config['embedding_dim_1'])
        self.embedding_dim_2 = int(model_config['embedding_dim_2'])
        self.noisy_drp_rate = float(model_config['noisy_drp_rate'])

        # Encoder
        self.enc_fc1 = nn.Sequential(nn.Linear(self.x_shape[0], 512),
                                     nn.BatchNorm1d(512),
                                     nn.Dropout(p=0.2)
        )
        self.enc_fc2 = nn.Sequential(nn.Linear(512, 512),
                                     nn.BatchNorm1d(512),
                                     nn.Dropout(p=0.2),
                                     nn.ReLU()
        )
        self.emb1_layer = nn.Sequential(nn.Linear(512, self.embedding_dim_1),
                                  nn.Tanh()
        )
        self.emb2_layer = nn.Sequential(nn.Linear(512, self.embedding_dim_2),
                                  nn.Tanh()
        )
        # Predictor (Predicts speaker class from embedding_1)
        self.pred_fc1 = nn.Sequential(nn.BatchNorm1d(self.embedding_dim_1),
                                  nn.Linear(self.embedding_dim_1, 256),
                                  nn.BatchNorm1d(256),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2)
        )
        self.pred_fc2 = nn.Sequential(nn.Linear(256, 512),
                                  nn.BatchNorm1d(512),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2)
        )
        self.pred_final = nn.Sequential(nn.Linear(512, self.n_spk)
        )

        # Noisy transformer
        self.noisy_transformer = nn.Sequential(nn.Dropout(p=self.noisy_drp_rate)
        )

        # Decoder (Reconsutrucs input from embedding_2 and noisy version of embedding_1)
        embedding_dimension = self.embedding_dim_1 + self.embedding_dim_2
        self.dec_fc1 = nn.Sequential(nn.Linear(embedding_dimension, 512),
                                  nn.BatchNorm1d(512),
                                  nn.ReLU()
        )
        self.dec_fc2 = nn.Sequential(nn.Linear(512, 512),
                                     nn.BatchNorm1d(512),
                                     nn.ReLU()
                                     )
        self.dec_fc3 = nn.Sequential(nn.Linear(512, self.x_shape[0])
        )

        # Disentangler_1-2
        # 1
        self.dis_1to2 = nn.Sequential(nn.Linear(self.embedding_dim_1, self.embedding_dim_1),
                                      nn.Dropout(p=0.2),
                                      nn.ReLU(),

                                     nn.Linear(self.embedding_dim_1, self.embedding_dim_1),
                                     nn.Dropout(p=0.2),
                                     nn.ReLU(),

                                     nn.Linear(self.embedding_dim_1, self.embedding_dim_2),
                                     nn.Tanh()
        )

        # Disentangler_2-1
        # 1
        self.dis_2to1 = nn.Sequential(nn.Linear(self.embedding_dim_2, self.embedding_dim_2),
                                     nn.Dropout(p=0.2),
                                     nn.ReLU(),

                                     nn.Linear(self.embedding_dim_2, self.embedding_dim_2),
                                     nn.Dropout(p=0.2),
                                     nn.ReLU(),

                                     nn.Linear(self.embedding_dim_2, self.embedding_dim_1),
                                     nn.Tanh()
        )
        
    def init_model_weights(self):
        # Initialize all weights using xavier_uniform (since, in original keras code, default was glorot_uniform),
        # except the embedding layers, which are initialized using orthogonal initializer
        # Encoder
        torch.nn.init.xavier_uniform_(self.enc_fc1[0].weight)
        torch.nn.init.xavier_uniform_(self.enc_fc2[0].weight)
        torch.nn.init.orthogonal_(self.emb1_layer[0].weight)
        torch.nn.init.orthogonal_(self.emb2_layer[0].weight)
        #torch.nn.init.xavier_uniform_(self.enc_fc3[0].weight)
        #torch.nn.init.xavier_uniform_(self.enc_fc4[0].weight)
        # Predictor
        torch.nn.init.xavier_uniform_(self.pred_fc1[1].weight)
        torch.nn.init.xavier_uniform_(self.pred_fc2[0].weight)
        torch.nn.init.xavier_uniform_(self.pred_final[0].weight)
        # Decoder
        torch.nn.init.xavier_uniform_(self.dec_fc1[0].weight)
        torch.nn.init.xavier_uniform_(self.dec_fc2[0].weight)
        torch.nn.init.xavier_uniform_(self.dec_fc3[0].weight)
        # Disentangler
        torch.nn.init.xavier_uniform_(self.dis_1to2[0].weight)
        torch.nn.init.xavier_uniform_(self.dis_1to2[3].weight)
        torch.nn.init.xavier_uniform_(self.dis_1to2[6].weight)
        torch.nn.init.xavier_uniform_(self.dis_2to1[0].weight)
        torch.nn.init.xavier_uniform_(self.dis_2to1[3].weight)
        torch.nn.init.xavier_uniform_(self.dis_2to1[6].weight)

        
        # Initialize bias with zeros for all layers
        # Encoder
        if self.enc_fc1[0].bias is not None:
            torch.nn.init.zeros_(self.enc_fc1[0].bias)
        if self.enc_fc2[0].bias is not None:
            torch.nn.init.zeros_(self.enc_fc2[0].bias)
        #if self.enc_fc3[0].bias is not None:
        #    torch.nn.init.zeros_(self.enc_fc3[0].bias)
        #if self.enc_fc4[0].bias is not None:
        #    torch.nn.init.zeros_(self.enc_fc4[0].bias)
        if self.emb1_layer[0].bias is not None:
            torch.nn.init.zeros_(self.emb1_layer[0].bias)
        if self.emb2_layer[0].bias is not None:
            torch.nn.init.zeros_(self.emb2_layer[0].bias)
        # Predictor
        if self.pred_fc1[1].bias is not None:
            torch.nn.init.zeros_(self.pred_fc1[1].bias)
        if self.pred_fc2[0].bias is not None:
            torch.nn.init.zeros_(self.pred_fc2[0].bias)
        if self.pred_final[0].bias is not None:
            torch.nn.init.zeros_(self.pred_final[0].bias)
        # Decoder
        if self.dec_fc1[0].bias is not None:
            torch.nn.init.zeros_(self.dec_fc1[0].bias)
        if self.dec_fc2[0].bias is not None:
            torch.nn.init.zeros_(self.dec_fc2[0].bias)
        if self.dec_fc3[0].bias is not None:
            torch.nn.init.zeros_(self.dec_fc3[0].bias)
        # Disentangler
        if self.dis_1to2[0].bias is not None:
            torch.nn.init.zeros_(self.dis_1to2[0].bias)
        if self.dis_1to2[3].bias is not None:
            torch.nn.init.zeros_(self.dis_1to2[3].bias)
        if self.dis_2to1[0].bias is not None:
            torch.nn.init.zeros_(self.dis_2to1[0].bias)
        if self.dis_2to1[3].bias is not None:
            torch.nn.init.zeros_(self.dis_2to1[3].bias)
        if self.dis_2to1[6].bias is not None:
            torch.nn.init.zeros_(self.dis_2to1[6].bias)
    

    def forward_prim(self, x):
        '''
        :param x: speaker embedding as input
        :return: speaker prediction logits and reconstructed input
        '''
        e1, e2 = self.encode(x)
        e1_prime = self.noisy_transformer(e1)

        x_hat = self.decode((e1_prime, e2))
        y_hat = self.predict(e1)

        return y_hat, x_hat, e1, e2

    def forward_dis(self, e1, e2):
        '''
        :param e1: speaker embedding
        param e2: nuisance embedding
        :return: predicted e1_hat and e2_hat
        '''
        e2_hat = self.dis_1to2(e1)
        e1_hat = self.dis_2to1(e2)

        return e1_hat, e2_hat

    def encode(self, x):
        '''
        :param x: speaker embedding as input
        :return: embeddings e1 and e2
        '''
        x = self.enc_fc1(x)
        x = self.enc_fc2(x)
        
        e1 = self.emb1_layer(x)
        e2 = self.emb2_layer(x)

        return e1, e2

    def decode(self, x):
        '''
        :param x: tuple of 2 elements (e1_prime, e2)
        :return: reconstructed input speaker embeddings
        '''
        e1_prime, e2 = x
        x = torch.cat((e1_prime, e2), 1)

        x = self.dec_fc1(x)
        x = self.dec_fc2(x)
        x = self.dec_fc3(x)

        return x

    def predict(self, x):
        '''
        :param x: e1 as input
        :return: speaker prediction logits
        '''
        x = self.pred_fc1(x)
        x = self.pred_fc2(x)
        x = self.pred_final(x)

        return x
