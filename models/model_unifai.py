import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Parameter
import pdb

class Xvector_UnifAI(nn.Module):

    def __init__(self):

        super().__init__()
        self.n_spk = 1000
        self.n_bias = 2 # Number of classes in biasing factor (2 for gender, 2 for age)
        self.x_shape = (512,)
        self.embedding_dim_1 = 128
        self.embedding_dim_2 = 128
        self.noisy_drp_rate = 0.2

    def init_model(self, model_config):
        self.embedding_dim_1 = int(model_config['embedding_dim_1'])
        self.embedding_dim_2 = int(model_config['embedding_dim_2'])
        self.noisy_drp_rate = float(model_config['noisy_drp_rate'])
        self.indirect_bias_FLAG = model_config.getboolean('indirect_bias_FLAG')

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
        
        # Predicting biasing factor
        self.pred_bias_fc1 = nn.Sequential(nn.Linear(self.embedding_dim_1, 64),
                                  nn.BatchNorm1d(64),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2)
        )
            
        self.pred_bias_final = nn.Sequential(nn.Linear(64, self.n_bias)
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
    
    def forward_bias(self, e):
        '''
        :param e1: speaker embedding
        :return: predicted bias
        '''
        b_hat = self.predict_bias(e)

        return b_hat 
        
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
    
    def predict_bias(self, x):
        '''
        :param x: e1 as input
        :return: bias prediction logits
        '''
        x = self.pred_bias_fc1(x)
        x = self.pred_bias_final(x)

        return x
