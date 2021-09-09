import sys
sys.path.append("..")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import pandas as pd
import numpy as np
from collections import defaultdict

from Dataset import MyDataset1
from SBERT import SentenceBERT
from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform


testset_seqs1 = []
testset_seqs2 = []
testset_labels = []
with open('../data/STS/sts-test.csv', 'r', encoding='utf-8') as f:
    for line in f:
        tmp = line.strip().split('\t')
        testset_seqs1.append(tmp[5])
        testset_seqs2.append(tmp[6])
        testset_labels.append(float(tmp[4]))

test_dataset = MyDataset1(testset_seqs1, testset_seqs2, testset_labels)

BATCH_SIZE = 64
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)


class ModelVAE(torch.nn.Module):

    def __init__(self, h_dim, z_dim, activation=F.relu, distribution='normal'):
        """
        ModelVAE initializer
        :param h_dim: dimension of the hidden layers
        :param z_dim: dimension of the latent representation
        :param activation: callable activation function
        :param distribution: string either `normal` or `vmf`, indicates which distribution to use
        """
        super(ModelVAE, self).__init__()

        self.z_dim, self.activation, self.distribution = z_dim, activation, distribution

        # 2 hidden layers encoder
        self.fc_e0 = nn.Linear(768, h_dim * 2)
        # self.fc_e0b = nn.BatchNorm1d(h_dim * 2)
        self.fc_e1 = nn.Linear(h_dim * 2, h_dim)
        # self.fc_e1b = nn.BatchNorm1d(h_dim)

        if self.distribution == 'normal':
            # compute mean and std of the normal distribution
            self.fc_mean = nn.Linear(h_dim, z_dim)
            self.fc_var = nn.Linear(h_dim, z_dim)
        elif self.distribution == 'vmf':
            # compute mean and concentration of the von Mises-Fisher
            self.fc_mean = nn.Linear(h_dim, z_dim)
            self.fc_var = nn.Linear(h_dim, 1)
        else:
            raise NotImplemented

        # 2 hidden layers decoder
        self.fc_d0 = nn.Linear(z_dim, h_dim)
        # self.fc_d0b = nn.BatchNorm1d(h_dim)
        self.fc_d1 = nn.Linear(h_dim, h_dim * 2)
        # self.fc_d1b = nn.BatchNorm1d(h_dim * 2)
        self.fc_logits = nn.Linear(h_dim * 2, 768)

    def encode(self, x):
        # 2 hidden layers encoder
        # x = self.activation(self.fc_e0b(self.fc_e0(x)))
        # x = self.activation(self.fc_e1b(self.fc_e1(x)))
        x = self.activation(self.fc_e0(x))
        x = self.activation(self.fc_e1(x))

        if self.distribution == 'normal':
            # compute mean and std of the normal distribution
            z_mean = self.fc_mean(x)
            z_var = F.softplus(self.fc_var(x))
        elif self.distribution == 'vmf':
            # compute mean and concentration of the von Mises-Fisher
            z_mean = self.fc_mean(x)
            z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
            # the `+ 1` prevent collapsing behaviors
            z_var = F.softplus(self.fc_var(x)) + 1
        else:
            raise NotImplemented

        return z_mean, z_var

    def decode(self, z):
        # x = self.activation(self.fc_d0b(self.fc_d0(z)))
        # x = self.activation(self.fc_d1b(self.fc_d1(x)))
        x = self.activation(self.fc_d0(z))
        x = self.activation(self.fc_d1(x))
        x = self.fc_logits(x)

        return x

    def reparameterize(self, z_mean, z_var):
        if self.distribution == 'normal':
            q_z = torch.distributions.normal.Normal(z_mean, z_var)
            p_z = torch.distributions.normal.Normal(torch.zeros_like(z_mean), torch.ones_like(z_var))
        elif self.distribution == 'vmf':
            q_z = VonMisesFisher(z_mean, z_var)
            p_z = HypersphericalUniform(self.z_dim - 1)
        else:
            raise NotImplemented

        return q_z, p_z

    def forward(self, x):
        z_mean, z_var = self.encode(x)
        q_z, p_z = self.reparameterize(z_mean, z_var)
        z = q_z.rsample()
        x_ = self.decode(z)

        return (z_mean, z_var), (q_z, p_z), z, x_

    def get_vae_vector(self, x):
        # x = self.activation(self.fc_e0b(self.fc_e0(x)))
        # x = self.activation(self.fc_e1b(self.fc_e1(x)))
        x = self.activation(self.fc_e0(x))
        x = self.activation(self.fc_e1(x))
        # x = self.fc_mean(x)
        return x


def Spearman(predicted_score, score):
    X1 = pd.Series(predicted_score)
    X2 = pd.Series(score)
    return X1.corr(X2, method='spearman')


def test(model, premodel):
    predict = []
    predict_only_sbert = []
    true = []
    for x1, x2, label in test_loader:
        x1 = premodel.get_sentence_vector(list(x1))
        x2 = premodel.get_sentence_vector(list(x2))
        predict_only_sbert += torch.sum(x1 * x2, 1).tolist()
        x1 = model.get_vae_vector(x1)
        x2 = model.get_vae_vector(x2)
        predict += torch.sum(x1 * x2, 1).tolist()
        true += list(label)
    score = Spearman(np.array(predict), np.array(true))
    score1 = Spearman(np.array(predict_only_sbert), np.array(true))
    print('Spearman(SVAE):', score)
    print('Spearman(SBERT):', score1)


def test_only_sbert(model):
    predict = []
    true = []
    for x1, x2, label in test_loader:
        x1 = model.get_sentence_vector(list(x1))
        x2 = model.get_sentence_vector(list(x2))
        predict += torch.sum(x1 * x2, 1).tolist()
        true += list(label)
    score = Spearman(np.array(predict), np.array(true))
    print('Spearman:', score)



config = 'bert-base-uncased'
H_DIM = 128
Z_DIM = 5
max_length = 30
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sbert = SentenceBERT(config, device, max_length)
param = torch.load('../pretrain_model/SBERT-STSb/pytorch_model.bin', map_location=device)
sbert_dict = sbert.state_dict()
sbert_dict.update({k: v for k, v in param.items() if 'bert' in k})
sbert.load_state_dict(sbert_dict)
sbert.eval()

# modelN = ModelVAE(h_dim=H_DIM, z_dim=Z_DIM, distribution='normal')
modelS = ModelVAE(h_dim=H_DIM, z_dim=Z_DIM + 1, distribution='vmf')
param = torch.load('../parameter/VAE_SNLI_STSb_BN_MSE_model', map_location=device)
modelS.load_state_dict(param)
test(modelS, sbert)
