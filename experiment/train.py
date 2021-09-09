import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from collections import defaultdict
import gc

from Dataset import MyDataset, MyDataset2
from SBERT import SentenceBERT
from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform
from scipy.io import loadmat


trainset_seqs = []
trainset_labels = []
testset_seqs = []
testset_labels = []
"""
with open('../data/STS/sts-train.csv', 'r', encoding='utf-8') as f:
    for line in f:
        tmp = line.strip().split('\t')
        trainset_seqs.append(tmp[5])
        trainset_seqs.append(tmp[6])
        trainset_labels.append(float(tmp[4]))
        trainset_labels.append(float(tmp[4]))
"""
with open('../data/STS/sts-test.csv', 'r', encoding='utf-8') as f:
    for line in f:
        tmp = line.strip().split('\t')
        testset_seqs.append(tmp[5])
        testset_labels.append(float(tmp[4]))
with open('../data/STS/sts-test.csv', 'r', encoding='utf-8') as f:
    for line in f:
        tmp = line.strip().split('\t')
        testset_seqs.append(tmp[6])
        testset_labels.append(float(tmp[4]))
"""
with open('../data/SNLI/train.txt', 'r', encoding='utf-8') as f:
    for line in f:
        tmp = line.strip().split('\t')
        trainset_seqs.append(tmp[0])
        trainset_seqs.append(tmp[1])
        trainset_labels.append(float(tmp[2]))
        trainset_labels.append(float(tmp[2]))
with open('../data/MultiNLI/train1.txt', 'r', encoding='utf-8') as f:
    for line in f:
        tmp = line.strip().split('\t')
        trainset_seqs.append(tmp[0])
        trainset_seqs.append(tmp[1])
        trainset_labels.append(float(tmp[2]))
        trainset_labels.append(float(tmp[2]))
with open('../data/SNLI/test.txt', 'r', encoding='utf-8') as f:
    for line in f:
        tmp = line.strip().split('\t')
        testset_seqs.append(tmp[0])
        testset_seqs.append(tmp[1])
        testset_labels.append(float(tmp[2]))
        testset_labels.append(float(tmp[2]))
with open('../data/MultiNLI/test1.txt', 'r', encoding='utf-8') as f:
    for line in f:
        tmp = line.strip().split('\t')
        testset_seqs.append(tmp[0])
        testset_seqs.append(tmp[1])
        testset_labels.append(float(tmp[2]))
        testset_labels.append(float(tmp[2]))
"""

m = loadmat("../experiment/recs/sts_red_3")
W = np.nan_to_num(np.array(m['W'], dtype=np.float32))
m = loadmat("../experiment/recs/stsb_neigh_3")
Neigh = np.array(m['neighborhood'], dtype=np.int)

# train_dataset = MyDataset(trainset_seqs, trainset_labels)
test_dataset = MyDataset2(testset_seqs, testset_labels, W, Neigh)

BATCH_SIZE = 64
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
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
            self.fc_var =  nn.Linear(h_dim, z_dim)
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

    
def log_likelihood(model, x, n=10):
    """
    :param model: model object
    :param optimizer: optimizer object
    :param n: number of MC samples
    :return: MC estimate of log-likelihood
    """

    z_mean, z_var = model.encode(x.reshape(-1, 768))
    q_z, p_z = model.reparameterize(z_mean, z_var)
    z = q_z.rsample(torch.Size([n]))
    x_mb_ = model.decode(z)

    log_p_z = p_z.log_prob(z)

    if model.distribution == 'normal':
        log_p_z = log_p_z.sum(-1)

    log_p_x_z = -nn.BCEWithLogitsLoss(reduction='none')(x_mb_, x.reshape(-1, 768).repeat((n, 1, 1))).sum(-1)

    log_q_z_x = q_z.log_prob(z)

    if model.distribution == 'normal':
        log_q_z_x = log_q_z_x.sum(-1)

    return ((log_p_x_z + log_p_z - log_q_z_x).t().logsumexp(-1) - np.log(n)).mean()


def train(model, premodel, optimizer):
    for i, (x_mb, y_mb, w_mb, neigh_mb) in enumerate(test_loader):
            if i % 32 == 0:
                beta = 0.0
            elif i % 32 <= 16:
                beta = 1 * ((i % 32) / 16)
            else:
                beta = 1
            optimizer.zero_grad()
            x_mb = premodel.get_sentence_vector(list(x_mb))
            # dynamic binarization
            # x_mb = (x_mb > torch.distributions.Uniform(0, 1).sample(x_mb.shape)).float()
            _, (q_z, p_z), _, x_mb_ = model(x_mb.reshape(-1, 768))
            x_mb_ngb = torch.tensor([])
            for j in range(len(x_mb)):
                neigh_mb_one = [testset_seqs[u-1] for u in neigh_mb[j]]
                neigh_mb_vec = premodel.get_sentence_vector(neigh_mb_one)
                _, (_, _), _, x_mb_ngb_tmp = model(neigh_mb_vec.reshape(-1, 768))
                neigh_mb_vec = torch.sum(neigh_mb_vec * torch.tensor(w_mb[j]).unsqueeze(1), 0)
                x_mb_ngb = torch.cat((x_mb_ngb, neigh_mb_vec.unsqueeze(0)), 0)
            # loss_recon = nn.BCEWithLogitsLoss(reduction='none')(x_mb_, x_mb.reshape(-1, 768)).sum(-1).mean()
            loss_recon = nn.MSELoss(reduction='mean')(x_mb_, x_mb.reshape(-1, 768))
            loss_ngb = nn.MSELoss(reduction='mean')(x_mb_, x_mb_ngb)
            if model.distribution == 'normal':
                loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean()
            elif model.distribution == 'vmf':
                loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).mean()
            else:
                raise NotImplemented

            loss = loss_recon + beta * loss_KL + loss_ngb

            loss.backward()
            optimizer.step()
            print(">>>      epoch:{0}       batch:{1}       loss:{2:.8f}        recon loss:{3:.8f}      KL loss:{4:.8f}     ngb loss:{5:.8f}".format(J, i, loss, loss_recon, loss_KL, loss_ngb))
            # print(">>>      epoch:{0}       batch:{1}       loss:{2:.8f}        recon loss:{3:.8f}      KL loss:{4:.8f}".format(J, i, loss, loss_recon, loss_KL))

    torch.save(model.state_dict(), '../parameter/VAE_STSb_MSE_PV_model')
            
def test(model, premodel):
    print_ = defaultdict(list)
    for x_mb, y_mb in test_loader:
        x_mb = premodel.get_sentence_vector(list(x_mb))
        # dynamic binarization
        x_mb = (x_mb > torch.distributions.Uniform(0, 1).sample(x_mb.shape)).float()
        
        _, (q_z, p_z), _, x_mb_ = model(x_mb.reshape(-1, 768))
        
        print_['recon loss'].append(float(nn.BCEWithLogitsLoss(reduction='none')(x_mb_,
            x_mb.reshape(-1, 768)).sum(-1).mean().data))
        
        if model.distribution == 'normal':
            print_['KL'].append(float(torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean().data))
        elif model.distribution == 'vmf':
            print_['KL'].append(float(torch.distributions.kl.kl_divergence(q_z, p_z).mean().data))
        else:
            raise NotImplemented
        
        print_['ELBO'].append(- print_['recon loss'][-1] - print_['KL'][-1])
        # print_['LL'].append(float(log_likelihood(model, x_mb).data))
    
    print({k: np.mean(v) for k, v in print_.items()})


# hidden dimension and dimension of latent space
config = 'bert-base-uncased'
EPOCH = 5
H_DIM = 128
Z_DIM = 5
max_length = 30
K = 3
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sbert = SentenceBERT(config, device, max_length)
param = torch.load('../pretrain_model/SBERT-STSb/pytorch_model.bin', map_location=device)
sbert_dict = sbert.state_dict()
sbert_dict.update({k: v for k, v in param.items() if 'bert' in k})
sbert.load_state_dict(sbert_dict)
sbert.eval()
"""
modelN = ModelVAE(h_dim=H_DIM, z_dim=Z_DIM, distribution='normal')
optimizerN = optim.Adam(modelN.parameters(), lr=1e-3)
for j in range(EPOCH):
    optimizerS = optim.SGD(modelN.parameters(), lr=1e-2 * (EPOCH - j) / EPOCH)
    train(modelN, sbert, optimizerS)
test(modelN, sbert)
"""
modelS = ModelVAE(h_dim=H_DIM, z_dim=Z_DIM + 1, distribution='vmf')
param = torch.load('../parameter/VAE_SNLI_STSb_BN_MSE_model', map_location=device)
modelS.load_state_dict(param)
gc.collect()
for J in range(EPOCH):
    optimizerS = optim.Adam(modelS.parameters(), lr=1e-3)
    train(modelS, sbert, optimizerS)
    test(modelS, sbert)
