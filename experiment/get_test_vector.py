import torch

from Dataset import MyDataset
from SBERT import SentenceBERT


def nearest_neighbors(data, k=3):
    neighbors = torch.tensor([])
    score = data @ data.t()
    index = score.sort(1, descending=True)[0][:, 0: k]
    for i in range(len(index)):
        neighbor = data.index_select(0, index[i])
        neighbors = torch.cat((neighbors, neighbor.unsqueeze(0)), 0)
    return neighbors


testset_seqs = []
testset_labels = []
with open('../data/STS/sts-test.csv', 'r', encoding='utf-8') as f:
    for line in f:
        tmp = line.strip().split('\t')
        testset_seqs.append(tmp[5])
        testset_seqs.append(tmp[6])
        testset_labels.append(float(tmp[4]))
        testset_labels.append(float(tmp[4]))


config = 'bert-base-uncased'
max_length = 30
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sbert = SentenceBERT(config, device, max_length)
param = torch.load('../pretrain_model/SBERT-STSb/pytorch_model.bin', map_location=device)
sbert_dict = sbert.state_dict()
sbert_dict.update({k: v for k, v in param.items() if 'bert' in k})
sbert.load_state_dict(sbert_dict)
sbert.eval()
w = []
for i in range(len(testset_seqs)):
    tmp = torch.zeros(len(testset_seqs))
    for j in range(len(testset_seqs)):
        if i != j:
            print(i,j)
            vector = sbert.get_sentence_vector([testset_seqs[i], testset_seqs[j]])
            dot = vector[0] * vector[1]
            tmp[j] = dot.sum()
    w.append(tmp.sort(0, descending=True)[1][:3])
torch.save(w, "../experiment/recs/index_3")
