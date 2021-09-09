import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer


class SentenceBERT(nn.Module):
    def __init__(self, config, device, max_length):
        super(SentenceBERT, self).__init__()

        self.max_length = max_length
        self.device = device
        self.dim = 768

        self.config = config
        self.config = AutoConfig.from_pretrained(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config)
        self.bert = AutoModel.from_pretrained(config).to(self.device)
        self.dnn = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.dim * 3, 3),
        ).to(device)

    def forward(self, sentence1, sentence2):
        input1 = self.tokenizer(sentence1, padding='max_length', truncation=True, max_length=self.max_length)
        input1_ids = torch.tensor(input1['input_ids'], dtype=torch.long).to(self.device)
        input1_att = torch.tensor(input1['attention_mask'], dtype=torch.long).to(self.device)
        input2 = self.tokenizer(sentence2, padding='max_length', truncation=True, max_length=self.max_length)
        input2_ids = torch.tensor(input2['input_ids'], dtype=torch.long).to(self.device)
        input2_att = torch.tensor(input2['attention_mask'], dtype=torch.long).to(self.device)
        tensor1 = self.bert(input1_ids, attention_mask=input1_att)[0]
        tensor2 = self.bert(input2_ids, attention_mask=input2_att)[0]
        out1 = (tensor1 * input1_att.unsqueeze(-1).expand(tensor1.size())).sum(1) / input1_att.sum(1).unsqueeze(1)
        out2 = (tensor2 * input2_att.unsqueeze(-1).expand(tensor2.size())).sum(1) / input2_att.sum(1).unsqueeze(1)
        difference = torch.abs(out1 - out2).to(self.device)
        return self.dnn(torch.cat((out1, out2, difference), 1)).to(self.device)

    def get_sentence_vector(self, sentence):
        input = self.tokenizer(sentence, padding='max_length', truncation=True, max_length=self.max_length)
        input_ids = torch.tensor(input['input_ids'], dtype=torch.long).to(self.device)
        input_att = torch.tensor(input['attention_mask'], dtype=torch.long).to(self.device)
        tensor = self.bert(input_ids, attention_mask=input_att)[0]
        out = (tensor * input_att.unsqueeze(-1).expand(tensor.size())).sum(1) / input_att.sum(1).unsqueeze(1)
        return out
