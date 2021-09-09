from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):

    def __init__(self, seqs, grades):
        super(MyDataset, self).__init__()
        self.seqs = seqs
        self.grades = grades

    def __getitem__(self, index):
        seq = self.seqs[index]
        grade = self.grades[index]
        return seq, grade

    def __len__(self):
        return len(self.seqs)


class MyDataset1(Dataset):

    def __init__(self, seqs1, seqs2, grades):
        super(MyDataset1, self).__init__()
        self.seqs1 = seqs1
        self.seqs2 = seqs2
        self.grades = grades

    def __getitem__(self, index):
        seq1 = self.seqs1[index]
        seq2 = self.seqs2[index]
        grade = self.grades[index]
        return seq1, seq2, grade

    def __len__(self):
        return len(self.seqs1)


class MyDataset2(Dataset):

    def __init__(self, seqs, grades, ws, neighs):
        super(MyDataset2, self).__init__()
        self.seqs = seqs
        self.grades = grades
        self.ws = ws
        self.neighs = neighs

    def __getitem__(self, index):
        seq = self.seqs[index]
        grade = self.grades[index]
        w = self.ws[index]
        neigh = self.neighs[index]
        return seq, grade, w, neigh

    def __len__(self):
        return len(self.seqs)
