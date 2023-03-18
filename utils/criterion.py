
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

class NCESoftmaxLoss(nn.Module):
    def  __init__(self, batch_size,cur_device):
        # keep code reptitive
        super(NCESoftmaxLoss,self).__init__()
        self.device =  cur_device
        self.temperature = 0.07
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = batch_size
        self.n_views = 2


    def info_nce_loss(self, features):
        # fesatures.shape[512,128]

        labels = torch.cat([torch.arange(self.batch_size) for i in range(self.n_views)],
                           dim=0)  # labels [512]  [0-255]+[0-255]
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # convert label in format of(0,1)
        labels = labels.to(self.device)  # labels [512,512]

        features = F.normalize(features, dim=1)  # for per sample

        # congusion matrix
        similarity_matrix = torch.matmul(features, features.T)  # similarity matrix[512,512]

        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix (i.e, discard all positive sample)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)  # mask [512,512]

        labels = labels[~mask].view(labels.shape[0], -1)  # labels [512,511]
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)  # similarity_matrix [512,511]
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to( self.device)

        logits = logits / self.temperature
        return logits, labels

    def forward(self, logits, label):
        contrastive_features = torch.concat((logits,label),dim=0)
        logits, labels = self.info_nce_loss(contrastive_features)
        loss = self.criterion(logits, labels)
        return loss



