import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
import torch
from model_comp.fc import GTH


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            #torch.mul(torch.tanh(weight_norm(nn.Linear(in_dim, hid_dim), dim=None),torch.sigmoid(weight_norm(nn.Linear(in_dim, hid_dim), dim=None)) )),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits

class MCAnswerPrediction(nn.Module):
    def __init__(self, qoi_dim, qo_dim, hid_dim):
        super(MCAnswerPrediction, self).__init__()
        #self.gated_tanh_1 = GTH([in_dim, hid_dim_1],act = "Tanh", dropout = dropout)
        #self.gated_tanh_2 = GTH([in_dim, hid_dim_2],act = "Tanh", dropout = dropout)
        self.project_qoi = weight_norm(nn.Linear(qoi_dim, hid_dim), dim=None)
        self.project_qi = weight_norm(nn.Linear(qo_dim, hid_dim), dim=None)
        self.project_opt = weight_norm(nn.Linear(1024, hid_dim), dim=None)
        self.relu = nn.ReLU()
        self.classifier = SimpleClassifier(in_dim= 1024, hid_dim=512, out_dim=1, dropout=0.5)
    def forward(self, qoi, qi, opts):
        project_qoi = self.project_qoi(qoi)
        project_qi = self.project_qi(qi)
        M = self.relu(project_qoi + project_qi)
        #logits = self.classifier(P)
        P_0 = self.classifier(M * opts[:, 0, :])
        P_1 = self.classifier(M * opts[:, 1, :])
        P_2 = self.classifier(M * opts[:, 2, :])
        P_3 = self.classifier(M * opts[:, 3, :])
        P_4 = self.classifier(M * opts[:, 4, :])
        logits = torch.stack((P_0,P_1,P_2,P_3,P_4),dim=1).squeeze(2)
        #torch.cat((P_0, P_1,P_2,P_3,P_4), dim=1)
        return logits

class AdvClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim_1, hid_dim_2, out_dim, dropout):
        super(AdvClassifier, self).__init__()
        self.gated_tanh_1 = GTH([in_dim, hid_dim_1],act = "Tanh", dropout = dropout)
        self.gated_tanh_2 = GTH([in_dim, hid_dim_2],act = "Tanh", dropout = dropout)
        self.linear_1 = weight_norm(nn.Linear(hid_dim_1, out_dim), dim=None)
        self.linear_2 = weight_norm(nn.Linear(hid_dim_2, out_dim), dim=None)
        
    def forward(self, x):
        v_1 = self.gated_tanh_1(x)
        v_2 = self.gated_tanh_2(x)
        
        v_1 = self.linear_1(v_1)
        v_2 = self.linear_2(v_2)
        
        logits = v_1 + v_2
        return logits