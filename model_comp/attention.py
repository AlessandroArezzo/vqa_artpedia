import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from model_comp.fc import FCNet,GTH,SimpleFC

class TripleAttention(nn.Module):
    def __init__(self, v_dim, q_dim, o_dim, num_hid, dropout=0.2):
        super(TripleAttention, self).__init__()

        self.v_proj = FCNet([v_dim, num_hid],act='ReLU',dropout=dropout)
        self.q_proj = FCNet([q_dim, num_hid],act='ReLU',dropout=dropout)
        self.o_proj = FCNet([o_dim, num_hid],act='ReLU',dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(q_dim, 1), dim=None)

    def forward(self, v, q, o):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q, o)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q, o):
        # 512 x 36 x 2048
        # 512 x 1024
        batch, k, _ = v.size()
        v_proj = self.v_proj(v) # [batch, k, num_hid] #512 x 36 x 1024
        #print(v_proj.size())
        q_proj = self.q_proj(q) # [512,1024]
        #print(q_proj.size())
        o_proj = self.o_proj(o) #[512, 1024]
        #print(o_proj.size())
        q_proj = q_proj.unsqueeze(1).repeat(1, k, 1)  #512 x36x1024
        o_proj = o_proj.unsqueeze(1).repeat(1, k, 1)
        joint_repr = v_proj * q_proj * o_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits

class Attention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid):
        super(Attention, self).__init__()
        self.nonlinear = FCNet([v_dim + q_dim, num_hid],act="ReLU",dropout=0.0)
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)


    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        vq = torch.cat((v, q), 2)
        joint_repr = self.nonlinear(vq)
        logits = self.linear(joint_repr)
        return logits
        
class AttentionP(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid):
        super(AttentionP, self).__init__()
        self.gated_tanh = GTH([v_dim + q_dim, num_hid], act="Tanh",dropout=0.3)
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)
        

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        vq = torch.cat((v, q), 2)
        joint_repr = self.gated_tanh(vq)
        logits = self.linear(joint_repr)
        return logits


class NewAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(NewAttention, self).__init__()

        self.v_proj = FCNet([v_dim, num_hid],act='ReLU',dropout=dropout)
        self.q_proj = FCNet([q_dim, num_hid],act='ReLU',dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(q_dim, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w, logits

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v) # [batch, k, qdim]
        q_proj = self.q_proj(q)
        q_proj= q_proj.unsqueeze(1).repeat(1, k, 1)
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits

class MCAttention(nn.Module):
    def __init__(self, el1_dim, el2_dim, num_hid, dropout=0.2):
        super(MCAttention, self).__init__()

        self.el1_encoding = SimpleFC([el1_dim, num_hid],act='ReLU',dropout=dropout)#FCNet([v_dim, num_hid],act='ReLU',dropout=dropout)
        self.el2_linear = weight_norm(nn.Linear(el2_dim, num_hid), dim=None)
        self.dropout = nn.Dropout(dropout)
        self.o_hat_proj = weight_norm(nn.Linear(el1_dim, num_hid), dim=None)
        self.el2_proj = weight_norm(nn.Linear(el2_dim, num_hid), dim=None)
        #self.final_linear = weight_norm(nn.Linear(el2_dim, 1), dim=None)

    def forward(self, el1, el2):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(el1, el2)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, el1, el2):
        batch, k, _ = el1.size()
        b_1 = self.el1_encoding(el1)
        b_2 = self.el2_linear(el2)
        el2_proj= b_2.unsqueeze(1).repeat(1, k, 1)
        beta = b_1 * el2_proj
        beta = nn.functional.softmax(beta, 1)
        o_hat  = (el1 * beta).sum(1)
        o_hat_proj = self.o_hat_proj(o_hat)
        el2_proj = self.el2_proj(el2)
        logits = o_hat_proj + el2_proj
        return logits

class QuestionAnswerContext(nn.Module):
    def __init__(self, o_dim, q_dim, num_hid, dropout=0.2):
        super(QuestionAnswerContext, self).__init__()
        self.o_proj = FCNet([o_dim, num_hid], act='ReLU', dropout=dropout)
        self.q_proj = FCNet([q_dim, num_hid],act='ReLU',dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(q_dim, 1), dim=None)

    def forward(self, q, o):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(q, o)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, q, o):
        o_proj = self.o_proj(o) #[512,1024]
        q_proj = self.q_proj(q) #[512,1024]
        joint_repr = q_proj * o_proj #[512,1024] #
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr) #[512,1]
        return logits
# class OptionAttention(nn.module):
#     def __init__(self, o_dim, q_dim, num_hid, dropout=0.2):
#         super(NewAttention, self).__init__()
#
#         self.o_proj = FCNet([o_dim, num_hid],act='ReLU',dropout=dropout)
#         self.q_proj = FCNet([q_dim, num_hid],act='ReLU',dropout=dropout)
#         self.dropout = nn.Dropout(dropout)
#         self.linear = weight_norm(nn.Linear(q_dim, 1), dim=None)
#
#     def forward(self, q, o):
#         """
#         v: [batch, k, vdim]
#         q: [batch, qdim]
#         """
#         logits = self.logits(q, o)
#         w = nn.functional.softmax(logits, 1)
#         return w
#
#     def logits(self, q, o):
#         #batch, k, _ = v.size()
#         o_proj = self.o_proj(o)
#         print(o_proj.size())
#         q_proj = self.q_proj(q)
#         print(q_proj.size())
#         #q_proj= q_proj.unsqueeze(1).repeat(1, k, 1)
#         joint_repr = q_proj * o_proj
#         print(joint_repr.size())
#         joint_repr = self.dropout(joint_repr)
#         logits = self.linear(joint_repr)
#         return logits
