import torch
import torch.nn as nn
from model_comp.attention import Attention, NewAttention, AttentionP
from model_comp.language_model import WordEmbedding, QuestionEmbedding
from model_comp.classifier import SimpleClassifier, AdvClassifier
from model_comp.fc import FCNet, GTH, SimpleFC
from model_comp.answer_classifier import AnswerClassifier
# from model_comp.grad_reversal_layer import GradientReversal
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

    def forward(self, q, v):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)  # [batch, q_dim]

        att, att_logits = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1)  # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits #, att_logits

class BUTDModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier):
        super(BUTDModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

    def forward(self, q, v):
        w_emb = self.embedding(q)
        q_emb = self.q_emb(w_emb)  # [batch, q_dim]
        att, att_logits = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1)  # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits#, att_logits



class EmbProj(nn.Module):
    def __init__(self):
        super(EmbProj, self).__init__()
        self.proj = SimpleFC([300, 1024], act=None, dropout=0.2)

    def forward(self, a_emb):
        proj = self.proj(a_emb)
        return proj


class ModelSiamese(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, answer_emb_mlp, classifier):
        super(ModelSiamese, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.answer_emb_mlp = answer_emb_mlp
        self.classifier = classifier

    def forward(self, q, v):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)  # [batch, q_dim]

        att, att_logits = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1)  # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr

        # joint_repr_emb = self.emb_fc(joint_repr)
        # contrastive
        joint_repr_emb = self.answer_emb_mlp(joint_repr)  # 2FC

        # VQA
        logits = self.classifier(joint_repr_emb)  # 2FC
        return logits, joint_repr_emb


def bottomUp(vocab_size, num_classes, glove_embed_dir):
    w_emb = WordEmbedding(vocab_size, glove_embed_dir, dropout=0.0)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=512, dropout=0.0)
    v_att = Attention(v_dim=2048, q_dim=512, num_hid=512)
    q_net = FCNet(dims=[512, 512], act="ReLU", dropout=0.0)
    v_net = FCNet(dims=[2048, 512], act="ReLU", dropout=0.0)
    classifier = SimpleClassifier(in_dim=512, hid_dim=2 * 512, out_dim=num_classes, dropout=0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)


def bottom_up_newatt(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, dropout=0.4)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=0.2)
    v_att = NewAttention(v_dim=dataset.v_dim, q_dim=q_emb.num_hid, num_hid=num_hid, dropout=0.3)
    q_net = FCNet(dims=[q_emb.num_hid, num_hid], act='ReLU', dropout=0.1)
    v_net = FCNet(dims=[dataset.v_dim, num_hid], act='ReLU', dropout=0.1)
    classifier = SimpleClassifier(in_dim=num_hid, hid_dim=num_hid * 2, out_dim=dataset.num_ans_candidates, dropout=0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)

def bottom_up_newatt_demo(dictionary, num_hid):
    w_emb = WordEmbedding(dictionary.ntoken, 300, dropout=0.4)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1, bidirect=False, dropout=0.2)
    v_att = NewAttention(v_dim= 2048, q_dim=q_emb.num_hid, num_hid=num_hid, dropout=0.3)
    q_net = FCNet(dims=[q_emb.num_hid, num_hid], act='ReLU', dropout=0.1)
    v_net = FCNet(dims=[2048, num_hid], act='ReLU', dropout=0.1)
    classifier = SimpleClassifier(in_dim=num_hid, hid_dim=num_hid * 2, out_dim=3129, dropout=0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)

def butd(pretrained_emb, token_size, answer_size, num_hid):
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, dropout=0.0)
    v_att = NewAttention(v_dim=2048, q_dim=q_emb.num_hid, num_hid=num_hid)
    q_net = FCNet(dims=[q_emb.num_hid, num_hid], act='ReLU', dropout=0.0)
    v_net = FCNet(dims=[2048, num_hid], act='ReLU', dropout=0.0)
    classifier = SimpleClassifier(in_dim=num_hid, hid_dim=num_hid * 2, out_dim=answer_size, dropout=0.5)
    return BUTDModel(pretrained_emb, token_size, q_emb, v_att, q_net, v_net, classifier)


def bottom_up_siamese(vocab_size, num_classes, glove_embed_dir, num_hid):
    w_emb = WordEmbedding(vocab_size, glove_embed_dir, dropout=0.0)
    q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, dropout=0.0)
    v_att = NewAttention(v_dim=2048, q_dim=q_emb.num_hid, num_hid=num_hid)
    q_net = FCNet(dims=[q_emb.num_hid, num_hid], act='ReLU', dropout=0.0)
    v_net = FCNet(dims=[2048, num_hid], act='ReLU', dropout=0.0)
    # joint_repr_emb = SimpleFC([num_hid, 300], act=None, dropout=0.0)
    answer_emb_mlp = SimpleClassifier(in_dim=num_hid, hid_dim=2048, out_dim=1024, dropout=0.5)
    classifier = SimpleClassifier(in_dim=num_hid, hid_dim=num_hid * 2, out_dim=num_classes, dropout=0.5)
    return ModelSiamese(w_emb, q_emb, v_att, q_net, v_net, answer_emb_mlp, classifier)

