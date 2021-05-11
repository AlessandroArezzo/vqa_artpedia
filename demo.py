import torch
import torch.nn as nn
import base_model
import dataset
from dataset import Dictionary, VQAFeatureDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable
import pickle
import os
import json
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--features_path', type=str, default='./data/dict_features_artpedia.pkl',
                    help='features dictionary pickle file path')
opt = parser.parse_args()

idx2ans = pickle.load(open(
    './data/dict_ans.pkl',
    'rb'))
#BUILD THE VQA MODEL
num_hid = 1024
constructor = 'bottom_up_newatt_demo'
dictionary = Dictionary.load_from_file('./data/dictionary.pkl')
#model = getattr(base_model, constructor)(dictionary, num_hid).cuda()
model = getattr(base_model, constructor)(dictionary, num_hid)
model.w_emb.init_embedding('./data/glove6b_init_300d.npy')
#model = nn.DataParallel(model).cuda()
model = nn.DataParallel(model)

#UPLOAD THE PRETRAINED WEIGHTS
print('upload model.pth')
#ckpt = torch.load('./data/model.pth')
ckpt = torch.load('./data/model.pth', map_location=torch.device('cpu'))
model.load_state_dict(ckpt['state_dict'])

with open('./data/artpedia_bottomup_qa.json') as f:
  question_answer_dict = json.load(f)
with open('./data/artpedia_visual_qa_cap.json') as f:
  question_answer_dict.update(json.load(f))

with open(opt.features_path, 'rb') as handle:
    features_dict = pickle.load(handle)

with open('./data/dict_ans.pkl', 'rb') as handle:
    idx2ans = pickle.load(handle)[0]
batch_size = 1
test_loader = dataset.ArtPediaDataset(features_dict=features_dict, question_answer_dict=question_answer_dict,
                                      dictionary=dictionary)
dataloader = DataLoader(test_loader, batch_size, shuffle=False, num_workers=1)

with torch.no_grad():

    score = 0
    for v, q, a in tqdm(iter(dataloader)):
        #v = Variable(v, volatile=True).cuda()
        #q = Variable(q, volatile=True).cuda()
        v = Variable(v, volatile=True)
        q = Variable(q, volatile=True)
        pred = model(q, v)
        #TODO get the answer idx
        pred_ans_idx = torch.argmax(pred, dim=1)
        pred_word = idx2ans[pred_ans_idx]
        answer_idx = (a == 1).nonzero(as_tuple=True)[1]
        answer_string = ""
        for idx in answer_idx:
            answer_string += dictionary.idx2word[idx] + " "
        for answer_word in answer_string.split(" "):
            if answer_word == pred_word:
                score += 1
                break
    accuracy = (score/len(dataloader))*100
    print("ACCURACY: "+"{:.2f}".format(accuracy)+"%")

