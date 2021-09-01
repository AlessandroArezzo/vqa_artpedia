import torch
import torch.nn as nn
import base_model
import dataset
from dataset import Dictionary
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable
import pickle
import os
import json
import argparse
import pandas as pd
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='./data',
                    help='data root features dictionary pickle file')
parser.add_argument('--dataset', type=str, default='artpedia',
                    help='type of images to considered: artpedia or artpedia_dt')
opt = parser.parse_args()
features_path = os.path.join(opt.data_root, "dict_features_"+opt.dataset+".pkl")
correct_rslt_path = os.path.join(opt.data_root, "output", "correct_pred_"+opt.dataset+".csv")
error_rslt_path = os.path.join(opt.data_root, "output", "error_pred_"+opt.dataset+".csv")
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

#UPLOAD GROUND TRUTH QUESTION/ANSWERS
with open('./data/artpedia_bottomup_qa.json') as f:
  question_answer_dict = json.load(f)
with open('./data/artpedia_visual_qa_cap.json') as f:
  question_answer_dict.update(json.load(f))

#UPLOAD IMAGES FEATURES
with open(features_path, 'rb') as handle:
    features_dict = pickle.load(handle)

#UPLOAD DICTIONARY INDEXES TO ANSWERS
with open('./data/dict_ans.pkl', 'rb') as handle:
    idx2ans = pickle.load(handle)[0]

#Generate Data Loader
batch_size = 1
test_loader = dataset.ArtPediaDataset(features_dict=features_dict, question_answer_dict=question_answer_dict,
                                      dictionary=dictionary)
dataloader = DataLoader(test_loader, batch_size, shuffle=False, num_workers=1)

output_columns = ['idx_image', 'question_token', 'question', 'answer', 'pred']
out_df_correct_rslts = pd.DataFrame(columns=output_columns)
out_df_error_rslts = pd.DataFrame(columns=output_columns)

#EVAL MODEL
with torch.no_grad():
    score = 0
    for image_idx, features, question_token, question, answer_token, answer in tqdm(iter(dataloader)):
        #v = Variable(v, volatile=True).cuda()
        #q = Variable(q, volatile=True).cuda()
        v = Variable(features, volatile=True)
        q = Variable(question_token, volatile=True)
        pred = model(q, v) # generate prediction
        pred_ans_idx = torch.argmax(pred, dim=1) # extract index of the dictionary of the word answer
        pred_word = idx2ans[pred_ans_idx] # get word answer from the index
        answer_idx = (answer_token == 1).nonzero(as_tuple=True)[1]  # get the indexes representation of the ground truth answer
        # get the answer from the indexes word
        answer_string = ""
        for idx in answer_idx:
            answer_string += dictionary.idx2word[idx] + " "
        # prepare data to write in file
        data_to_add = [image_idx[0], question_token.tolist()[0], question, answer[0], pred_word]
        data_df_scores = np.hstack((np.array(data_to_add).reshape(1, -1)))
        # eval prediction: if the word predicted is contained in ground truth the prediction is correctly
        correct_pred = False
        for answer_word in answer_string.split(" "):
            if answer_word == pred_word:
                score += 1
                out_df_correct_rslts = out_df_correct_rslts.append(pd.Series(data_df_scores.reshape(-1),
                                                                 index=out_df_correct_rslts.columns),
                                                                 ignore_index=True)
                out_df_correct_rslts.to_csv(correct_rslt_path, index=False, header=True)
                correct_pred = True
                break
        if not correct_pred:
            out_df_error_rslts = out_df_error_rslts.append(pd.Series(data_df_scores.reshape(-1),
                                                                         index=out_df_error_rslts.columns),
                                                                         ignore_index=True)
            out_df_error_rslts.to_csv(error_rslt_path, index=False, header=True)

    # total acuracy is calculated: (number of the correct prediction / total prediction) * 100
    accuracy = (score/len(dataloader))*100
    print("ACCURACY: "+"{:.2f}".format(accuracy)+"%")

