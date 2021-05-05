import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import Dictionary, VQAFeatureDataset
import base_model
from exec import train, evaluate
import utils
import os
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
epochs = 30
num_hid = 1024
output = 'butd_baseline/'
batch_size = 512
ckpts_dir = 'ckpts/'
save_dir = ckpts_dir + output
only_evaluate = True

if os.path.isdir(ckpts_dir) == False:
    os.mkdir(ckpts_dir)

if os.path.isdir(save_dir) == False:
    os.mkdir(save_dir)

if __name__ == '__main__':
    #torch.manual_seed(args.seed)
    #torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file('./preprocessing/data/dictionary.pkl')
    train_dset = VQAFeatureDataset('train', dictionary)
    eval_dset = VQAFeatureDataset('val', dictionary)
    batch_size = batch_size

    constructor = 'bottom_up_newatt'
    model = getattr(base_model, constructor)(train_dset, num_hid).cuda()
    model.w_emb.init_embedding('./preprocessing/data/glove6b_init_300d.npy')

    model = nn.DataParallel(model).cuda()

    if only_evaluate:
        if os.path.isfile(save_dir + 'model.pth') == True:
            print('resume model from ' + output + 'model.pth')
            print(output + 'model.pth')
            ckpt = torch.load(save_dir + 'model.pth')
            model.load_state_dict(ckpt['state_dict'])
            model.eval()
            #best_eval_score = ckpt['score']
            #print('resumed from epoch: ' + str(ckpt['epoch'] + 'score: ' + str(ckpt['score'])))

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1)
    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=1)

    #train(model, train_loader, eval_loader, epochs, save_dir)
    _,_ = evaluate(model, eval_loader, save_dir, eval_dset.label2ans, evaluate_with_official = True)