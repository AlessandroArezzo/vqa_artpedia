import os
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import evaluation.evaluate as ev
import json

def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def train(model, train_loader, eval_loader, num_epochs, output):

    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0
    print(output)
    if os.path.isfile(output+'model.pth') == True:
        print('resume model from '+ output+'model.pth')
        print(output+'model.pth')
        ckpt = torch.load(output+'model.pth', map_location='gpu')
        model.load_state_dict(ckpt['state_dict'])
        optim.load_state_dict(ckpt['optim_state_dict'])
        best_eval_score = ckpt['score']
        print('resumed from epoch: ' + str(ckpt['epoch'] + 'score: ' + str(ckpt['score'])))


    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()

        for i, (v, b, q, a, _) in tqdm(enumerate(train_loader)):
            v = Variable(v).cuda()
            b = Variable(b).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()

            pred = model(q, v)
            loss = instance_bce_with_logits(pred, a)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.item() * v.size(0)
            train_score += batch_score

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)
        logger.write('epoch %d, time: %.2f' % (epoch, time.time() - t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
        if np.mod(epoch+1, 5) == 0 or epoch == 0:
            model.train(False)
            eval_score, bound = evaluate(model, eval_loader, output, label2ans= None)

            if eval_score > best_eval_score:
                model_path = os.path.join(output, 'model.pth')
                torch.save( {
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'optim_state_dict': optim.state_dict(),
                    'score': eval_score}, model_path)
                best_eval_score = eval_score
            logger.write('\teval score: %.2f (upper bound %.2f)' % (100 * eval_score, 100 * bound))
        model.train(True)


def evaluate(model, dataloader, output, label2ans, evaluate_with_official = False):
    score = 0
    upper_bound = 0
    num_data = 0
    model.eval()
    result = []

    preds = []
    gt = []
    for v, b, q, a, q_id in tqdm(iter(dataloader)):
        v = Variable(v, volatile=True).cuda()
        b = Variable(b, volatile=True).cuda()
        q = Variable(q, volatile=True).cuda()
        pred = model(q, v)
        batch_score = compute_score_with_logits(pred, a.cuda()).sum()
        score += batch_score
        upper_bound += (a.max(1)[0]).sum()
        num_data += pred.size(0)

        for j in range(0, len(pred)):
            preds.append(torch.argmax(pred[j]).cpu().data)
            gt.append(torch.argmax(a[j]).cpu().data)

        if evaluate_with_official == True:
            pred_idx = torch.argmax(pred, dim=1).cpu().data.numpy()
            for idx in range(0, len(pred)):
                result.append({
                    'answer': label2ans[pred_idx[idx]],  # ix_to_ans(load with json) keys are type of string
                    'question_id': int(q_id[idx])
                })

    from sklearn.metrics import confusion_matrix
    labels = [i for i in range(0,len(label2ans))]
    conf_matrix = confusion_matrix(gt, preds,labels=labels)
    np.save(output + 'conf_matrix.npy', conf_matrix)

    if evaluate_with_official == True:
        json.dump(result, open(output + 'result.pkl','w'))
        ev.Evaluate('/equilibrium/pietrobongini/VQA/Annotations/v2_mscoco_val2014_annotations.json','/equilibrium/pietrobongini/VQA/Questions/v2_OpenEnded_mscoco_val2014_questions.json', output + 'result.pkl')

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    return score, upper_bound


