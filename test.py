import os
import time
import string
import re

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from nltk.metrics.distance import edit_distance

from utils import get_args
from utils import Averager

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def validation(model, criterion, evaluation_loader, converter, opt):
    """ validation or evaluation """
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()

    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        # For max length prediction
        target = converter.encode(labels)

        start_time = time.time()

        preds = model(image, seqlen=converter.batch_max_length)
        _, preds_index = preds.topk(1, dim=-1, largest=True, sorted=True)
        preds_index = preds_index.view(-1, converter.batch_max_length)
        forward_time = time.time() - start_time
        cost = criterion(preds.view(preds.shape[0], preds.shape[2], preds.shape[1]),
                         target.view(target.shape[0], target.shape[2], target.shape[1]))

        length_for_pred = torch.IntTensor([converter.batch_max_length - 1] * batch_size).to(device)
        preds_str = converter.decode(preds_index[:, 1:], length_for_pred)

        infer_time += forward_time
        valid_loss_avg.add(cost)

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):

            pred_EOS = pred.find('[s]')
            pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
            pred_max_prob = pred_max_prob[:pred_EOS]

            # To evaluate 'case sensitive model' with alphanumeric and case insensitve setting.
            if opt.sensitive and opt.data_filtering_off:
                pred = pred.lower()
                gt = gt.lower()
                alphanumeric_case_insensitve = '0123456789abcdefghijklmnopqrstuvwxyz'
                out_of_alphanumeric_case_insensitve = f'[^{alphanumeric_case_insensitve}]'
                pred = re.sub(out_of_alphanumeric_case_insensitve, '', pred)
                gt = re.sub(out_of_alphanumeric_case_insensitve, '', gt)

            if pred == gt:
                n_correct += 1

            # ICDAR2019 Normalized Edit Distance
            if len(gt) == 0 or len(pred) == 0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)

            # calculate confidence score (= multiply of pred_max_prob)
            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
            confidence_score_list.append(confidence_score)
            # print(pred, gt, pred==gt, confidence_score)

    accuracy = n_correct / float(length_of_data) * 100
    norm_ED = norm_ED / float(length_of_data)  # ICDAR2019 Normalized Edit Distance

    return valid_loss_avg.val(), accuracy, norm_ED, preds_str, confidence_score_list, labels, infer_time, length_of_data

