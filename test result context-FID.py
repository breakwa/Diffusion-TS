import os
import torch
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname('__file__'), './'))
from Utils.context_fid import Context_FID
from Utils.metric_utils import display_scores
from Utils.cross_correlation import CrossCorrelLoss

iterations = 5

sim_data = np.load('./OUTPUT/None/ddpm_fake_None.npy')
gt_data = np.load('./OUTPUT/None/samples/etth_norm_truth_24_train.npy')


context_fid_score = []

for i in range(iterations):
    context_fid = Context_FID(gt_data[:], sim_data[:gt_data.shape[0]])
    context_fid_score.append(context_fid)
    print(f'Iter {i}: ', 'context-fid =', context_fid, '\n')

display_scores(context_fid_score)