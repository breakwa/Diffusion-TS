import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sim_data = np.load('./OUTPUT/None/ddpm_fake_None.npy')
gt_data = np.load('./OUTPUT/None/samples/etth_norm_truth_24_train.npy')
print(sim_data.shape)
print(gt_data.shape)    #sim_data形状为18009，gt_data形状为17397
sim_data = sim_data[:gt_data.shape[0]] #该公式由metric_pytorch得到，因为sim的时候受到整bantch的影响。
idx = 10
plt.plot(sim_data[idx,:,-1], linestyle='-.', color='y', label=f'Simulated')
plt.plot(gt_data[idx,:,-1], linestyle='--', color='k', label=f'Sampled(Ground Truth)')
plt.title('sample id: {}'.format(idx))
plt.ylim(0, 1)
plt.legend()
plt.show()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append(os.path.join(os.path.dirname('__file__'), './'))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from Utils.metric_utils import display_scores
from Utils.discriminative_metric import discriminative_score_metrics
from Utils.predictive_metric import predictive_score_metrics


'''discriminative score'''
iterations = 5
sim_data = np.load('./OUTPUT/None/ddpm_fake_None.npy')
gt_data = np.load('./OUTPUT/None/samples/etth_norm_truth_24_train.npy')


predictive_score = []
for i in range(iterations):
    temp_pred = predictive_score_metrics(gt_data[:], sim_data[:gt_data.shape[0]])
    predictive_score.append(temp_pred)
    print(i, ' epoch: ', temp_pred, '\n')

print('sine:')
display_scores(predictive_score)
print()