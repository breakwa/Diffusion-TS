import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sim_data = np.load('./OUTPUT/None/ddpm_fake_None.npy')
gt_data = np.load('./OUTPUT/None/samples/etth_norm_truth_24_train.npy')
print(sim_data.shape)
print(gt_data.shape)    #sim_data形状为18009，gt_data形状为17397
sim_data = sim_data[:gt_data.shape[0]] #该公式由metric_pytorch得到，因为sim的时候受到整bantch的影响。
idx = [2,3,5,7]
for idxi in idx:
    plt.plot(sim_data[idxi, :, -1], linestyle='-.', color='y', label=f'Simulated')
    plt.plot(gt_data[idxi, :, -1], linestyle='--', color='k', label=f'Sampled(Ground Truth)')
    plt.title('sample id: {}'.format(idxi))
    plt.legend()
    plt.ylim(0,1)
    plt.savefig('./OUTPUT/None/samples/etth_norm_truth_24_train_id{}.png'.format(idxi))
    plt.show()
    plt.close()