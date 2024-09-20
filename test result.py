import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#open numpy file
sim_data = np.load('./OUTPUT/None/ddpm_fake_None.npy')
gt_data = np.load('./OUTPUT/None/samples/etth_norm_truth_24_train.npy')
print(sim_data.shape)
print(gt_data.shape)    #sim_data形状为18009，gt_data形状为17397
sim_data = sim_data[:gt_data.shape[0]] #该公式由metric_pytorch得到，因为sim的时候受到整bantch的影响。
idx = 5
plt.plot(sim_data[idx,:,-1], label='sim')
plt.plot(gt_data[idx,:,-1], label='gt')
plt.legend()
plt.show()
mse_value_list = []
mae_value_list = []
rmse_value_list = []
r2_value_list = []

sim_data = sim_data[:,:,-1]
gt_data = gt_data[:,:,-1]


sim_data = sim_data.flatten()
gt_data = gt_data.flatten()

mse_value = mean_squared_error(sim_data, gt_data)
mae_value = mean_absolute_error(sim_data, gt_data)
rmse_value = np.sqrt(mse_value)
r2_value = r2_score(sim_data, gt_data)
print('mse:', mse_value)
print('mae:', mae_value)
print('rmse:', rmse_value)
print('r2:', r2_value)

