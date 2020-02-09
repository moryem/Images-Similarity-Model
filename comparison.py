# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 21:41:04 2020

@author: Mor
"""

import os
from utils import data_prep_perc
from DeepModel import similarity_net
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim



# Read the data
data_path = os.path.normpath(os.getcwd() + os.sep + os.pardir)
root = os.path.join(data_path,'cifar-10-batches-py')
right_train, right_val, right_test, left_train, left_val, left_test, y_tr, y_val, y_test, l_tr, l_val, l_te = data_prep_perc(root)

# Define hyper-parameters
bs = 32; ep = 1000; dp = 0.55; lr = 5e-5
imgs = [1, 20, 4, 15, 8011, 8060, 8033, 344]

reg = similarity_net(dp, bs, ep, lr, l_tr, l_val, l_te, right_train, right_val, right_test,
                 left_train, left_val, left_test, im_to_vis = imgs, vis = 0, plt_loss = 0, plt_mat=0, plt_model=0)

# predict and visualize on test set
reg.test()
y_pred = reg.y_pred

#%% comparison to MSE between the images

MSE = lambda x: (x**2).mean(axis=None)
mse_pairs = list(map(MSE,(right_test - left_test)))
mse_pairs = np.asarray(mse_pairs)

# mae
mae_value = mae(l_te, mse_pairs)
print('mae = ' + str(mae_value))
# mse
mse_value = mse(l_te, mse_pairs)
print('mse = ' + str(mse_value))
# R2
r2 = r2_score(l_te, mse_pairs)
print('R2 = ' + str(r2))

# plot dependencies
plt.scatter(l_te, mse_pairs, s=10, facecolors='none', edgecolors='black')
plt.xlabel('True Target')
plt.ylabel('MSE between pairs')
plt.legend(['$R^2$ = ' + str(np.round(r2,decimals=2))])
plt.savefig('Evaluation MSE')
plt.show(); plt.close()

#%%
# Comparison to SSIM

ssim_pairs = [ssim(right_test[i,:,:,:],left_test[i,:,:,:],multichannel=True) for i in range(right_test.shape[0])]
ssim_pairs = np.asarray(ssim_pairs)
dssim = (1-ssim_pairs)/2

# DSSIM results
# mae
dssim_mae_value = mae(l_te, dssim)
print('dssim mae = ' + str(dssim_mae_value))
# mse
dssim_mse_value = mse(l_te, dssim)
print('dssim mse = ' + str(dssim_mse_value))
# R2
dssim_r2 = r2_score(l_te,dssim)
print('dssim R2 = ' + str(dssim_r2))

# plot dependencies
plt.scatter(l_te, dssim, s=10, facecolors='none', edgecolors='black')
plt.xlabel('True Target')
plt.ylabel('DSSIM between pairs')
plt.legend(['$R^2$ = ' + str(np.round(dssim_r2,decimals=2))])
plt.savefig('Evaluation DSSIM')
plt.show(); plt.close()








