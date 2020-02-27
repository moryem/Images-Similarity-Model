# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 21:41:04 2020

@author: Mor
"""

import os
from utils import data_prep_perc
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from DeepModel import similarity_net


def KPIs(y_true, y_pred, name):
    mae_value = mae(y_true, y_pred)
    print('mae = ' + str(mae_value))
    # mse
    mse_value = mse(y_true, y_pred)
    print('mse = ' + str(mse_value))
    # R2
    r2 = r2_score(y_true, y_pred)
    print('R2 = ' + str(r2))
    
    return r2

def plot_KPIs(y_true, y_pred, name):
    
    r2 = KPIs(y_true, y_pred, name)
    plt.scatter(y_true, y_pred, s=10, facecolors='none', edgecolors='black')
    plt.xlabel('True Target')
    plt.ylabel(name + ' between pairs')
    plt.legend(['$R^2$ = ' + str(np.round(r2,decimals=2))])
    plt.savefig('Evaluation ' + name)
    plt.show(); plt.close()
    
#%% 
# Read the data
data_path = os.path.normpath(os.getcwd() + os.sep + os.pardir)
root = os.path.join(data_path,'cifar-10-batches-py')
right_train, right_val, right_test, left_train, left_val, left_test, y_tr, y_val, y_test, l_tr, l_val, l_te = data_prep_perc(root)

#%% 
# =============================================================================
# Comparison to MSE between the images
# =============================================================================

MSE = lambda x: (x**2).mean(axis=None)
mse_pairs = list(map(MSE,(right_test - left_test)))
mse_pairs = np.asarray(mse_pairs)

plot_KPIs(l_te, mse_pairs, 'MSE')

#%%
# =============================================================================
# Comparison to DSSIM
# =============================================================================

ssim_pairs = [ssim(right_test[i,:,:,:],left_test[i,:,:,:],multichannel=True) for i in range(right_test.shape[0])]
ssim_pairs = np.asarray(ssim_pairs)
dssim = (1-ssim_pairs)/2

plot_KPIs(l_te, dssim, 'DSSIM')

#%%
# =============================================================================
# Comparison to IMGonline
# =============================================================================

# Get our model's results

# Define hyper-parameters
bs = 32; ep = 1000; dp = 0.55; lr = 5e-5
imgs = [1, 20, 4, 15, 8011, 8060, 8033, 344]

sim = similarity_net(dp, bs, ep, lr, l_tr, l_val, l_te, right_train, right_val, right_test,
                 left_train, left_val, left_test, im_to_vis = imgs, vis = 1, plt_loss = 1, plt_model=0)

# predict and visualize on test set
sim.test()
y_pred = sim.y_pred

# =============================================================================
#  Extract pairs
# =============================================================================
os.chdir(os.path.join('.','comparison with the web'))

nums = range(1000,1100)
# Take out zeros and ones labeled
states = np.where(~(np.logical_or(l_te[nums] == 0, l_te[nums] == 1)))
nums_updated = [nums[i] for i in states[0]]

for num in nums_updated:
    img1 = right_test[num,:,:]
    plt.imshow(img1); plt.axis('off')
    plt.savefig(str(num) + 'R.png')
    plt.show()
    
for num in nums_updated:
    img2 = left_test[num,:,:,:]
    plt.imshow(img2); plt.axis('off')
    plt.savefig(str(num) + 'L.png')
    plt.show()

y_true = np.ndarray.round(l_te[nums_updated],3)
y_pr = np.ndarray.round(y_pred[nums_updated],3)

# =============================================================================
# Compare results
# =============================================================================

os.chdir(os.path.join('.','comparison with the web'))
import pandas as pd

xl = pd.read_excel(os.path.join('.','web.xlsx'))

y_true = xl['True Value'][:100]
y_pred = xl['Predicted'][:100]
y_web = xl['Residual'][:100]

# Our results
model_r2 = KPIs(y_true, y_pred, 'model')
# Web results
web_r2 = KPIs(y_true, y_web, 'web')

# plot dependencies
plt.scatter(abs(1-y_true), abs(1-y_pred), s=10, facecolors='none', edgecolors='black')
plt.scatter(abs(1-y_true), abs(1-y_web), s=10, facecolors='none', edgecolors='red')
plt.xlabel('True Target')
plt.ylabel('Predicted Target')
plt.legend(['Model $R^2$ = ' + str(np.round(model_r2,decimals=2)),'WEB $R^2$ = ' + str(np.round(web_r2,decimals=2))])
plt.savefig('Evaluation')
plt.show(); plt.close()


