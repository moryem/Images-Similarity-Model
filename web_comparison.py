# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 21:30:12 2019

@author: Dror
"""

import os
from utils import data_prep_perc
from DeepModel import similarity_net
import matplotlib.pyplot as plt
import numpy as np


# Read the data
data_path = os.path.normpath(os.getcwd() + os.sep + os.pardir)
root = os.path.join(data_path,'cifar-10-batches-py')
right_train, right_val, right_test, left_train, left_val, left_test, y_tr, y_val, y_test, l_tr, l_val, l_te = data_prep_perc(root)

# Define hyper-parameters
bs = 32; ep = 1000; dp = 0.55; lr = 5e-5
imgs = [1, 20, 4, 15, 8011, 8060, 8033, 344]

reg = similarity_net(dp, bs, ep, lr, l_tr, l_val, l_te, right_train, right_val, right_test,
                 left_train, left_val, left_test, im_to_vis = imgs, vis = 1, plt_loss = 1, plt_mat=0, plt_model=0)

# predict and visualize on test set
reg.test()
y_pred = reg.y_pred

#%%
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

#%%
# Compare results

os.chdir(os.path.join('.','comparison with the web'))

from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
import pandas as pd

xl = pd.read_excel(os.path.join('.','web.xlsx'))

y_true = xl['True Value'][:100]
y_pred = xl['Predicted'][:100]
y_web = xl['Residual'][:100]

# Our results
# mae
model_mae_value = mae(y_true, y_pred)
print('model mae = ' + str(model_mae_value))
# mse
model_mse_value = mse(y_true, y_pred)
print('model mse = ' + str(model_mse_value))
# R2
model_r2 = r2_score(y_true, y_pred)
print('model R2 = ' + str(model_r2))

# Web results
# mae
web_mae_value = mae(y_true, y_web)
print('web mae = ' + str(web_mae_value))
# mse
web_mse_value = mse(y_true, y_web)
print('web mse = ' + str(web_mse_value))
# R2
web_r2 = r2_score(y_true,y_web)
print('web R2 = ' + str(web_r2))

# plot dependencies
plt.scatter(abs(1-y_true), abs(1-y_pred), s=10, facecolors='none', edgecolors='black')
plt.scatter(abs(1-y_true), abs(1-y_web), s=10, facecolors='none', edgecolors='red')
plt.xlabel('True Target')
plt.ylabel('Predicted Target')
plt.legend(['Model $R^2$ = ' + str(np.round(model_r2,decimals=2)),'WEB $R^2$ = ' + str(np.round(web_r2,decimals=2))])
plt.savefig('Evaluation')
plt.show(); plt.close()

