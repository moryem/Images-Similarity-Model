"""
Created on Wed Oct 24 14:00:25 2018

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


# Read the data
data_path = os.path.normpath(os.getcwd() + os.sep + os.pardir)
root = os.path.join(data_path,'cifar-10-batches-py')
right_train, right_val, right_test, left_train, left_val, left_test, y_tr, y_val, y_test, l_tr, l_val, l_te = data_prep_perc(root)

#%%
# Define hyper-parameters
bs = 32; ep = 1000; dp = 0.55; lr = 5e-5
# Choose images to visualize
imgs = [1, 20, 4, 15, 8011, 8060, 8033, 344]

sim = similarity_net(dp, bs, ep, lr, l_tr, l_val, l_te, right_train, right_val, right_test,
                 left_train, left_val, left_test, im_to_vis = imgs, vis = 1, plt_loss = 0, plt_model=0)

# train the model
sim.train()

# test the model
sim.test()
y_pred = sim.y_pred

#%%
# Calculate KPIs and plot dependencies
# mae
mae_value = mae(l_te, y_pred)
print('mae = ' + str(mae_value))
# mse
mse_value = mse(l_te, y_pred)
print('mse = ' + str(mse_value))
# R2
r2 = r2_score(l_te, y_pred)
print('R2 = ' + str(r2))

# plot dependencies
plt.scatter(abs(1-l_te), abs(1-y_pred), s=10, facecolors='none', edgecolors='black')
plt.xlabel('True Target')
plt.ylabel('Predicted Target')
plt.legend(['$R^2$ = ' + str(np.round(r2,decimals=2))])
plt.savefig('Evaluation')
plt.show(); plt.close()

