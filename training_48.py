#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math

import tensorflow as tf
import keras
from tensorflow.keras import layers
from alnn import ALNN_GRU
from sklearn.model_selection import KFold
import time as tm
import keras.backend as K
from sklearn.metrics import roc_auc_score as auc_score
from sklearn.metrics import precision_recall_curve,roc_curve,auc
import sklearn.metrics
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter
from utils import binary_focal_loss

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



# In[2]:


prior_hours=48
bound=120

Time_=pd.read_csv(f'data/time_{prior_hours}_padded_imputed.csv')
M_=pd.read_csv(f'data/mask_{prior_hours}_padded_imputed.csv')
X_=pd.read_csv(f'data/values_{prior_hours}_padded_imputed.csv')
y=pd.read_csv(f'data/target_{prior_hours}_padded_imputed.csv')


# In[3]:


Time_=np.array(Time_).reshape(-1,bound,12)
M_=np.array(M_).reshape(-1,bound,12)
X_=np.array(X_).reshape(-1,bound,12)
y=y.values.reshape(-1)
X_.shape,Time_.shape,M_.shape,y.shape


# In[4]:


X_tempo=np.nan_to_num(X_,nan=0)
X_tempo=X_tempo*M_
means=(np.sum(np.sum(X_tempo,axis=0),0)/np.sum(np.sum(M_,axis=0),0))
#Replace CRR missing values by the mode.
means[9]=1
del X_tempo
means


# In[5]:


X_non_aligned=[]
for k in range(Time_.shape[0]):
    all_=[]
    for j in range(Time_.shape[1]):
        tempo=[]
        for l in range(Time_.shape[2]):
            if math.isnan(X_[k][j][l]):
                tempo.append(means[l] )
            else:
                tempo.append(X_[k][j][l])
        all_.append(tempo)
    X_non_aligned.append(all_)
X_non_aligned=np.array(X_non_aligned).reshape(-1,bound,12)   
print(X_non_aligned.shape)


# In[ ]:


Delta_time=[]
for k in range(Time_.shape[0]):
    all_=[]
    for j in range(Time_.shape[1]):
        tempo=[]
        for p in range(Time_.shape[2]):
            if j==0:
                tempo.append(0)
            else:
                if M_[k][j][p]==0:
                    tempo.append(Time_[k][j][p]-Time_[k][j-1][p]+all_[j-1][p])
                else:
                    tempo.append(Time_[k][j][p]-Time_[k][j-1][p])
        all_.append(tempo)
    Delta_time.append(all_)
Delta_time=np.array(Delta_time).reshape(-1,bound,12)   
Delta_time.shape


# In[10]:


X_non_aligned.shape,Time_.shape,M_.shape,Delta_time.shape,y.shape


# In[11]:


kfold = KFold(n_splits=5, shuffle=True)
bc=keras.losses.BinaryCrossentropy(from_logits=False)
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=2, mode='min')
epoch=2
start= tm.perf_counter()
MAEs,aucs,aucpr=[],[],[]


# In[12]:


for train, test in kfold.split(X_non_aligned,y):

    model=ALNN_GRU(prior_hours,0,prior_hours+1)
    model.compile(loss=binary_focal_loss(2.5,.05),optimizer=opt,metrics=["accuracy"],)

    model.fit([X_non_aligned[train],Time_[train],M_[train],Delta_time[train]], y[train],verbose=1,batch_size=100,epochs=epoch)
    loss_test, accuracy_test = model.evaluate([X_non_aligned[test],Time_[test],M_[test],Delta_time[test]],y[test],verbose=1,batch_size=100,callbacks=[callback])

    y_probas = model.predict([X_non_aligned[test],Time_[test],M_[test],Delta_time[test]]).ravel()
    fpr,tpr,thresholds=roc_curve(y[test],y_probas)
    aucs.append(auc(fpr,tpr))
    print('AUC',auc(fpr,tpr))
        
    auprc_ = sklearn.metrics.average_precision_score(y[test], y_probas)
    aucpr.append(auprc_)
    print('AUPRC', auprc_)
        
        
finish=tm.perf_counter()
print(f"Finished in {round(finish-start,2)},second(s)")
print(f'AUC: mean{np.round(np.mean(np.array(aucs)),3)},std{np.round(np.std(np.array(aucs)),3)}')
print(f'AUPRC: mean{np.round(np.mean(np.array(aucpr)),3)},std{np.round(np.std(np.array(aucpr)),3)}')


# In[14]:


wrong_hdmid_classified=[]
def trainingFuction(MedicNet,kfold,X_non_aligned,Time_,M_,Delta_time,y,epc=35):
    start= tm.perf_counter()
    MAEs,aucs,aucpr=[],[],[]
    compteur=1;
    
    for train, test in kfold.split(X_non_aligned,y[:,1]):
        
#         w_0=(len(y)/(2*Counter(y[train])[0]))
#         w_1=(len(y)/(2*Counter(y[train])[1]))
        
        model=MedicNet(48,0,49)
        model.compile(loss=binary_focal_loss(),optimizer=opt,metrics=["accuracy"],)
        

        
        model.fit([X_non_aligned[train],Time_[train],M_[train],Delta_time[train]], y[:,1][train],verbose=1,batch_size=100,epochs=epc)
        loss_test, accuracy_test = model.evaluate([X_non_aligned[test],Time_[test],M_[test],Delta_time[test]],y[:,1][test],verbose=1,batch_size=100,callbacks=[callback])
        
#         model.save_weights(f'../../Data/MIMIC3/my_model_{compteur}')
        
#         compteur=compteur+1
        
        y_probas = model.predict([X_non_aligned[test],Time_[test],M_[test],Delta_time[test]]).ravel()
        fpr,tpr,thresholds=roc_curve(y[:,1][test],y_probas)
        aucs.append(auc(fpr,tpr))
        print('AUC',auc(fpr,tpr))
        
        auprc_ = sklearn.metrics.average_precision_score(y[:,1][test], y_probas)
        aucpr.append(auprc_)
        print('AUPRC', auprc_)
        
        #Threshold moving
        gmeans = np.sqrt(tpr * (1-fpr))
        ix = np.argmax(gmeans)
        print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
        
        tps=[]
        for classs,hdmm,prob in zip(y[:,1][test],y[:,0][test],y_probas):
                if (prob>=thresholds[ix]) and (classs==1):
                    tps.append(hdmm)
        
        print(len(tps))
        wrong_hdmid_classified.append(tps)  
        
        
        print('Confusion matrix with the default threshold')
        y_pred = np.where(y_probas>0.5,1,0)
        cm = confusion_matrix(y[:,1][test], y_pred)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(cm)
        ax.grid(False)
        ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
        ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
        ax.set_ylim(1.5, -0.5)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
        plt.show()
        
        
        print('Confusion matrix with the best threshold')
        y_pred = np.where(y_probas>thresholds[ix],1,0)
        cm = confusion_matrix(y[:,1][test], y_pred)
        print(cm)
        print('Specificity= {}'.format(cm[0][0]/(cm[0][0]+cm[0][1])))
        print('Sensitivity= {}'.format(cm[1][1]/(cm[1][0]+cm[1][1])))
        print('Ratio_1 {} Ratio_2 {}'.format((cm[0][0]/(cm[0][0]+cm[0][1]))/(cm[1][1]/(cm[1][0]+cm[1][1])),(cm[1][1]/(cm[1][0]+cm[1][1]))/(cm[0][0]/(cm[0][0]+cm[0][1]))))
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(cm)
        ax.grid(False)
        ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
        ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
        ax.set_ylim(1.5, -0.5)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
        plt.show()
        
    finish=tm.perf_counter()
    print(f"Finished in {round(finish-start,2)},second(s)")
    print(f'AUC: mean{np.round(np.mean(np.array(aucs)),3)},std{np.round(np.std(np.array(aucs)),3)}')
    print(f'AUPRC: mean{np.round(np.mean(np.array(aucpr)),3)},std{np.round(np.std(np.array(aucpr)),3)}')
    print(f'sensitivity ')


# ### focal loss

# In[16]:


trainingFuction(ALNN_GRU,kfold,X_non_aligned,Time_,M_,Delta_time,y)#2.5, 0.05


# In[11]:


np.round(0.184855,3)


# In[15]:


trainingFuction(ALNN_GRU,kfold,X_non_aligned,Time_,M_,Delta_time,y)#2.5, 0.05


# In[18]:


b=wrong_hdmid_classified[0]+wrong_hdmid_classified[1]+wrong_hdmid_classified[2]+wrong_hdmid_classified[3]+wrong_hdmid_classified[4]


# In[19]:


len(Counter(b))


# In[20]:


a=pd.read_csv('pack_a.csv')


# In[23]:


a=np.array(a['0'].unique().tolist(),dtype=np.int64)


# In[34]:


len(list(set(b)))


# In[39]:


len(list(set(a.tolist())))-3043


# In[29]:


len(np.intersect1d(np.array(b),a))


# In[40]:


3838-3043


# <p>wbc</p>

# In[15]:


trainingFuction(ALNN_GRU,kfold,X_non_aligned,Time_,M_,Delta_time,y)


# <p>bc</p>

# In[15]:


trainingFuction(ALNN_GRU,kfold,X_non_aligned,Time_,M_,Delta_time,y)


# In[ ]:





# In[ ]:


trainingFuction(ALNN_GRU,kfold,X_non_aligned,Time_,M_,Delta_time,y)#2.5, 0.05

