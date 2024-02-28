# %% imports and helpers
import pandas as pd
import sys,os
from pathlib import Path
from metrics import calculate_metrics
import pickle
import numpy as np
from epoch import Epoch

base_dir = Path('/home/georg/code/cosyne/spike-sorting-benchmark-sandbox')
os.chdir(base_dir)
GaellesQDf = pd.read_csv("QM_table.csv", delimiter=',')

data_dir = Path("/media/georg/Seagate Expansion Drive/ibl")
pids = os.listdir(data_dir)

# filter empty
bad_pids = []
for pid in pids:
    if len(os.listdir(data_dir / pid)) != 3:
        bad_pids.append(pid)

pids = [pid for pid in pids if not pid in bad_pids]

def get_data(folder):
    try:
        with open(folder / "spikes.pckl", 'rb') as fH:
            spikes = pickle.load(fH)
        with open(folder / "channels.pckl", 'rb') as fH:
            channels = pickle.load(fH)
        with open(folder / "clusters.pckl", 'rb') as fH:
            clusters = pickle.load(fH)
    except:
        print("problem fetching data for pid: %s" % folder.stem)

    return spikes, channels, clusters


# %% calculating the metrics on the datasets

# some params
# params = dict(include_pcs=False, isi_threshold=0.002,  min_isi=0.0001, tbin_sec=1)
# dims = ['x','y']
# dims = ['x','y','z']
# epoch = Epoch("test", 100, 160)

PredsDf = []
from tqdm import tqdm
for pid in tqdm(pids[:2]):
    folder = data_dir / pid
    D = dict(pid=pid)
    try:
        spikes, channels, clusters = get_data(folder)
        # channel_pos = np.vstack([channels[d] for d in dims]).T
        # MetricsDf = calculate_metrics(spikes['times'], spikes['clusters'], spikes['clusters'], spikes['amps'], None, channel_pos, None, None, None, params, epochs = [epoch])
        
        # quantile based
        quantiles = np.linspace(0,100,5).astype('int32')
        keys = clusters.metrics.keys()[1:]
        for key in keys:
            values = clusters.metrics[key]
            qvals = np.percentile(values, quantiles)
            for v,q in zip(qvals, quantiles):
                D["%s_%i" % (key, q)] = v

        # extra descriptors
        D["label_0.3"] = np.sum(clusters['metrics']['label'] == 1/3)
        D["label_0.6"] = np.sum(clusters['metrics']['label'] == 2/3)
        D["label_1.0"] = np.sum(clusters['metrics']['label'] == 3/3)

        D['n_clusters'] = clusters.metrics.shape[0]
        D['n_spikes'] = spikes.times.shape[0]

        PredsDf.append(pd.Series(D))
    except:
        print("problem with %s" % pid)

PredsDf = pd.DataFrame(PredsDf)
# %%


# %%store
PredsDf.to_csv(base_dir / "PredsDf.csv")

# %% read
PredsDf = pd.read_csv(base_dir / "PredsDf.csv")

# %% find subset
label_cols = ['AP_1', 'AP_1', 'AP_1', 'LF_1', 'LF_2', 'LF_3']
good_ix = np.sum(pd.isna(GaellesQDf[label_cols]),axis=1) == 0
GaellesQDf = GaellesQDf.loc[good_ix]

# %%
Df = pd.merge(GaellesQDf, PredsDf, on='pid')

# %% turning Gaelles labels into an classifier lables
n_pass = np.sum(Df[label_cols] == "PASS", axis=1)

import matplotlib.pyplot as plt
fig, axes = plt.subplots()
axes.hist(n_pass,bins=np.arange(6))

# %%
from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split

# PRE
y = n_pass.values
# y = n_pass < 4

# control: label permute
ix = np.arange(y.shape[0])
np.random.shuffle(ix)
y_shuff = y[ix]

# Scores = []
# Scores_shuff = []
pred_cols = PredsDf.columns[1:]
X = Df[pred_cols].values

# center
X = X - np.average(X,axis=0)[np.newaxis,:]
# X = X / np.std(X, axis=0)[np.newaxis,:]

plt.matshow(X)

# %%
clf = svm.SVC(kernel='rbf', C=1, random_state=22)
scores = cross_val_score(clf, X, y, cv=5)
scores_shuff = cross_val_score(clf, X, y_shuff, cv=5)

print(np.average(scores), np.std(scores))
print(np.average(scores_shuff), np.std(scores_shuff))
# %%
