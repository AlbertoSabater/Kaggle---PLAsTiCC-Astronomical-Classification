#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np


DATA_DIR = './data/'


# %%

dataset = pd.read_csv(DATA_DIR+'training_set.csv.zip')


# %%

groups_obj = dataset.groupby('object_id')
# Print num object observation histogram
plt.hist([ len(g) for i,g in groups_obj ], bins='auto');


# %%

dataset['mjd_short'] = dataset['mjd'].apply(lambda x: round(x*10000//10000))
print('mjd_short calculated')
group_obs = dataset.groupby(['object_id', 'mjd_short'])
print('dataset grouped')

def group_process(name, obs):
	g_proc = {'object_id': name[0],
		      'mjd_short': name[1]
			  }
	
	# Initialize passband parameters
	for i in range(6):
		g_proc['mjd_{}'.format(i)] = 0.0
		g_proc['flux_{}'.format(i)] = 0.0
		g_proc['flux_err_{}'.format(i)] = 0.0
		g_proc['detected_{}'.format(i)] = 0.0
	
	# Fill passband parameters
	for i,r in obs.iterrows():
		g_proc['mjd_{}'.format(int(r['passband']))] = r['mjd']
		g_proc['flux_{}'.format(int(r['passband']))] = r['flux']
		g_proc['flux_err_{}'.format(int(r['passband']))] = r['flux_err']
		g_proc['detected_{}'.format(int(r['passband']))] = r['detected']
	
#	observations.append(g_proc)
	return g_proc
	
observations = Parallel(n_jobs=8)(delayed(group_process)(name, obs) for name, obs in tqdm(group_obs, total=len(group_obs)))
dataset_proc = pd.DataFrame(observations)
dataset_proc.to_pickle(DATA_DIR + 'processed/observations_1.pckl')


# %%
	
dataset_proc = pd.read_pickle(DATA_DIR + 'processed/observations_1.pckl')
dataset_proc['passband_diff'] = 0


def process_time_columns(dataset, obj_id):
	# Add time difference between two single observations
	a = dataset[dataset.object_id == obj_id]['mjd_short'].values
	diff = [0] + (a[1:] - a[:-1]).tolist()
	dataset.loc[dataset.object_id == obj_id, 'passband_diff'] = diff
	
	n_obs = 0
	# Add number of observation
	for i,r in dataset.loc[dataset_proc.object_id == obj_id].iterrows():
		if r.passband_diff > 100: n_obs += 1
		dataset.loc[i, 'n_obs'] = n_obs
		
	return dataset

obj_ids = dataset_proc.object_id.drop_duplicates()
dataset_proc = Parallel(n_jobs=8)(delayed(process_time_columns)(dataset_proc, obj_id) 
							for obj_id in tqdm(obj_ids, total=len(obj_ids)))


# %%

dataset_proc = pd.read_pickle(DATA_DIR + 'processed/observations_1.pckl')
dataset_proc['passband_diff'] = 0

# Add time difference between complete observations
a = dataset_proc['mjd_short'].values
diff = [0] + (a[1:] - a[:-1]).tolist()
dataset_proc['passband_diff'] = diff
dataset_proc.loc[dataset_proc.passband_diff < 0, 'passband_diff'] = 0


n_obs = 0
n_obs_array = []
for pbd in dataset_proc['passband_diff']:
	if pbd > 100: n_obs += 1
	elif pbd == 0: n_obs = 0
	n_obs_array.append(n_obs)

dataset_proc['n_obs'] = n_obs_array

dataset_proc.to_pickle(DATA_DIR + 'processed/observations_2.pckl')


# %%

obj_id = np.random.choice(dataset_proc.object_id.drop_duplicates())
g = dataset_proc[dataset_proc.object_id==obj_id]

fig = plt.figure(figsize=(11,6));
#plt.plot();

for s_ind, i in enumerate(g.n_obs.drop_duplicates()):
	
	print()
	
	data = g[g.n_obs==i]
	
	ax = plt.subplot('22{}'.format(s_ind+1));
	plt.plot(data.mjd_short, data.flux_1);
	plt.plot(data.mjd_short, data.flux_2);
	plt.plot(data.mjd_short, data.flux_3);
	plt.plot(data.mjd_short, data.flux_4);
	plt.plot(data.mjd_short, data.flux_5);
	ax.set_title('{} - {}'.format(s_ind, len(data)))
	
fig.suptitle(obj_id);
fig.tight_layout()

# %%

# passband_diff histogram
plt.hist([dataset_proc.passband_diff], bins='auto');


# %%

# Number of observations histogram
lens = [ max(g.n_obs)+1 for name,g in dataset_proc.groupby('object_id')]
plt.hist(lens, bins='auto');

for l in set(lens):
	l_lens = len([ li for li in lens if li==l])
	print('{}: {:.2f}% - {}'.format(l, l_lens/len(lens), l_lens))
	
	
# %%
	
# Length of each observation
lens = [ len(g) for name,g in dataset_proc.groupby(['object_id', 'n_obs'])]
plt.hist(lens, bins='auto');



