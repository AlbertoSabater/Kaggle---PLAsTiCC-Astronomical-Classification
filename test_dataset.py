#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np


DATA_DIR = './data/'
DATA_DIR = ''


# %%

dataset = pd.read_csv(DATA_DIR+'training_set.csv.zip')


# %%

groups_obj = dataset.groupby('object_id')
# Print num object observation histogram
plt.hist([ len(g) for i,g in groups_obj ], bins='auto');


# %%

dataset['mjd_short'] = dataset['mjd'].apply(lambda x: round(x*10000//10000))
print('mjd_short calculated')
group_obs = dataset[:5000].groupby(['object_id', 'mjd_short'])
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


# %%
	
dataset_proc = pd.DataFrame(observations)
dataset_proc['passband_diff'] = 0


# %%

obj_ids = dataset_proc.object_id.drop_duplicates()
for obj_id in tqdm(obj_ids, total=len(obj_ids)):
	a = dataset_proc[dataset_proc.object_id == obj_id]['mjd_short'].values
	diff = [0] + (a[1:] - a[:-1]).tolist()
	dataset_proc.loc[dataset_proc.object_id == obj_id, 'passband_diff'] = diff
	
	n_obs = 0
	for i,r in dataset_proc.loc[dataset_proc.object_id == obj_id].iterrows():
		dataset_proc.loc[i-1, 'n_obs'] = n_obs
		if r.passband_diff > 100: n_obs += 1
	

# %%
	
# TODO: Añadir número de observación partiendo las obsrevaciones por diff>~100


# %%

obj_id = np.random.choice(dataset_proc.object_id.drop_duplicates())
print(obj_id)
g = dataset_proc[dataset_proc.object_id==obj_id]

plt.figure(figsize=(20,5));
plt.plot();

for i in g.n_obs.drop_duplicates():
	
	data = g[g.n_obs==i]
	
	plt.plot(data.mjd_short, data.flux_1);
	plt.plot(data.mjd_short, data.flux_2);
	plt.plot(data.mjd_short, data.flux_3);
	plt.plot(data.mjd_short, data.flux_4);
	plt.plot(data.mjd_short, data.flux_5);


# %%

plt.hist([dataset_proc.passband_diff], bins='auto')


