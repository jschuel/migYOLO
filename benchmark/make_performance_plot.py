import os
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 16})

timestamp_dir = 'timestamps/'

if len(os.listdir(timestamp_dir)) == 0:
    raise ValueError("No files in directory %s, either choose a different directory, or run the pipeline to create timestamp files"%(timestamp_dir))

timestamps = np.array([float(np.loadtxt(timestamp_dir+fi)) for fi in sorted(os.listdir(timestamp_dir))])

elapsed_time = timestamps-timestamps[0]
ims_processed = np.array([200*i for i in range(len(timestamps))])

fig, ax = plt.subplots(figsize = (9,6))

ax.plot(ims_processed,ims_processed/elapsed_time,'o')
xticks = np.array([5000 * i for i in range(0,21)])
ax.set_xticks(xticks)
ax.set_xticklabels((xticks/1000).astype('int'))
ax.set_xlabel(r'# Processed images [$\times 1000$]')
ax.set_ylabel('Processing speed [fps]')
plt.tight_layout()
plt.savefig('benchmark.jpg')

