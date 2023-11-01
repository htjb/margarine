import numpy as np
import matplotlib.pyplot as plt
from anesthetic import read_chains
from anesthetic.samples import MCMCSamples
from margarine.maf import MAF

import pandas as pd

# from margarine.clustered import clusterMAF as MAF
# load chains
names = [i for i in range(5)]
samples = read_chains('chains/gaussian', columns=names)
# df=pd.read_csv('chains/gaussian_low.txt', delim_whitespace=True, header=None)
# weights=df.to_numpy()[...,0]
# theta=df.to_numpy()[...,2:]
theta = samples[names].values
weights = samples.get_weights()

flow = MAF(theta, weights=weights)
flow.train(10000, early_stop=True)
fs = flow.sample(5000)
flow.save('maf.pkl')

fs = MCMCSamples(data=fs, columns=names)
axes = samples.plot_2d(names[:5])
fs.plot_2d(axes, color='C1', alpha=0.5)
# plt.show()
plt.savefig("flow.pdf")