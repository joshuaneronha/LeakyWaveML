
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 2

onemm = 20*np.log10(np.loadtxt('Baseline/lambda1mm.csv',delimiter=','))
twomm = 20*np.log10(np.loadtxt('Baseline/lambda2mm.csv',delimiter=','))
fourmm = 20*np.log10(np.loadtxt('Baseline/lambda4mm.csv',delimiter=','))
eightmm = 20*np.log10(np.loadtxt('Baseline/lambda8mm.csv',delimiter=','))

fig, ax = plt.subplots(figsize=(4,3))
ax.plot(onemm[:,1],color='#c7e9b4')
ax.plot(twomm[:,1],color='#7fcdbb')
ax.plot(fourmm[:,1],color='#1d91c0')
ax.plot(eightmm[:,1],color='#0c2c84')
ax.legend(['$\lambda$ = 1mm','$\lambda$ = 2mm','$\lambda$ = 4mm','$\lambda$ = 8mm'])
plt.xlim([0, 180])
# plt.savefig('theoreticallambda.pdf')
# %% codecell
!pip install tensorflow
# %% codecell
