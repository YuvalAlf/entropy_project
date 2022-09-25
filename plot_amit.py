import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams['pdf.fonttype'] = 42  # truetype font
mpl.rcParams['ps.fonttype'] = 42  # truetype fontfrom
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['lines.markeredgewidth'] = 0.4


mpl.rcParams.update({'font.size': 9})

xs = np.load('data/Range.npy')
mean_random = np.load('data/Mean_random_values.npy')
best_random = np.load('data/Mean_best_random_values.npy')
max_value = np.load('data/Max_value.npy')

plt.figure(figsize=(3.7, 3))
plt.plot(xs, mean_random, color='r', label='Mean Random')
plt.plot(xs, best_random, color='g', label='Best Random')
plt.plot(xs, max_value, color='b', label='Max Value')
plt.plot((min(xs), max(xs)), (3.82, 3.82), color='gray', label='True Entropy', linestyle='--')

plt.legend(loc='upper right')  #loc='upper left'  #, bbox_to_anchor=(0.06, 0.85))
plt.ylim((3, 5))
plt.xlim((0, 20))
plt.ylabel('Entropy')
plt.savefig('amit.pdf', dpi=300, bbox_inches='tight')



