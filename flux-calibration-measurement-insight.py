'''

This script is just investigating what I can do to the input values
to change the way the (R-2.98)/dR diagnostic measurement comes out.

For reference, exploring this because in my first pass of the actual
data, it seems like there's a negative skew for some of the spaxels in 
the SGAS1723 galaxy.
--- first thought is that it may be a continuum-fitting error; but 
    before I dive into that, I wanted to familiarize myself with what
    this measurement (and variations on it) actually implies.


'''

__author__ = 'Taylor Hutchison'
__email__ = 'astro.hutchison@gmail.com'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from scipy.stats import norm
import sys

np.random.seed(seed=88) # fixing the random seed



# setting up a range of R, dR values
# ----------------------------------
R_changing = np.linspace(-10,10,100)
dR_changing = np.linspace(1e-3,1,100)
R_stable = np.ones(100)*2.98 + np.random.normal(0,0.1,100) # adding some noise
dR_stable = np.ones(100)*0.1 + np.random.normal(0,0.01,100) # close to median
R_skewedLow = np.ones(100)*2.7 + np.random.normal(0,0.1,100) # adding some noise
dR_skewedHigh = np.ones(100)*0.5 + np.random.normal(0,0.1,100) # adding some noise

def measure(R,dR):
    return (R-2.98)/dR


ratio_name = '$( R_{OIII} - 2.98 )\,/\,dR_{OIII}$'



# "STABLE" VERSION OF MEASUREMENT
# --------------------------------

# want we ideally want to see, where the line flux ratio
# of the two [OIII] lines is around 2.98

plt.figure(figsize=(6.5,5))

plt.hist(measure(R_stable,dR_stable), bins=14, alpha=0.5, 
         density=True, label='stable R & dR'+'\nwhere R~2.98,'
                                     +'\n           dR~0.1')

# plotting normal dist
x = np.linspace(-5,5,200)
plt.plot(x,norm.pdf(x,0,1)*1.5,lw=2,color='k',label='unit gaussian')

plt.axvline(measure(2.98,0.1),color='k',ls=':',zorder=0)
plt.legend(fontsize=14)
plt.xlim(-5,5)

plt.gca().set_yticklabels([])
plt.xlabel(ratio_name)

plt.tight_layout()
# plt.savefig('plots-data/flux-calibration/measurement-insight-stableVals.pdf')
plt.show()
plt.close('all')
print(end='\n\n')


# sys.exit(0) # un-comment to prevent code below from running



# VARIATIONS
# ----------

# running through variations on equation parameters, compared to stable solution
# using all the gridspecs yay

plt.figure(figsize=(12,14))
gs = gridspec.GridSpec(3,1,height_ratios=[1,1,1],hspace=0.22)

gs1 = gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=gs[0],
							width_ratios=[1,1],wspace=0.04)
gs2 = gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=gs[1],
							width_ratios=[1,1],wspace=0.04)
gs3 = gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=gs[2],
							width_ratios=[1,1],wspace=0.04)

subpanels = [gs1,gs2,gs3]

count_subpanel = 0 # 0, 1, or 2
count_subplot = 0 # 0 or 1

comparisons = ['stable R, \nchanging dR',
               'changing R, \nstable dR',
               'changing R, \nchanging dR',
               'skewed low R, \nstable dR',
               'skewed low R, \nchanging dR',
               'skewed low R, \nskewed high dR']

comparisons_data = [measure(R_stable,dR_changing),
                    measure(R_changing,dR_stable),
                    measure(R_changing,dR_changing),
                    measure(R_skewedLow,dR_stable),
                    measure(R_skewedLow,dR_changing),
                    measure(R_skewedLow,dR_skewedHigh)]


# specifying range for stable data, for legend label
stable = measure(R_stable,dR_stable)
stable_range = [round(np.nanmin(stable),2),round(np.nanmax(stable),2)]
stable_label = 'stable R & dR' + '\n[$%s:%s$]'%(stable_range[0],stable_range[1])


# manually getting the bin edges this time (yes it's ugly)
flattened = np.asarray([item for sublist in comparisons_data for item in sublist])
flattened = flattened[(flattened>-5)&(flattened<5)] # clipping out data outside range
bins=np.histogram(np.hstack((stable,flattened)), bins=20)[1] #get the bin


# plotting
for i in range(6):
    subpanel = subpanels[count_subpanel]
    ax = plt.subplot(subpanel[count_subplot])
    
    # getting range of dataset
    data_range = [round(np.nanmin(comparisons_data[i]),2),
                  round(np.nanmax(comparisons_data[i]),2)]
    
    label = comparisons[i] + '\n[$%s:%s$]'%(data_range[0],data_range[1])
    
    
    ax.hist(stable, bins=bins, alpha=0.5, label=stable_label)
    ax.hist(comparisons_data[i], bins=bins, alpha=0.5, 
            label=label, hatch='/', color=f'C{i+1}')


    ax.legend(fontsize=11)

    ax.set_ylim(0,30)
    ax.set_xlim(-5,5)
    ax.set_yticklabels([])
    ax.set_xlabel(ratio_name)
    
    
    # setting counters
    if i%2 == 1:
        count_subplot = 0
        count_subpanel += 1
    else:
        count_subplot += 1



plt.tight_layout()
# plt.savefig('plots-data/flux-calibration/measurement-insight-changingVals.pdf')
plt.show()
plt.close('all')
        



