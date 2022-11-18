'''
These functions & class are used to make a color bar (and 2D coloring) that 
uses a specific colormap but then highlights a certain value & range with 
a different color.

'''

__author__ = 'Taylor Hutchison'
__email__ = 'astro.hutchison@gmail.com'


import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def make_colormap(cen,dmin,dmax,size):
    # coming up with a range that helps us figure out where the color is
    t = np.linspace(dmin,dmax,size)
    s = np.linspace(0,1,size)
    n = size*10.

    x = cen/(dmax-dmin)
    print('x: %s, Min: %s, Max: %s\n'%(x,dmin,dmax))

    lower = plt.cm.inferno(np.linspace(0,x-0.01*x,int(n*round(x-0.01*x,2))))
    focus = plt.cm.rainbow(np.ones(3)*0.3) # to make it bright blue
    upper = plt.cm.inferno(np.linspace(x+0.01*x,1,int(n*round(1-(x+0.01*x),2))))
    colors = np.vstack((lower, focus, upper))
    tmap = matplotlib.colors.LinearSegment('taylor', colors)
    return tmap

def make_colormap_range(cen,width,dmin,dmax,size,span=70):
    # coming up with a range that helps us figure out where the color is
    t = np.linspace(dmin,dmax,size)
    s = np.linspace(0,1,size)
    n = size*10.

    x1 = (cen-width)/(dmax-dmin)
    x2 = (cen+width*0.3)/(dmax-dmin) # plots skewed to the higher end?
    print('x1: %s, x2: %s, Min: %s, Max: %s\n'%(x1,x2,dmin,dmax))

    lower = plt.cm.Blues(np.linspace(0,x1-0.01*x1,int(n*round(x1-0.01*x1,2))))
    focus = plt.cm.rainbow(np.ones(span)*0.9) # to make it bright blue
    upper = plt.cm.Blues(np.linspace(x2+0.01*x2,1,int(n*round(1-(x2+0.01*x2),2))))
    colors = np.vstack((lower, focus, upper))
    tmap = matplotlib.colors.LinearSegmentedColormap.from_list('taylor', colors)
    return tmap

class FixPointNormalize(matplotlib.colors.Normalize):
    """ 
    Inspired by https://stackoverflow.com/questions/20144529/shifted-colorbar-matplotlib
    Subclassing Normalize to obtain a colormap with a fixpoint 
    somewhere in the middle of the colormap.

    This may be useful for a `terrain` map, to set the "sea level" 
    to a color in the blue/turquise range as shown in example:
    https://stackoverflow.com/questions/40895021/python-equivalent-for-matlabs-demcmap-elevation-appropriate-colormap
    """
    def __init__(self, vmin=None, vmax=None, fixme=5, fixhere=0.26, clip=False):
        # fixme is the fix point of the colormap (in data units)
        self.fixme = fixme
        # fixhere is the color value in the range [0,1] that should represent fixme
        self.fixhere = fixhere
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.fixme, self.vmax], [0, self.fixhere, 1]
        return np.ma.masked_array(np.interp(value, x, y))    