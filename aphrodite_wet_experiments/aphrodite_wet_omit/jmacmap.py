import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def jmacmap():
    jmacolors=np.array(
       [[242,242,242,1],#white
        [160,210,255,1],
        [33 ,140,255,1],
        [0  ,65 ,255,1],
        [250,245,0,1],
        [255,153,0,1],
        [255,40,0,1],
        [180,0,104,1]],dtype=np.float)

    jmacolors[:,:3] /=256
    jmacmap=ListedColormap(jmacolors)
    jmacmap2=LinearSegmentedColormap.from_list("jmacmap2",colors=jmacolors)

    return jmacmap2
