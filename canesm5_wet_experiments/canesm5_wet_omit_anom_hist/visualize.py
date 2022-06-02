import os
import pickle
import numpy as np
import netCDF4 as nc
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from jmacmap import jmacmap
from range_selection import range_selection

def main():
    root = '/docker/home/hasegawa/docker-gpu/reconstructionAI/'\
           'canesm5_wet_experiments/canesm5_wet_omit_anom_hist/data/'

    vname = 'canesm5_wet_omit_anom_hist_valid.npy'
    outname = 'output1000000.npy'

    outpath = os.path.join(root, outname)
    vpath = os.path.join(root, vname)

    output = np.load(outpath)
    val = np.load(vpath)

    output_list = ['input', 'mask', 'output', 'output_comp', 'gt']
    index = 4
    print(output_list[index])
    show(output[index])

def read(name):
    """ Detail is shown in data_docker-conda/preparation/omit.py """
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data

def show(img):
    """ show val_data, input, gt, and output """
    # img_extent & norm
    path = '/docker/home/hasegawa'\
           '/docker-gpu/reconstructionAI/canesm5_wet_experiments'\
           '/canesm5_wet_omit_anom_hist'\
           '/data/canesm5_wet_omit_anom_hist.pickle'
    dct = read(path)

    anom = np.array(dct['rain_anom'])
    hist_rain, bin_edges = np.histogram(anom, bins=256)
    zero_center = np.where(bin_edges >= 0)[0][0]

    Range_selection = range_selection()
    rain_omit, img_extent, omit_extent = Range_selection.execute_selection(
        Range_selection.pr,
        -55.0, 57.5, 0, 360,
        Range_selection.lat, Range_selection.lat_bnds,
        Range_selection.lon, Range_selection.lon_bnds)
    norm = mcolors.TwoSlopeNorm(
        vmin=zero_center-30, vmax=zero_center+30, vcenter=zero_center)

    # figure object
    fig = plt.figure()
    proj = ccrs.PlateCarree(central_longitude = 180)
    ax = plt.subplot(projection = proj)

    ax.coastlines(resolution='50m', lw=0.5)
    ax.gridlines(xlocs = mticker.MultipleLocator(90),
                 ylocs = mticker.MultipleLocator(45),
                 linestyle = '-',
                 color = 'gray')

    tp = ax.imshow(img, cmap='RdBu_r' ,origin='lower',
                   extent=img_extent, transform=proj, norm=norm)

    fig.colorbar(tp, ax=ax, orientation='horizontal')
    plt.show()

if __name__ == '__main__':
    main()
