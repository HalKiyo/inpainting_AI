import os
import pickle
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from range_selection import range_selection

def main():
    root = '/docker/home/hasegawa/docker-gpu/reconstructionAI/'\
           'canesm5_wet_experiments/canesm5_wet_omit_anom/data/'

    vname = 'canesm5_wet_omit_anom_valid.npy'
    outname = 'output700000.npy'
    output_list = ['input', 'mask', 'output', 'output_comp', 'gt']

    outpath = os.path.join(root, outname)
    vpath = os.path.join(root, vname)

    output = np.load(outpath)
    val = np.load(vpath)

    multi_show(output, output_list)

def read(name):
    """ Detail is shown in data_docker-conda/preparation/omit.py """
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data

def show(img):
    """ show val_data, input, gt, and output """
    # img_extent & norm
    Range_selection = range_selection()
    rain_omit, img_extent, omit_extent = Range_selection.execute_selection(
                                            Range_selection.pr,
                                            -55.0, 57.5, 0, 360,
                                            Range_selection.lat, Range_selection.lat_bnds,
                                            Range_selection.lon, Range_selection.lon_bnds )
    norm = mcolors.TwoSlopeNorm( vmin=-0.0001,
                                 vmax=0.0001,
                                 vcenter=0 )

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

def multi_show(imgs, imgs_label):
    """ Compare val_data, input, gt and output """
    # figure indeces
    proj = ccrs.PlateCarree(central_longitude = 180)
    Range_selection = range_selection()
    rain_omit, img_extent, omit_extent = Range_selection.execute_selection(
                                            Range_selection.pr,
                                            -55.0, 57.5, 0, 360,
                                            Range_selection.lat, Range_selection.lat_bnds,
                                            Range_selection.lon, Range_selection.lon_bnds )

    # figure layout
    nrows = 1
    ncols = len(imgs)
    pos_1 = nrows*100 + ncols*10 + 1
    pos = [i for i in range(pos_1, pos_1 + nrows*ncols)]

    # figure object
    fig = plt.figure()

    for i, num in enumerate(pos[:ncols]):
        if i == 1:
            pass

        else:
            ax = plt.subplot(num, projection=proj)

            ax.coastlines(resolution='50m', lw=0.5)
            ax.gridlines(xlocs = mticker.MultipleLocator(90),
                         ylocs = mticker.MultipleLocator(45),
                         linestyle = '-',
                         color = 'gray')

            norm = mcolors.TwoSlopeNorm( vmin=-0.0001,
                                         vmax=0.0001,
                                         vcenter=0 )

            tp = ax.imshow(imgs[i], cmap='RdBu_r' ,origin='lower',
                           extent=img_extent, transform=proj, norm=norm)

            ax.set_title(imgs_label[i])
            fig.colorbar(tp, ax=ax, orientation='horizontal')

    plt.show()


if __name__ == '__main__':
    main()
