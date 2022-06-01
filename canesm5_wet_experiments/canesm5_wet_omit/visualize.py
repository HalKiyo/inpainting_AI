import os
import pickle
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from range_selection import range_selection
from jmacmap import jmacmap

def main():
    root = '/docker/home/hasegawa/docker-gpu/reconstructionAI/'\
           'canesm5_wet_experiments/canesm5_wet_omit'

    vname = 'valid/valid700000.npy'
    outname = 'data/output600000.npy'
    output_list = ['input', 'mask', 'output', 'output_comp', 'gt']

    vpath = os.path.join(root, vname)
    outpath = os.path.join(root, outname)

    val = np.load(vpath)
    output = np.load(outpath)


    show(val[0,4,:,:])
    #val_show(val)
    #multi_show(output, output_list)

def read(name):
    """ Detail is shown in data_docker-conda/preparation/omit.py """
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data

def val_show(dt):
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.imshow(dt[3000,2,:,:])
    plt.show()

def show(img):
    vmin, vmax = 0, 0.0004
    grids = 20
    cname = jmacmap()
    interval = (vmax - vmin)/grids
    ticks = np.round( np.arange(vmin, vmax+interval, interval), 6 )
    cmap = plt.get_cmap(cname, grids)
    proj = ccrs.PlateCarree( central_longitude = 180 )

    """ show val_data, input, gt, and output """
    # img_extent & norm
    proj = ccrs.PlateCarree(central_longitude = 180)
    Range_selection = range_selection()
    rain_omit, img_extent, omit_extent = Range_selection.execute_selection(
                                            Range_selection.pr,
                                            -55.0, 57.5, 0, 360,
                                            Range_selection.lat, Range_selection.lat_bnds,
                                            Range_selection.lon, Range_selection.lon_bnds )

    # figure object

    fig = plt.figure()
    ax = plt.subplot(projection = proj)

    ax.coastlines(resolution='50m', lw=0.5)
    ax.gridlines(xlocs = mticker.MultipleLocator(90),
                 ylocs = mticker.MultipleLocator(45),
                 linestyle = '-',
                 color = 'gray')

    mat = ax.matshow( img,
                      cmap=cmap,
                      origin='lower',
                      extent=img_extent,
                      transform=proj,
                      vmin=vmin - interval/2.0,
                      vmax=vmax + interval/2.0 )
    cbar = fig.colorbar( mat,
                         ax=ax,
                         orientation='horizontal' )
    cbar.set_ticks(ticks[::1])
    cbar.ax.set_xticklabels(ticks[::1], rotation=-90)

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

            tp = ax.imshow( imgs[i],
                            cmap=jmacmap() ,
                            origin='lower',
                            extent=img_extent,
                            transform=proj,
                            vmin=0,
                            vmax=0.0004 )

            ax.set_title(imgs_label[i])
            fig.colorbar(tp, ax=ax, orientation='horizontal')

    plt.show()


if __name__ == '__main__':
    main()
