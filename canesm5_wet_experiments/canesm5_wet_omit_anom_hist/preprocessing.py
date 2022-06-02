import os
import pickle
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from range_selection import range_selection

def main():
    """ Preprocess data for model """

    home = '/docker/home/hasegawa'
    root = 'docker-gpu/reconstructionAI/canesm5_wet_experiments'
    experiment = 'canesm5_wet_omit_anom_hist'
    pcl = 'data/canesm5_wet_omit_anom_hist.pickle'

    path = os.path.join(home, root, experiment, pcl)
    train = os.path.join(home, root, experiment, 'data/canesm5_wet_omit_anom_hist_train.npy')
    valid = os.path.join(home, root, experiment, 'data/canesm5_wet_omit_anom_hist_valid.npy')
    mask = os.path.join(home, root, experiment, 'data/canesm5_wet_omit_anom_hist_mask.npy')

    gen_npy(path, train, valid, save='on')
    canvas = gen_mask(path, mask, save='on')
    maskshow(canvas)

def read(name):
    """ Detail is shown in data_docker-conda/preparation/omit.py """
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data

def gen_npy(path, train, valid, save='off'):
    """ Save ndarray file to /experiment/data/ folder """
    dct = read(path)
    hst = np.array(dct['rain_hist'])

    rng = np.random.default_rng()
    perm = rng.permutation(len(hst))
    train_end = int(0.8 * len(hst))
    valid_end = int(0.2 * len(hst)) + train_end

    train_npy = hst[perm[:train_end]]
    valid_npy = hst[perm[train_end:valid_end]]

    if save == 'on':
        print(save)
        np.save(train, train_npy)
        np.save(valid, valid_npy)

def gen_mask(path, mask, save='off'):
    ind = 0

    dct = read(path)
    hst = np.array(dct['rain_hist'])

    canvas = np.ones(hst[ind].shape)

    Range_selection = range_selection()
    rain_omit, img_extent, omit_extent = Range_selection.execute_selection(
        Range_selection.pr,
        -55.0, 57.5, 0, 360,
        Range_selection.lat, Range_selection.lat_bnds,
        Range_selection.lon, Range_selection.lon_bnds)

    llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon = 5, 25, 95, 110

    ds, img_extent, ds_extent = Range_selection.execute_selection(
        rain_omit,
        llcrnrlat, urcrnrlat,
        llcrnrlon, urcrnrlon,
        Range_selection.lat[      omit_extent[0] : omit_extent[1] ],
        Range_selection.lat_bnds[ omit_extent[0] : omit_extent[1] ],
        Range_selection.lon[      omit_extent[2] : omit_extent[3] ],
        Range_selection.lon_bnds[ omit_extent[2] : omit_extent[3] ])

    print(omit_extent)

    canvas[ds_extent[0] : ds_extent[1], ds_extent[2] : ds_extent[3]] = 0

    if save == 'on':
        print(save)
        np.save(mask, canvas)

    return canvas

def show(path):
    """
    Load data then visualize a sample binary
    dct is sorted by month, then year looks random if ind=0 is selected.
    """
    # arguments object
    ind = 1

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

    hst = dct['rain_hist']
    img = np.array(hst[ind])
    title = f'{dct["year"][ind]}/{dct["month"][ind]}'

    proj = ccrs.PlateCarree(central_longitude = 180)
    norm = mcolors.TwoSlopeNorm(
        vmin=img.min(), vmax=img.max(), vcenter=zero_center)

    # figure object
    fig = plt.figure()
    ax = plt.subplot(projection = proj)

    ax.coastlines(resolution='50m', lw=0.5)
    ax.gridlines(xlocs = mticker.MultipleLocator(90),
                 ylocs = mticker.MultipleLocator(45),
                 linestyle = '-',
                 color = 'gray')

    tp = ax.imshow(img, cmap='RdBu_r' ,origin='lower',
                   extent=img_extent, transform=proj, norm=norm)

    ax.set_title(title)
    fig.colorbar(tp, ax=ax, orientation='horizontal')

    plt.show()

def maskshow(img):
    """ Visualize mask sample """
    Range_selection = range_selection()
    rain_omit, img_extent, omit_extent = Range_selection.execute_selection(
        Range_selection.pr,
        -55.0, 60, 0, 360,
        Range_selection.lat, Range_selection.lat_bnds,
        Range_selection.lon, Range_selection.lon_bnds)

    proj = ccrs.PlateCarree(central_longitude = 180)
    fig = plt.figure()
    ax = plt.subplot(projection = proj)
    ax.coastlines(resolution='50m', lw=0.5)
    ax.gridlines(xlocs = mticker.MultipleLocator(90),
                 ylocs = mticker.MultipleLocator(45),
                 linestyle = '-',
                 color = 'gray')

    tp = ax.imshow(img, cmap='RdBu_r' ,origin='lower',
                   extent=img_extent, transform=proj)
    fig.colorbar(tp, ax=ax, orientation='horizontal')
    plt.show()

if __name__ == '__main__':
    main()
