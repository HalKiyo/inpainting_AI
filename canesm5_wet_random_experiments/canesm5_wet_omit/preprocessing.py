import os
import pickle
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from jmacmap import jmacmap
from range_selection import range_selection

def main():
    """ Preprocess data for model """

    home = '/docker/home/hasegawa'
    root = 'docker-gpu/reconstructionAI/canesm5_wet_random_experiments'
    experiment = 'canesm5_wet_omit'
    pcl = 'data/canesm5_wet_omit.pickle'

    path = os.path.join(home, root, experiment, pcl)
    train = os.path.join(home, root, experiment, 'data/canesm5_wet_omit_train.npy')
    valid = os.path.join(home, root, experiment, 'data/canesm5_wet_omit_valid.npy')
    mask_path = os.path.join(home, root, experiment, 'data/canesm5_wet_omit_eval_mask.npy')
    tmsr_path = os.path.join(home, root, experiment, 'data/timeseries')

    #gen_npy(path, train, valid, save='on')
    #gen_timeseriese_valid(path, tmsr_path, save='on')
    #canvas = gen_mask(mask_path, save='off')
    #maskshow(canvas)
    show(path)

def read(name):
    """ Detail is shown in data_docker-conda/preparation/omit.py """
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data

def gen_npy(path, train, valid, save='off'):
    """
    Save ndarray file to /experiment/data/ folder
    Detale is shown in /data_preparation/omit_anom.py/{omitting}
    """
    dct = read(path)
    omt = np.array(dct['rain_omit'])

    rng = np.random.default_rng()
    perm = rng.permutation(len(omt))
    train_end = int(0.8 * len(omt))
    valid_end = int(0.2 * len(omt)) + train_end

    train_npy = omt[perm[:train_end]]
    valid_npy = omt[perm[train_end:valid_end]]

    if save == 'on':
        print(f'input_save: {save}')
        np.save(train, train_npy)
        np.save(valid, valid_npy)

def gen_timeseriese_valid(path, savepath, save='off'):
    dct = read(path)
    df = pd.DataFrame(dct)
    month = df[ df['month'] == 6 ]

    for i in range(1850, 2015):
        year = month[ month['year'] == i ]
        year_npy = np.empty((len(year), 40, 128))

        for j in range(len(year)):
            year_npy[j,:,:] = year['rain_omit'].iat[j]

        if not os.path.exists(f'{savepath}/{i}'):
            os.makedirs(f'{savepath}/{i}')

        if save == 'on':
            filepath = f'{savepath}/{i}/valid_june_{i}.npy'
            np.save(filepath, year_npy)
            print(f'timeseriese_save{i}: {save}')

def add_obs(canvas, rain_omit, omit_extent, a, b, c, d):
    llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon = a, b, c, d

    Range_selection = range_selection()
    ds, img_extent, ds_extent = Range_selection.execute_selection(
        rain_omit,
        llcrnrlat, urcrnrlat,
        llcrnrlon, urcrnrlon,
        Range_selection.lat[      omit_extent[0] : omit_extent[1] ],
        Range_selection.lat_bnds[ omit_extent[0] : omit_extent[1] ],
        Range_selection.lon[      omit_extent[2] : omit_extent[3] ],
        Range_selection.lon_bnds[ omit_extent[2] : omit_extent[3] ])

    canvas[ds_extent[0] : ds_extent[1], ds_extent[2] : ds_extent[3]] = 1

    return canvas

def gen_mask(mask_path, save='off'):
    image_size = (40, 128)

    canvas = np.zeros(image_size)

    Range_selection = range_selection()
    rain_omit, img_extent, omit_extent = Range_selection.execute_selection(
        Range_selection.pr,
        -55.0, 57.5, 0, 360,
        Range_selection.lat, Range_selection.lat_bnds,
        Range_selection.lon, Range_selection.lon_bnds)


    canvas = add_obs(canvas, rain_omit, omit_extent, -35, -10, 10, 40)
    canvas = add_obs(canvas, rain_omit, omit_extent, -45, -5, 110, 155)
    canvas = add_obs(canvas, rain_omit, omit_extent, -45, -5, 300, 330)
    canvas = add_obs(canvas, rain_omit, omit_extent, 5, 40, 90, 120)
    canvas = add_obs(canvas, rain_omit, omit_extent, 5, 90, 230, 310)
    canvas = add_obs(canvas, rain_omit, omit_extent, 30, 90, 345, 360)
    canvas = add_obs(canvas, rain_omit, omit_extent, 30, 90, 0, 165)
    canvas = add_obs(canvas, rain_omit, omit_extent, 50, 90, 180, 240)

    if save == 'on':
        print(f'mask_save: {save}')
        np.save(mask_path, canvas)

    return canvas

def show(path):
    """
    Load data then visualize a sample binary
    dct is sorted by month, then year looks random if ind=0 is selected.
    """
    # arguments object
    ind = 1
    dct = read(path)
    omt = np.array(dct['rain_omit'])
    img = omt[ind]

    Range_selection = range_selection()
    rain_omit, img_extent, omit_extent = Range_selection.execute_selection(
        Range_selection.pr,
        -55.0, 57.5, 0, 360,
        Range_selection.lat, Range_selection.lat_bnds,
        Range_selection.lon, Range_selection.lon_bnds)
    title = f'{dct["year"][ind]}/{dct["month"][ind]}'
    proj = ccrs.PlateCarree(central_longitude = 180)

    # figure object
    fig = plt.figure()
    ax = plt.subplot(projection = proj)

    ax.coastlines(resolution='50m', lw=0.5)
    ax.gridlines(xlocs = mticker.MultipleLocator(90),
                 ylocs = mticker.MultipleLocator(45),
                 linestyle = '-',
                 color = 'gray')

    tp = ax.imshow(img, cmap=jmacmap() ,origin='upper',
                   extent=img_extent, transform=proj,
                   vmin=0, vmax=0.0004)

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
