import os
import pickle
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
from jmacmap import jmacmap
from range_selection import range_selection

def main():
    root = '/docker/home/hasegawa/docker-gpu/reconstructionAI/'\
           'canesm5_wet_experiments/canesm5_wet_omit'
    vname = 'valid/valid700000.npy'

    # data loading
    vpath = os.path.join(root, vname)
    val = np.load(vpath) # shape (6165, 5, 40, 128)

    # mask range selection
    img_extent, mask_extent = mask_range() # (west, east, north, south)
    val_thailand = val[:, :, mask_extent[0]:mask_extent[1], mask_extent[2]:mask_extent[3]]

    # evaluation by grid
    mae_eval = mae(val_thailand) # shape (6, 5)
    rb_eval = relative_bias(val_thailand)
    cc_eval = correlation_coefficient(val_thailand)
    csi_eval = critical_index(val_thailand)
    csi_rb_eval = csi_rb(val_thailand)

    # evaluation with all grid
    ensemble_mean_mm, mae_tmsr_mm = timeseriese_MAE()

    # collection plotting
    val_all = [ val_thailand[3,2,:,:],
                val_thailand[3,4,:,:],
                mae_eval,
                rb_eval,
                cc_eval,
                csi_eval ]

    # visualization optional comment out
    #sample_selection(val_thailand) # 1769, 4463, 109
    #hist_show(val_thailand)
    #show_fullrange(val)
    #rb_show(rb_eval, img_extent)
    #mae_show(mae_eval, img_extent)
    #cc_show(cc_eval, img_extent)
    #csi_show(csi_eval, img_extent)
    csi_rb_show(csi_rb_eval, img_extent)
    #tmsr_show(ensemble_mean_mm, mae_tmsr_mm)
    #val_show(val_thailand[1769], img_extent)
    #all_show(val_all, img_extent)

def sample_selection(data):
    map_sample = data[0,0,:,:]
    pred = data[:,2,:,:]
    true = data[:,4,:,:]
    bias = np.abs(pred - true)

    valid_bias = []
    for i in range(len(data)):
        bias_grids = []
        for j in range(map_sample.shape[0]):
            for k in range(map_sample.shape[1]):
                bias = np.abs( pred[i,j,k] - true[i,j,k] )
                bias_grids.append(bias)
        valid_bias.append(sum(bias_grids))

    valid_arry = np.array(valid_bias)
    valid_sort = np.argsort(valid_arry)
    print(valid_sort)

def hist_show(data):
    """ Grid base """
    row, column = len(data[0,4,:,0]), len(data[0,4,0,:])
    fig = plt.figure(figsize=(3,4))
    fig.suptitle('histgram of monthly rainfall (mm/day)')

    t = 0
    for i in range(row):
        for j in range(column):
            t += 1
            ax = fig.add_subplot(row, column, t)
            rain = [ data[k, 4, i, j] * 86400 for k in range(len(data)) ]
            ax.hist(rain, color='black')
            ax.axvline(x=20, ymin=0, ymax=4000, color='crimson')
            ax.set_xlim(0,25)
            ax.set_xticks(np.arange(0,25,5))
            ax.set_yticks(np.arange(0,5000,2000))

    plt.show()

def mae(data):
    map_sample = data[0,0,:,:]
    mae_eval = np.zeros(map_sample.shape)
    pred = data[:,2,:,:]
    true = data[:,4,:,:]
    for i in range(map_sample.shape[0]):
        for j in range(map_sample.shape[1]):
            mae_eval[i,j] = mean_absolute_error( pred[:,i,j], true[:,i,j] )

    return mae_eval

def relative_bias(data):
    map_sample = data[0,0,:,:]
    rb_eval = np.zeros(map_sample.shape)
    pred = data[:,2,:,:]
    true = data[:,4,:,:]
    for i in range(map_sample.shape[0]):
        for j in range(map_sample.shape[1]):
            #rb = np.sum( np.abs( pred[:,i,j] - true[:,i,j] ) ) / np.sum( true[:,i,j] )
            rb = np.sum( pred[:,i,j] - true[:,i,j] ) / np.sum( true[:,i,j] )
            rb_eval[i,j] = rb*100

    return rb_eval

def correlation_coefficient(data):
    map_sample = data[0,0,:,:]
    cc_eval = np.zeros(map_sample.shape)
    pred = data[:,2,:,:]
    true = data[:,4,:,:]
    for i in range(map_sample.shape[0]):
        for j in range(map_sample.shape[1]):
            cc = np.corrcoef( pred[:,i,j], true[:,i,j] )[0,1]
            cc_eval[i,j] = cc

    return cc_eval

def critical_index(data):
    """ underestimation evaluation """
    sample = data[0,0,:,:]
    pred = data[:,2,:,:]
    true = data[:,4,:,:]

    criterion = 0.00026
    csi_eval = np.zeros(sample.shape)
    bl_pr = np.zeros( (len(data), len(sample[:,0]), len(sample[0,:])) )
    bl_tr = np.zeros( (len(data), len(sample[:,0]), len(sample[0,:])) )

    for i in range(sample.shape[0]):
        for j in range(sample.shape[1]):
            for k in range(len(data[:,4,:,:])):
                if pred[k,i,j] > criterion:
                    bl_pr[k,i,j] = 1
                else:
                    bl_pr[k,i,j] = 0
                if true[k,i,j] > criterion:
                    bl_tr[k,i,j] = 1
                else:
                    bl_tr[k,i,j] = 0

    for i in range(sample.shape[0]):
        for j in range(sample.shape[1]):
            cm = confusion_matrix( bl_tr[:,i,j], bl_pr[:,i,j] )
            csi = cm[1,1]/(cm[0,1] + cm[1,0] + cm[1,1])
            csi_eval[i,j] = csi

    return csi_eval

def csi_rb(data):
    """ bias evaluation with csi
                         Predicted
                     Negative  Positive
    Actual Negative     TN        FP
           Positive     FN        TP
    """
    sample = data[0,0,:,:]
    pred = data[:,2,:,:]*86400
    true = data[:,4,:,:]*86400

    criterion = 20
    rng = 5
    rb = criterion * 0.4
    csi_rb_eval = np.zeros(sample.shape)
    bl_pr = np.zeros( (len(data), len(sample[:,0]), len(sample[0,:])) )
    bl_tr = np.zeros( (len(data), len(sample[:,0]), len(sample[0,:])) )

    for i in range(sample.shape[0]):
        for j in range(sample.shape[1]):
            for k in range(len(data[:,4,:,:])):
                condition = criterion < true[k,i,j] < criterion + rng
                if condition and np.abs(pred[k,i,j] - true[k,i,j]) < rb:
                    bl_pr[k,i,j] = 1
                else:
                    bl_pr[k,i,j] = 0
                if condition:
                    bl_tr[k,i,j] = 1
                else:
                    bl_tr[k,i,j] = 0

    for i in range(sample.shape[0]):
        for j in range(sample.shape[1]):
            cm = confusion_matrix( bl_tr[:,i,j], bl_pr[:,i,j] )
            csi = cm[1,1]/(cm[0,1] + cm[1,0] + cm[1,1])
            csi_rb_eval[i,j] = csi

    return csi_rb_eval

def timeseriese_MAE():
    """
    Ensemble mean of rainfall in June every year from 1850 to 2014.
    Returns:
        ensemble_mean_mm: ensemble mean of true value transformed to mm/day.
        mae_tmsr_mm: ensemble mean of absolute error transformed to mm/day.
    """
    path = '/docker/home/hasegawa/docker-gpu/reconstructionAI/'\
           'canesm5_wet_experiments/canesm5_wet_omit/data/timeseries'

    begin, end = 1850, 2015
    count = 2015 - 1850
    ensemble_mean = []
    mae_tmsr = []

    for year in range(begin, end):
        ypath = f'{path}/{year}/valid/valid700000.npy'
        val = np.load(ypath) # shape (40-60, 5, 40, 128)

        pred = val[:,2,:,:].mean(axis=1).mean(axis=1)
        true = val[:,4,:,:].mean(axis=1).mean(axis=1)

        ensemble = np.mean(true)
        ensemble_mean.append(ensemble)

        mae = mean_absolute_error( pred, true )
        mae_tmsr.append(mae)

    ensemble_mean_mm = [ n*86400 for n in ensemble_mean ]
    mae_tmsr_mm = [ n*86400 for n in mae_tmsr ]

    return ensemble_mean_mm, mae_tmsr_mm

def mask_range():
    llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon = 5, 25, 95, 110

    # 1 selection. (180, 360) -> (112.5, 360)
    Range_selection = range_selection()
    rain_omit, _, omit_extent = Range_selection.execute_selection(
        Range_selection.pr,
        -55.0, 57.5, 0, 360,
        Range_selection.lat, Range_selection.lat_bnds,
        Range_selection.lon, Range_selection.lon_bnds)

    # 2 selection. (112.5, 360) -> (20, 15)
    mask, img_extent, mask_extent = Range_selection.execute_selection(
        rain_omit,
        llcrnrlat, urcrnrlat,
        llcrnrlon, urcrnrlon,
        Range_selection.lat[      omit_extent[0] : omit_extent[1] ],
        Range_selection.lat_bnds[ omit_extent[0] : omit_extent[1] ],
        Range_selection.lon[      omit_extent[2] : omit_extent[3] ],
        Range_selection.lon_bnds[ omit_extent[2] : omit_extent[3] ])

    return img_extent, mask_extent

def show_fullrange(img):
    gt = img[:,4,:,:]
    gt = gt*86400
    gt = gt[1769]
    img_extent, mask_extent = mask_range()
    gt[ mask_extent[0] : mask_extent[1], mask_extent[2] : mask_extent[3] ] = 0

    img_extent = (-180, 180, -55, 57.5)
    imgs_label = ['input', 'mask', 'model_output', 'output_comp', 'ground_truth']
    vmin, vmax = 0, 0.0004*86400
    grids = 20

    cname = jmacmap()
    interval = (vmax - vmin)/grids
    ticks = np.round( np.arange(vmin, vmax+interval, interval),1)
    cmap = plt.get_cmap(cname, grids)
    proj = ccrs.PlateCarree(central_longitude = 180)

    # figure object
    fig = plt.figure()

    ax = plt.subplot(projection=proj)

    ax.coastlines(resolution='50m', lw=0.5)
    ax.gridlines(xlocs = mticker.MultipleLocator(90),
                 ylocs = mticker.MultipleLocator(45),
                 linestyle = '-',
                 color = 'gray')

    mat = ax.matshow( gt,
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

def mae_show(img, img_extent):
    vmin, vmax = 0, 0.0004*86400
    grids = 15
    cname = 'rainbow'
    interval = (vmax - vmin)/grids
    ticks = np.round( np.arange(vmin, vmax+interval, interval) )
    cmap = plt.get_cmap(cname, grids)
    proj = ccrs.PlateCarree( central_longitude = 180 )

    fig = plt.figure()
    ax = plt.subplot(projection=proj)

    ax.coastlines(resolution='50m', lw=0.5)
    ax.gridlines(xlocs = mticker.MultipleLocator(90),
                 ylocs = mticker.MultipleLocator(45),
                 linestyle = '-',
                 color = 'gray')
    ax.set_title('mean absolute error (mm/day)')

    mat = ax.matshow( img*86400,
                      cmap=cmap,
                      origin='lower',
                      extent=img_extent,
                      transform=proj,
                      vmin=vmin - interval/2.0,
                      vmax=vmax + interval/2.0 )
    cbar = fig.colorbar( mat,
                         ax=ax )
    cbar.set_ticks(ticks[::1])
    #cbar.ax.set_xticklabels(ticks[::2], rotation=-90)

    plt.show()

def rb_show(img, img_extent):
    vmin, vmax = 0, 100
    grids = 20
    cname = 'rainbow'
    interval = (vmax - vmin)/grids
    ticks = np.arange(vmin, vmax+interval, interval)
    cmap = plt.get_cmap(cname, grids)
    proj = ccrs.PlateCarree( central_longitude = 180 )

    fig = plt.figure()
    ax = plt.subplot(projection=proj)

    ax.coastlines(resolution='50m', lw=0.5)
    ax.gridlines( xlocs = mticker.MultipleLocator(90),
                  ylocs = mticker.MultipleLocator(45),
                  linestyle = '-',
                  color = 'gray' )
    ax.set_title( 'relative bias(%)' )

    mat = ax.matshow( img,
                      cmap=cmap,
                      origin='lower',
                      extent=img_extent,
                      transform=proj,
                      vmin=vmin - interval/2.0,
                      vmax=vmax + interval/2.0 )
    cbar = fig.colorbar( mat,
                         ax=ax )
    cbar.set_ticks(ticks[::1])
    #cbar.ax.set_xticklabels(ticks[::1], rotation=-90)

    plt.show()

def cc_show(img, img_extent):
    vmin, vmax = 0, 1
    grids = 5
    cname = 'RdPu'
    interval = ( vmax - vmin ) / grids
    ticks = np.round( np.arange( vmin, vmax + interval, interval ), 1 )
    cmap = plt.get_cmap( cname, grids )
    proj = ccrs.PlateCarree( central_longitude = 180 )

    fig = plt.figure()
    ax = plt.subplot( projection = proj )

    ax.coastlines( resolution='50m', lw=0.5 )
    ax.gridlines( xlocs = mticker.MultipleLocator(90),
                  ylocs = mticker.MultipleLocator(45),
                  linestyle = '-',
                  color = 'gray' )
    ax.set_title( 'correlation coefficient' )

    mat = ax. matshow( img,
                       cmap=cmap,
                       origin='lower',
                       extent=img_extent,
                       transform=proj,
                       vmin=vmin - interval/2.0,
                       vmax=vmax + interval/2.0 )
    cbar = fig.colorbar( mat,
                         ax=ax,
                         orientation='horizontal' )
    cbar.set_ticks( ticks[::1] )
    cbar.ax.set_xticklabels( ticks[::1], rotation=-90 )

    plt.show()

def csi_show(img, img_extent):
    vmin, vmax = 0, 1
    grids = 5
    cname = 'RdPu'
    interval = ( vmax - vmin ) / grids
    ticks = np.round( np.arange( vmin, vmax + interval, interval ), 1 )
    cmap = plt.get_cmap( cname, grids )
    proj = ccrs.PlateCarree( central_longitude = 180 )

    fig = plt.figure()
    ax = plt.subplot( projection = proj )

    ax.coastlines( resolution='50m', lw=0.5 )
    ax.gridlines( xlocs = mticker.MultipleLocator(90),
                  ylocs = mticker.MultipleLocator(45),
                  linestyle = '-',
                  color = 'gray' )
    ax.set_title( 'correlation coefficient' )

    mat = ax. matshow( img,
                       cmap=cmap,
                       origin='lower',
                       extent=img_extent,
                       transform=proj,
                       vmin=vmin - interval/2.0,
                       vmax=vmax + interval/2.0 )
    cbar = fig.colorbar( mat,
                         ax=ax,
                         orientation='horizontal' )
    cbar.set_ticks( ticks[::1] )
    cbar.ax.set_xticklabels( ticks[::1], rotation=-90 )

    plt.show()

def csi_rb_show(img, img_extent):
    criterion = 20
    vmin, vmax = 0, 1
    grids = 10
    cname = 'RdPu'
    interval = ( vmax - vmin ) / grids
    ticks = np.round( np.arange( vmin, vmax + interval, interval ), 1 )
    cmap = plt.get_cmap( cname, grids )
    proj = ccrs.PlateCarree( central_longitude = 180 )

    fig = plt.figure()
    ax = plt.subplot( projection = proj )

    ax.coastlines( resolution='50m', lw=0.5 )
    ax.gridlines( xlocs = mticker.MultipleLocator(90),
                  ylocs = mticker.MultipleLocator(45),
                  linestyle = '-',
                  color = 'gray' )
    ax.set_title( f'critical success index >{criterion}' )

    mat = ax. matshow( img,
                       cmap=cmap,
                       origin='lower',
                       extent=img_extent,
                       transform=proj,
                       vmin=vmin - interval/2.0,
                       vmax=vmax + interval/2.0 )
    cbar = fig.colorbar( mat,
                         ax=ax,
                         orientation='horizontal' )
    cbar.set_ticks( ticks[::1] )
    cbar.ax.set_xticklabels( ticks[::1], rotation=-90 )

    plt.show()

def tmsr_show(gt, mae):
    xaxis = np.arange(1850, 2015)
    rb = [ (i*100)/j for i, j in zip(mae, gt) ]

    fig, ax = plt.subplots()
    #b = ax.bar(xaxis, gt, label='ground truth')
    #p = ax.plot(xaxis, mae, label='mean absolute error')
    p2 = ax.scatter(xaxis, rb, label='relative bias(%)', color='mediumturquoise')
    ax.set_yticks(np.arange(10,21,5))
    plt.legend()
    plt.show()

def val_show(imgs, img_extent):
    imgs_label = ['input', 'mask', 'model_output', 'output_comp', 'ground_truth']
    vmin, vmax = 0, 0.0004*86400
    grids = 10
    cname = jmacmap()
    interval = (vmax - vmin)/grids
    ticks = np.rint( np.arange(vmin, vmax+interval, interval) )
    cmap = plt.get_cmap(cname, grids)
    proj = ccrs.PlateCarree(central_longitude = 180)

    # figure layout
    nrows = 1
    ncols = 2
    pos_1 = nrows*100 + ncols*10 + 1
    pos = [i for i in range(pos_1, pos_1 + nrows*ncols)]
    index = [2, 4] # output, gt

    # figure object
    fig = plt.figure()

    for ind, num in zip(index,pos[:ncols]):
        ax = plt.subplot(num, projection=proj)

        ax.coastlines(resolution='50m', lw=0.5)
        ax.gridlines(xlocs = mticker.MultipleLocator(90),
                     ylocs = mticker.MultipleLocator(45),
                     linestyle = '-',
                     color = 'gray')
        ax.set_title(imgs_label[ind])


        mat = ax.matshow( imgs[ind]*86400,
                          cmap=cmap,
                          origin='lower',
                          extent=img_extent,
                          transform=proj,
                          vmin=vmin - interval/2,
                          vmax=vmax + interval/2 )
        cbar = fig.colorbar( mat,
                             ax=ax )
        cbar.set_ticks(ticks[::1], size=14)

    plt.show()

def all_show(imgs, img_extent):
    imgs_label = [ 'model_output (kg/m2/s)',
                   'ground_truth',
                   'mean_absolute_error',
                   'relative_bias',
                   'correlaton_coefficient',
                   'critical_success_index' ]
    proj = ccrs.PlateCarree(central_longitude = 180)

    fig = plt.figure()

    # model_output
    ax1 = plt.subplot(221, projection=proj)
    ax1.coastlines(resolution='50m', lw=0.5)
    ax1.gridlines(xlocs = mticker.MultipleLocator(90),ylocs = mticker.MultipleLocator(45),linestyle = '-',color = 'gray')
    ax1.set_title(imgs_label[0])
    mat = ax1.imshow(imgs[0],cmap=jmacmap(),origin='lower',extent=img_extent,transform=proj,vmin=0,vmax=0.0004 )
    cb = fig.colorbar(mat, ax=ax1, orientation='horizontal')

    # ground_truth
    ax2 = plt.subplot(222, projection=proj)
    ax2.coastlines(resolution='50m', lw=0.5)
    ax2.gridlines(xlocs = mticker.MultipleLocator(90),ylocs = mticker.MultipleLocator(45),linestyle = '-',color = 'gray')
    ax2.set_title(imgs_label[1])
    mat = ax2.imshow(imgs[1],cmap=jmacmap(),origin='lower',extent=img_extent,transform=proj,vmin=0,vmax=0.0004 )
    cb = fig.colorbar(mat, ax=ax2, orientation='horizontal')

    # mean_absolute_error
    vmin, vmax = 0, 0.0004
    grids = 20
    cname = 'rainbow'
    interval = (vmax - vmin)/grids
    ticks = np.round( np.arange(vmin, vmax+interval, interval), 6)
    cmap = plt.get_cmap(cname, grids)
    ax3 = plt.subplot(223, projection=proj)
    ax3.coastlines(resolution='50m', lw=0.5)
    ax3.gridlines(xlocs = mticker.MultipleLocator(90),ylocs = mticker.MultipleLocator(45),linestyle = '-',color = 'gray')
    ax3.set_title(imgs_label[2])
    mat = ax3.imshow(imgs[2],cmap=cmap,origin='lower',extent=img_extent,transform=proj,vmin=vmin,vmax=vmax )
    cb = fig.colorbar(mat, ax=ax3, orientation='horizontal')
    cb.set_ticks(ticks[::1])
    cb.ax.set_xticklabels(ticks[::1], rotation=-90)

    # critical_success_index
    vmin, vmax = 0, 1
    grids = 5
    cname = 'RdPu'
    interval = (vmax - vmin)/grids
    ticks = np.round( np.arange(vmin, vmax+interval, interval), 6)
    cmap = plt.get_cmap(cname, grids)
    ax4 = plt.subplot(224, projection=proj)
    ax4.coastlines(resolution='50m', lw=0.5)
    ax4.gridlines(xlocs = mticker.MultipleLocator(90),ylocs = mticker.MultipleLocator(45),linestyle = '-',color = 'gray')
    ax4.set_title(imgs_label[5])
    mat = ax4.imshow(imgs[5],cmap=cmap,origin='lower',extent=img_extent,transform=proj,vmin=vmin,vmax=vmax )
    cb = fig.colorbar(mat, ax=ax4, orientation='horizontal')
    cb.set_ticks(ticks[::1])
    cb.ax.set_xticklabels(ticks[::1], rotation=-90)

    plt.show()


if __name__ == '__main__':
    main()
