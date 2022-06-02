import os
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from jmacmap import jmacmap

def main():
    """ Test source """
    Range_selection = range_selection()

    ds, img_extent, ds_extent = Range_selection.execute_selection(
        Range_selection.pr,
        Range_selection.llcrnrlat, Range_selection.urcrnrlat,
        Range_selection.llcrnrlon, Range_selection.urcrnrlon,
        Range_selection.lat, Range_selection.lat_bnds,
        Range_selection.lon, Range_selection.lon_bnds)

    print(img_extent)
    print(ds.shape)
    print(Range_selection.llcrnrlat, Range_selection.urcrnrlat,
          Range_selection.llcrnrlon, Range_selection.urcrnrlon)
    print(ds_extent)
    print(Range_selection.lat[ ds_extent[0] : ds_extent[1] ])
    print(Range_selection.lon[ ds_extent[2] : ds_extent[3] ])

    llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon = -5, 30, 30, 170

    ds_a, img_extent_a, ds_extent_a = Range_selection.execute_selection(
        ds,
        llcrnrlat, urcrnrlat,
        llcrnrlon, urcrnrlon,
        Range_selection.lat[      ds_extent[0] : ds_extent[1] ],
        Range_selection.lat_bnds[ ds_extent[0] : ds_extent[1] ],
        Range_selection.lon[      ds_extent[2] : ds_extent[3] ],
        Range_selection.lon_bnds[ ds_extent[2] : ds_extent[3] ])

    Range_selection.plot(ds, img_extent)
    Range_selection.plot(ds_a, img_extent_a)

class range_selection():

    def __init__(self):
        #manual setting
        self.llcrnrlat = -30
        self.urcrnrlat = 50
        self.llcrnrlon = 30
        self.urcrnrlon = 170
        self.droot = '/docker/mnt/d/research/D1/data/CMIP6'
        self.model_name = 'CCCma.CanESM5'
        self.file_name = 'pr_Amon_CanESM5_historical_r1i1p1f1_gn_185001-201412.nc'
        self.path = os.path.join(self.droot,self.model_name,self.file_name)

        #auto determined
        self.dt = nc.Dataset(self.path,'r')
        self.lat = self.dt.variables['lat'][:]
        self.lat_bnds = self.dt.variables['lat_bnds'][:]
        self.lon = self.dt.variables['lon'][:]
        self.lon_bnds = self.dt.variables['lon_bnds'][:]
        self.time = self.dt.variables['time'][:]
        self.pr = self.dt.variables['pr'][:]
        self.pr = self.pr[8,:,:]

    def llcrnr_selection(self, target, field, bnds):
        center_index = np.argmin(np.abs(field - target))
        center = field[center_index]
        if (center - target) >= 0:
            index = center_index
            coordinate = bnds[index, 0]
        else:
            index = center_index + 1
            coordinate = bnds[index, 0]

        return index, coordinate

    def urcrnr_selection(self, target, field, bnds):
        center_index = np.argmin(np.abs(field - target))
        center = field[center_index]
        if (center - target) >= 0:
            index = center_index - 1
            coordinate = bnds[index, 1]
        else:
            index = center_index
            coordinate = bnds[index, 1]

        return index, coordinate

    def execute_selection(
            self, ds, llcrnrlat, urcrnrlat, llcrnrlon, urcrnrlon,
            lat, lat_bnds, lon, lon_bnds):
        """
        Reshape worldmap into disgnated lon/lat

        Parameters
        -----------
        llcrnrlat: float
            lower left corner latitude -90 to 90
        urcrnrlat: float
            upper right corner latitude -90 to 90
        llcrnrlon: float
            lower left corner longitude 0 to 360(0E centered)
        urcrnrlon: float
            upper right corner logitude 0 to 360(0E centered)
        ds: 2D ndarray
            dataset

        Returns
        -----------
        ds: 2D ndarray
            reshaped dataset. original => reshaped(lon/lat)
            grid value is converted to integer from float lon/lat values.
        img_extent: tuple
            (llcrnrlon, urcrnrlon, llcrnrlat, urcrnrlat)
            shifter for 180E centered worldmap
            cartopy world map is restricted following domain
        """
        llcrnrlat_index, selected_llcrnrlat = self.llcrnr_selection(
            llcrnrlat, lat, lat_bnds)
        urcrnrlat_index, selected_urcrnrlat = self.urcrnr_selection(
            urcrnrlat, lat, lat_bnds)
        llcrnrlon_index, selected_llcrnrlon = self.llcrnr_selection(
            llcrnrlon, lon, lon_bnds)
        urcrnrlon_index, selected_urcrnrlon = self.urcrnr_selection(
            urcrnrlon, lon, lon_bnds)

        img_extent = (selected_llcrnrlon-180, selected_urcrnrlon-180,
                      selected_llcrnrlat, selected_urcrnrlat)

        if llcrnrlat == -90:
            llcrnrlat_index = 0
            img_extent = (
                img_extent[0], img_extent[1], -90, img_extent[3])
        if urcrnrlat == 90:
            urcrnrlat_index = len(ds[:,0])
            img_extent = (
                img_extent[0], img_extent[1], img_extent[2], 90)
        if llcrnrlon == 0:
            llcrnrlon_index = 0
            img_extent = (
                -180, img_extent[1], img_extent[2], img_extent[3])
        if urcrnrlon == 360:
            urcrnrlon_index = len(ds[0,:])
            img_extent = (
                img_extent[0], 180, img_extent[2], img_extent[3])

        ds_extent = (llcrnrlat_index, urcrnrlat_index,
                     llcrnrlon_index, urcrnrlon_index)

        ds = ds[llcrnrlat_index:urcrnrlat_index,
                llcrnrlon_index:urcrnrlon_index]

        return ds, img_extent, ds_extent

    def plot(self, ds, img_extent):
        fig = plt.figure(figsize=(10,10))
        proj = ccrs.PlateCarree(central_longitude = 180)
        ax = plt.axes(projection=proj)
        tp = ax.imshow(
            ds ,cmap = jmacmap(), origin='lower',
            extent=img_extent,transform=proj)
        ax.gridlines(draw_labels=True)
        ax.coastlines()
        fig.colorbar(tp, ax=ax, orientation="horizontal")

        plt.show()

    def save(self,figpath):
        plt.savefig(figpath)

    def mk_mask(self):
        ept = np.ones((64,128))
        ept[34:40,35:38] = 0

        return ept

if __name__ == '__main__':
    main()
