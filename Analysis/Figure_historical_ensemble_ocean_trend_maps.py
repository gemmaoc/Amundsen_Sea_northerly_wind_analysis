#%% Load packages
import numpy as np
import xarray as xr
from scipy.stats import linregress
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import matplotlib.patches as mpatches
from shapely.geometry import Polygon
from matplotlib.path import Path
from scipy import stats
import matplotlib.path as mpath
import Functions_data_analysis as fda
import Functions_load_output as flo

#%% User parameters--------------------------------------------------------------
data_dir = '/Users/gemma/Documents/Data/Model/Ocean/Naughten_2022/'
region = 'AS_near'
region = 'full_model_domain'

#%% Load Naughten 2022 data-------------------------------------------------------------

# # Load Naughten on_shelf temp data ensemble timeseries
fname = 'MITgcm_AmundsenSeaContinentalShelf_1920-2013_sea_water_potential_temperature_200-700m_PACE'
ens_list = [f'{i:02}' for i in range(1, 21)]
n_ens = len(ens_list)

# get times for plotting all members
ds = xr.open_dataset(data_dir + fname + ens_list[0] + '.nc')
temp = ds.sea_water_potential_temperature
years = np.arange(1920,2014,1)

# load T, uwinds, and vwinds spatial trends from Naughten et al. 2022

# time = 20 ensemble members
# X = 600 lons
# Y = 384 lats

# on-shelf temps
ds = xr.open_dataset(data_dir + 'temp_btw_200_700m.nc')
naught_T_trends = ds.temp_btw_200_700m_trend #shape (20 time, 384 Y, 600 X)

# uwinds
ds = xr.open_dataset(data_dir + 'MITgcm_AmundsenSea_1920-2013_trend_in_eastward_wind_PACE.nc')
naught_uwind_trends = ds['trend_in_eastward_wind'] #shape (20 mems, 384 lats, 600 lons, but dims are labeled "time", "Y", and "X" )
naught_lons,naught_lats = naught_uwind_trends.X, naught_uwind_trends.Y

# vwinds
ds = xr.open_dataset(data_dir + 'MITgcm_AmundsenSea_1920-2013_trend_in_northward_wind_PACE.nc')
naught_vwind_trends = ds['trend_in_northward_wind'] 

#%% Get bathyemtry and ice data

lat1,lat2,lon1,lon2 = fda.plot_regions[region]
land_ice_ds = flo.get_bathymetry_and_troughs()
land_ice_ds = land_ice_ds.sel(lat=slice(lat1,lat2),lon=slice(lon1,lon2))
ice_lons,ice_lats = land_ice_ds.lon, land_ice_ds.lat

#%% Calculate on-shelf temp trends for full ensemble----------------------------------------------------

cent_CDW_trends = []
plt.figure(figsize=(12,6))
for i in range(n_ens):
    ens_ds = xr.open_dataset(data_dir + fname + ens_list[i] + '.nc')
    ens_temp = ens_ds.sea_water_potential_temperature
    ens_temp_ann = ens_temp.resample(time='1Y').mean()
    plt.plot(years,ens_temp_ann,label=ens_list[i])
    # calc trend
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, ens_temp_ann)
    if p_value < 0.05:
        sig = '*'
    else:
        sig = ''
    print(ens_list[i] + ': {:.2f} degC/cent'.format(slope*100)+sig)
    cent_CDW_trends.append(slope*100)
plt.legend(ncol=10)
plt.title('Naughten et al., 2022 ensemble on-shelf temperatures')
plt.ylabel('Pot. Temp [°C]')

print('Ensemble mean trend:', np.mean(cent_CDW_trends))

#%% Make subplots of all ensemble member T, uwind, and vwind trends-------------------------------------------------
fig, axes = plt.subplots(4, 5, figsize=(15, 11),
                         subplot_kw={'projection': ccrs.SouthPolarStereo(central_longitude=-105)})
axes = axes.flatten()  

lat1, lat2, lon1, lon2 = fda.plot_regions[region]      
T_lim = 1.0  # colorbar limit for temp trend
grid_proj = ccrs.PlateCarree()

for i in range(n_ens):

    print(i)
    ax = axes[i]
    
    # get ensemble member data
    temp_trend = naught_T_trends.isel(time=i)*100  # convert to per century
    u10_trend = naught_uwind_trends.isel(time=i)*100
    v10_trend = naught_vwind_trends.isel(time=i)*100

    #mask large values where ice is 
    temp_trend_masked = np.where(temp_trend > 1e30, np.ma.masked, temp_trend)
    u10_trend_masked = np.where(u10_trend > 1e30, np.ma.masked, u10_trend)
    v10_trend_masked = np.where(v10_trend > 1e30, np.ma.masked, v10_trend)
    cf = ax.pcolormesh(naught_lons, naught_lats, temp_trend_masked, vmin=-T_lim, vmax=T_lim,
                       transform=grid_proj, cmap='RdBu_r', zorder=0)
    
    # # Plot wind trend quivers
    # 600 lons, 384 lats, shape of data = (384 lats, 600 lons)
    if region == 'AS_near':
        lat_skip, lon_skip, scale = 10, 20, 4
    elif region == 'full_model_domain':
        lat_skip, lon_skip, scale = 40, 80, 10
    Q = ax.quiver(
        naught_lons[::lon_skip], naught_lats[::lat_skip],
        u10_trend_masked[::lat_skip,::lon_skip], v10_trend_masked[::lat_skip,::lon_skip],
        transform=ccrs.PlateCarree(),
        scale=scale, width=0.015, headwidth=3, headlength=4,
        headaxislength=3.5, color='k', zorder=3,
        edgecolor='white', linewidth=0.1,pivot='middle')
    
    # Plot bathyemtry and ice and plot
    land_ice_ds = land_ice_ds.sel(lat=slice(lat1,lat2),lon=slice(lon1,lon2))
    lons,lats = land_ice_ds.lon, land_ice_ds.lat
    blevs = (1000,)
    ax.contour(lons,lats,land_ice_ds.bathy,blevs,colors='k',transform=grid_proj,linewidths=1,zorder=1)
    ax.contourf(lons,lats,land_ice_ds.all_ice,transform=grid_proj,colors=['lightgray']*2,alpha=0.6,zorder=2)
    ax.contourf(lons,lats,land_ice_ds.grounded_ice,transform=grid_proj,cmap='binary_r',zorder=2)
    
    # Set shape of map to match shape of data rather than a rectangle
    rect = mpath.Path([[lon1, lat2], [lon2, lat2],[lon2, lat1], [lon1, lat1], [lon1, lat2]]).interpolated(50)
    proj_to_data = ccrs.PlateCarree()._as_mpl_transform(ax) - ax.transData
    rect_in_target = proj_to_data.transform_path(rect)
    ax.set_boundary(rect_in_target)
    ax.set_xlim(rect_in_target.vertices[:,0].min(), rect_in_target.vertices[:,0].max())
    ax.set_ylim(rect_in_target.vertices[:,1].min(), rect_in_target.vertices[:,1].max())
    
    ax.set_title(f"Ensemble {ens_list[i]}" + "\n{:.2f}°C per cent".format(cent_CDW_trends[i]), fontsize=12)    

# Add colorbar
cbar_ax = fig.add_axes([0.25, 0.1, 0.5, 0.02])
cb = fig.colorbar(cf, cax=cbar_ax, orientation='horizontal',extend='both')
cb.ax.tick_params(labelsize=12)
cb.set_label('On-shelf Temp Trend (°C/century)', fontsize=12)
plt.suptitle('Naughten et al., 2022 ensemble trends (1920-2013)', fontsize=16)

plt.show()

#%% Make single plot of one ensemble member T, uwind, and vwind trends-------------------------------------------------
