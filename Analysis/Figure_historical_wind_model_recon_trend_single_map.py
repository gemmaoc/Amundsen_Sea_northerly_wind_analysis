#%%
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

#%% User parameters
cutoff_lat = -50
# plot_data = 'pace2_recon' #CESM1_PAC_PACE, pace2_recon
plot_data = 'CESM1_PAC_PACE'
time_per = [1920,2005]  # period for trend calculation

#%% Get data

data_dir = "/Users/gemma/Documents/Data/"
# File paths
file_dict = {
    'CESM1_PAC_PACE': {
        'psl': data_dir + "Model/Atmos/CESM1_PAC_PACE/annual_psl_PAC_PACE_ens_mean_1920_2005.nc",
        'u10': data_dir + "Model/Atmos/CESM1_PAC_PACE/annual_u10_PAC_PACE_ens_mean_1920_2005.nc",
        'v10': data_dir + "Model/Atmos/CESM1_PAC_PACE/annual_v10_PAC_PACE_ens_mean_1920_2005.nc"
    },
    'pace2_recon': {
        'psl': data_dir + "Proxy_reconstructions/PAC_PACE2_super_GKO1_all_bilinPSM_1mc_1800_2005_GISBrom_mitgcm_vars_psl.nc",
        'u10': data_dir + "Proxy_reconstructions/PAC_PACE2_super_GKO1_all_bilinPSM_1mc_1800_2005_GISBrom_mitgcm_vars_u10.nc",
        'v10': data_dir + "Proxy_reconstructions/PAC_PACE2_super_GKO1_all_bilinPSM_1mc_1800_2005_GISBrom_mitgcm_vars_v10.nc"
    }
}
psl_path = file_dict[plot_data]['psl']
u10_path = file_dict[plot_data]['u10']
v10_path = file_dict[plot_data]['v10']

# Load data
psl = xr.open_dataset(psl_path)['psl'].sel(time=slice(time_per[0],time_per[1])).squeeze()  # shape: (time, lat, lon)
u10 = xr.open_dataset(u10_path)['u10'].sel(time=slice(time_per[0],time_per[1])).squeeze()
v10 = xr.open_dataset(v10_path)['v10'].sel(time=slice(time_per[0],time_per[1])).squeeze()

# Convert SLP to hPa 
psl = psl / 100

#%% Calculate linear trend for SLP, u10, v10

def calc_trend(da):
    """Calculate linear trend along time axis for each grid point. Return in units per century."""
    years = da['time'].values
    X = years - years[0]
    trend = xr.apply_ufunc(
        lambda y: linregress(X, y)[0],  # slope
        da,
        input_core_dims=[['time']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float]
    )
    cent_trend = trend * 100  # convert to per century

    return cent_trend

psl_trend = calc_trend(psl)
u10_trend = calc_trend(u10)
v10_trend = calc_trend(v10)

# Focus on Antarctic region 
lat_mask = psl_trend['lat'] <= cutoff_lat
psl_trend_antarctic = psl_trend.sel(lat=lat_mask)
u10_trend_antarctic = u10_trend.sel(lat=lat_mask)
v10_trend_antarctic = v10_trend.sel(lat=lat_mask)

#%% Make map

# Set up figure
fig = plt.figure(figsize=(5, 6))
proj = ccrs.SouthPolarStereo(central_longitude=-100)
ax = plt.subplot(1, 1, 1, projection=proj)
ax.set_extent([-180, 180, -90, cutoff_lat], ccrs.PlateCarree())



# Plot SLP trend as filled contours
psl_trend_antarctic_cyclic, cyclic_lons = add_cyclic_point(psl_trend_antarctic.values, coord=psl_trend_antarctic['lon'])
u10_trend_antarctic_cyclic, _ = add_cyclic_point(u10_trend_antarctic.values, coord=u10_trend_antarctic['lon'])
v10_trend_antarctic_cyclic, _ = add_cyclic_point(v10_trend_antarctic.values, coord=v10_trend_antarctic['lon'])
lon2d, lat2d = np.meshgrid(cyclic_lons, psl_trend_antarctic['lat'])
c = ax.contourf(
    lon2d, lat2d, psl_trend_antarctic_cyclic,
    transform=ccrs.PlateCarree(), 
    cmap='BrBG_r', levels=np.arange(-3, 3.5, 0.5), extend='both')
# set colorbar
cb = plt.colorbar(c, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8, aspect=20)
cb.ax.tick_params(labelsize=12)
cb.set_label('SLP Trend (hPa/century)', fontsize=12)

# Plot wind trend quivers
# lat_density, lon_density, scale. larger density # means fewer arrows. larger quiver scale means
skip_dict = {'CESM1_PAC_PACE': (4, 12, 10), 
             'pace2_recon': (4, 12, 10)}
quiver_lat_density, quiver_lon_density, quiver_scale = skip_dict[plot_data]   # more arrows in latitude # larger number means fewer arrows #4 for pace em
# quiver_lon_density = 8  # fewer arrows in longitude #12 for pace em
# quiver_scale = 12 #larger number means smaller arrows #8 for pace em
skip = (slice(None, None, quiver_lat_density), slice(None, None, quiver_lon_density))
Q = ax.quiver(
    lon2d[skip], lat2d[skip],
    u10_trend_antarctic_cyclic[skip], v10_trend_antarctic_cyclic[skip],
    transform=ccrs.PlateCarree(),
    scale=quiver_scale, width=0.01, headwidth=3, headlength=4, 
    headaxislength=3.5, color='k',zorder=3,pivot='middle',
    edgecolor='white', linewidth=0.2)
ax.quiverkey(Q, 0.92, 0.05, 1, '1 m/s\nper century', labelpos='S')

# Make circular boundary
# Build a circle at cutoff_lat in lon/lat
lons_bound = np.linspace(-180, 180, 1000)   # dense sampling for smoothness
lats_bound = np.full_like(lons_bound, cutoff_lat)
# Project those points into the SouthPolarStereo coordinates
proj_pts = proj.transform_points(ccrs.PlateCarree(), lons_bound, lats_bound)
xs = proj_pts[:, 0]
ys = proj_pts[:, 1]
# Make a Path and set it as the boundary in data coordinates
verts = np.column_stack([xs, ys])
boundary_path = Path(verts)
ax.set_boundary(boundary_path, transform=ax.transData)
ax.patch.set_edgecolor("none")

# Add coastlines
ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=4)
ax.add_feature(cfeature.COASTLINE, linewidth=1, zorder=4)

# optional: draw a visible outline at 50S (will coincide with the boundary)
ax.plot(lons_bound, lats_bound, transform=ccrs.PlateCarree(), linewidth=3, color='k', zorder=10)


plt.title(plot_data + ' trends (1920-2005)',fontsize=16)
plt.tight_layout()
plt.show()
# %%
