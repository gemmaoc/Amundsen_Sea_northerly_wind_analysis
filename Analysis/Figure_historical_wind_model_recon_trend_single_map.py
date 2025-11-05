#%%

# Plot one trend map of quivers and color contours for either historical or future data
#  (either proxy recons of ensemble mean of simulations)

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

# map northern lat extent
cutoff_lat = -40

# historical params
plot_data = 'CESM1_LME_recon' 
# plot_data = 'CESM1_PAC_PACE'
# plot_data = 'CESM1_LENS_historical'
time_per = [1920,2005]  # period for trend calculation

# future params
# plot_data = 'CESM2_SSP585'
# plot_data = 'CESM2_SSP370'
# plot_data = 'CESM2_SSP245'
# time_per = [2015,2100]

#%% Get data

data_dir = "/Users/gemma/Documents/Data/"
# File paths
def get_data(data_source, variable, time_period):

    """Load dataset for given data type and variable."""

    base_paths = {
        # Simulations
        'CESM1_PAC_PACE': data_dir + "Model/Atmos/CESM1_PAC_PACE/" +
                         "annual_{}_PAC_PACE_ens_mean_1920_2005.nc",
        'CESM1_LENS_historical': data_dir + "Model/Atmos/CESM1_LENS_historical/" +
                                "annual_{}_LENS_ens_mean_1920_2005.nc",
        'CESM2_SSP585': data_dir + "Model/Atmos/CESM2_SSP585/" +
                       "b.e21.BSSP585smbb.f09_g17.ens_mean.{}.2015-2100.nc",
        'CESM2_SSP370': data_dir + "Model/Atmos/CESM2_SSP370/" +
                       "b.e21.BSSP370cmip6smbb.f09_g17.ens_mean.{}.2015-2100.nc", 
        'CESM2_SSP245': data_dir + "Model/Atmos/CESM2_SSP245/" +
                       "b.e21.BSSP245smbb.f09_g17.ens_mean.{}.2015-2100.nc",
        
        # Proxy recons   
        'CESM1_LME_recon': data_dir + "Proxy_reconstructions/iCESM1_LME_recon_1800_2005/" +
                          "iCESM1_LME_recon_1800_2005_{}.nc",
        'CESM1_PAC_PACE_recon': data_dir + "Proxy_reconstructions/CESM1_PAC_PACE_recon_1800_2005/" +
                               "CESM1_PAC_PACE_recon_1800_2005_{}.nc",
        'CESM1_LENS_recon': data_dir + "Proxy_reconstructions/CESM1_LENS_recon_1800_2005/" +
                           "CESM1_LENS_recon_1800_2005_{}.nc",
        'CESM2_PAC_PACE_recon': data_dir + "Proxy_reconstructions/CESM2_PAC_PACE_recon_1800_2005/" +
                               "CESM2_PAC_PACE_recon_1800_2005_{}.nc",
        'CESM2_LENS_recon': data_dir + "Proxy_reconstructions/CESM2_LENS_recon_1800_2005/" +
                           "CESM2_LENS_recon_1800_2005_{}.nc",
    }
    
    if 'CESM2' in data_source and 'recon' not in data_source:
        cesm2_var_dict = { 'psl': 'PSL', 'tas': 'TS', 'u10': 'u1000', 'v10': 'v1000' }
        variable = cesm2_var_dict[variable]
    file_path = base_paths[data_source].format(variable)
    data = xr.open_dataset(file_path)[variable].sel(time=slice(time_period[0], time_period[1])).squeeze()

    return data

# Load data
psl = get_data(plot_data, 'psl', time_per)
u10 = get_data(plot_data, 'u10', time_per)
v10 = get_data(plot_data, 'v10', time_per)

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

#%% Make map ------------------------------------------------------

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
    cmap='BrBG_r', levels=np.arange(-4, 4.5, 0.5), extend='both')
# set colorbar
cb = plt.colorbar(c, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8, aspect=20)
cb.ax.tick_params(labelsize=12)
cb.set_label('SLP Trend (hPa/century)', fontsize=12)

# Plot wind trend quivers
# lat_density, lon_density, scale. larger density # means fewer arrows. larger quiver scale means
skip_dict = {'CESM1_PAC_PACE': (4, 12, 10), 
             'pace2_recon': (4, 12, 10)}
quiver_lat_density, quiver_lon_density, quiver_scale = (4,12,16)   # more arrows in latitude # larger number means fewer arrows #4 for pace em
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

# Plot box over Amundsen Sea Embayment in magenta
ase_box_lons = [-115, -100, -100, -115, -115]
ase_box_lats = [-76, -76, -70, -70, -76]
ax.plot(ase_box_lons, ase_box_lats, transform=ccrs.PlateCarree(), color='magenta', linewidth=2, zorder=5)


# CESM1 LENS historical n=40, CESM1 PAC PACE n=20
ens_num_dict = {
    'CESM1_PAC_PACE': 20,
    'CESM1_LENS_historical': 40,
    'CESM2_SSP585': 15,
    'CESM2_SSP370': 100,
    'CESM2_SSP245': 16}
if plot_data in ens_num_dict:
    plt.title(f"{plot_data} EM trends ({time_per[0]}-{time_per[1]}), n={ens_num_dict[plot_data]}",fontsize=16)
else:
    plt.title(f"{plot_data} trends ({time_per[0]}-{time_per[1]})",fontsize=16)
plt.tight_layout()
plt.show()
# %%
