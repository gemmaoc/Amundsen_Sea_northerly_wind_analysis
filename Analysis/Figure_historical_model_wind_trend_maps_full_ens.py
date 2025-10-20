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
plot_data = 'CESM1_PAC_PACE'
time_per = [1920,2005]  # period for trend calculation

#%% Get data-----------------------------------------------------------

data_dir = "/Users/gemma/Documents/Data/"
n_ens = 20
ens_ids = [f"{i:02d}" for i in range(1, n_ens + 1)]

# File templates
file_template = {
    'CESM1_PAC_PACE': {
        'psl': data_dir + "Model/Atmos/CESM1_PAC_PACE/annual_psl_PAC_PACE_ens_{ens}_1920_2005.nc",
        'u10': data_dir + "Model/Atmos/CESM1_PAC_PACE/annual_u10_PAC_PACE_ens_{ens}_1920_2005.nc",
        'v10': data_dir + "Model/Atmos/CESM1_PAC_PACE/annual_v10_PAC_PACE_ens_{ens}_1920_2005.nc"
    }
}

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

# Load all ensemble members
psl_trends = []
u10_trends = []
v10_trends = []
for ens in ens_ids:
    print(f"Loading ensemble member {ens}...")
    psl_path = file_template['CESM1_PAC_PACE']['psl'].format(ens=ens)
    u10_path = file_template['CESM1_PAC_PACE']['u10'].format(ens=ens)
    v10_path = file_template['CESM1_PAC_PACE']['v10'].format(ens=ens)
    psl = xr.open_dataset(psl_path)['psl']
    u10 = xr.open_dataset(u10_path)['u10']
    v10 = xr.open_dataset(v10_path)['v10']
    psl = psl / 100  # Convert SLP to hPa
    psl = psl.sel(time=slice(time_per[0], time_per[1])).squeeze()
    u10 = u10.sel(time=slice(time_per[0], time_per[1])).squeeze()
    v10 = v10.sel(time=slice(time_per[0], time_per[1])).squeeze()
    psl_trend = calc_trend(psl)
    u10_trend = calc_trend(u10)
    v10_trend = calc_trend(v10)
    lat_mask = psl_trend['lat'] <= cutoff_lat
    psl_trends.append(psl_trend.sel(lat=lat_mask))
    u10_trends.append(u10_trend.sel(lat=lat_mask))
    v10_trends.append(v10_trend.sel(lat=lat_mask))

#%% Make subplots for all ensemble members-------------------------------------------------
fig, axes = plt.subplots(4, 5, figsize=(10, 11),
                         subplot_kw={'projection': ccrs.SouthPolarStereo(central_longitude=-100)})
axes = axes.flatten()

for i in range(n_ens):
    print(i)
    ax = axes[i]
    ax.set_extent([-180, 180, -90, cutoff_lat], ccrs.PlateCarree())
    psl_trend_antarctic_cyclic, cyclic_lons = add_cyclic_point(psl_trends[i].values, coord=psl_trends[i]['lon'])
    u10_trend_antarctic_cyclic, _ = add_cyclic_point(u10_trends[i].values, coord=u10_trends[i]['lon'])
    v10_trend_antarctic_cyclic, _ = add_cyclic_point(v10_trends[i].values, coord=v10_trends[i]['lon'])
    lon2d, lat2d = np.meshgrid(cyclic_lons, psl_trends[i]['lat'])
    c = ax.contourf(
        lon2d, lat2d, psl_trend_antarctic_cyclic,
        transform=ccrs.PlateCarree(),
        cmap='BrBG_r', levels=np.arange(-3, 3.5, 0.5), extend='both')
    # Plot wind trend quivers
    skip = (slice(None, None, 4), slice(None, None, 12))
    Q = ax.quiver(
        lon2d[skip], lat2d[skip],
        u10_trend_antarctic_cyclic[skip], v10_trend_antarctic_cyclic[skip],
        transform=ccrs.PlateCarree(),
        scale=10, width=0.01, headwidth=3, headlength=4,
        headaxislength=3.5, color='k', zorder=3,
        edgecolor='white', linewidth=0.1,pivot='middle')
    # Circular boundary
    lons_bound = np.linspace(-180, 180, 1000)
    lats_bound = np.full_like(lons_bound, cutoff_lat)
    proj_pts = ax.projection.transform_points(ccrs.PlateCarree(), lons_bound, lats_bound)
    xs = proj_pts[:, 0]
    ys = proj_pts[:, 1]
    verts = np.column_stack([xs, ys])
    boundary_path = Path(verts)
    ax.set_boundary(boundary_path, transform=ax.transData)
    ax.patch.set_edgecolor("none")
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=4)
    ax.add_feature(cfeature.COASTLINE, linewidth=1, zorder=4)
    ax.plot(lons_bound, lats_bound, transform=ccrs.PlateCarree(), linewidth=2, color='k', zorder=10)
    ax.set_title(f"Ensemble {ens_ids[i]}", fontsize=14)

# Add colorbar
cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.02])
cb = fig.colorbar(c, cax=cbar_ax, orientation='horizontal')
cb.ax.tick_params(labelsize=12)
cb.set_label('SLP Trend (hPa/century)', fontsize=12)

plt.suptitle('CESM1_PAC_PACE Ensemble Member Trends (1920-2005)', fontsize=20)
plt.tight_layout(rect=[0, 0.1, 1, 0.97])
plt.show()

# %%
