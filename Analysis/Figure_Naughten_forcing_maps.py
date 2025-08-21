#%%

import os
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.path as mpath

#%%
def load_data(base_dir, scenario):
    """
    Load temperature, u-wind, and v-wind trend files for a given scenario.
    """
    scenario_dir = os.path.join(base_dir, scenario)

    files = {
        'temp': [f for f in os.listdir(scenario_dir) if 'temperature' in f][0],
        'u': [f for f in os.listdir(scenario_dir) if 'eastward_wind' in f][0],
        'v': [f for f in os.listdir(scenario_dir) if 'northward_wind' in f][0]
    }

    temp_ds = xr.open_dataset(os.path.join(scenario_dir, files['temp']))
    u_ds = xr.open_dataset(os.path.join(scenario_dir, files['u']))
    v_ds = xr.open_dataset(os.path.join(scenario_dir, files['v']))

    return temp_ds, u_ds, v_ds

def plot_trends(temp_ds, u_ds, v_ds, scenario, member_idx=None):
    """
    Plot temperature trend and wind trend (quivers) on a map for selected ensemble member(s).
    If member_idx is None, plot for all available ensemble members in the dataset.
    """
    # Access coordinates
    lon = temp_ds['X']
    lat = temp_ds['Y']

    # Determine available member indices
    available_members = temp_ds['time'].values

    # Helper function to plot for a single member
    def plot_for_member(idx):
        # Ensemble slice, and convert from per year to per century
        temp = temp_ds['air_temperature'].sel(time=idx) * 100
        u = u_ds['eastward_wind'].sel(time=idx) * 100
        v = v_ds['northward_wind'].sel(time=idx) * 100

        # Mask data where values are greater than 1e30 (placeholders for ice shelves)
        temp_masked = temp.where(temp < 1e30)
        u_masked = u.where(u < 1e30)
        v_masked = v.where(v < 1e30)
        
        # Convert masked data to numpy arrays for plotting
        lon_np = np.array(lon)
        lat_np = np.array(lat)
        temp_np = np.array(temp_masked)
        u_np = np.array(u_masked)
        v_np = np.array(v_masked)

        # Set up plot
        fig = plt.figure(figsize=(8,7))
        ax = plt.axes(projection=ccrs.SouthPolarStereo(central_longitude=-110))

        # Set extent to match the data subregion
        lon_min = np.nanmin(lon_np)
        lon_max = np.nanmax(lon_np)
        lat_min = np.nanmin(lat_np)
        lat_max = np.nanmax(lat_np)
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

        # Set boundary to match data shape
        lon1, lon2, lat1, lat2 = lon_min, lon_max, lat_min, lat_max
        rect = mpath.Path([[lon1, lat2], [lon2, lat2],[lon2, lat1], [lon1, lat1], [lon1, lat2]]).interpolated(50)
        proj_to_data = ccrs.PlateCarree()._as_mpl_transform(ax) - ax.transData
        rect_in_target = proj_to_data.transform_path(rect)
        ax.set_boundary(rect_in_target)
        ax.set_xlim(rect_in_target.vertices[:,0].min(), rect_in_target.vertices[:,0].max())
        ax.set_ylim(rect_in_target.vertices[:,1].min(), rect_in_target.vertices[:,1].max())

        # Add features
        ax.coastlines()
        ax.add_feature(cfeature.LAND, zorder=100, facecolor='lightgray')
        ax.gridlines(draw_labels=True)

        # Contourf plot for masked temperature trend
        clevels = np.linspace(-3, 3, 21)
        cf = ax.contourf(lon_np, lat_np, temp_np, levels=clevels, transform=ccrs.PlateCarree(), cmap='coolwarm', extend='both')

        # Subsample data to reduce arrow density
        step = 40  # Increase for fewer arrows
        lon_sub = lon_np[::step]
        lat_sub = lat_np[::step]
        u_sub = u_np[::step, ::step]
        v_sub = v_np[::step, ::step]

        # Quiver plot for masked wind trends
        ax.quiver(
            lon_sub, lat_sub, u_sub, v_sub,
            transform=ccrs.PlateCarree(),
            scale=10, width=0.01,
            headlength=4, headaxislength=4, minshaft=2,
            pivot='middle', edgecolor='black', facecolor='black',
            zorder=200
        )

        # Colorbar
        cbar = plt.colorbar(cf, orientation='horizontal', pad=0.05, shrink=0.7)
        cbar.set_label("Temperature Trend [degK/cent]")

        plt.title(f"{scenario} - Ensemble Member {idx}: Temp and Wind Trends (2006–2100)")
        plt.tight_layout()
        plt.show()

    # Plot for all members if member_idx is None
    if member_idx is None:
        for idx in available_members:
            print(f"Plotting ensemble member {idx}...")
            plot_for_member(idx)
    else:
        # Accept either index or member number (1-based)
        if isinstance(member_idx, int) and member_idx in available_members:
            plot_for_member(member_idx)
        elif isinstance(member_idx, int) and 1 <= member_idx <= len(available_members):
            plot_for_member(available_members[member_idx - 1])
        else:
            raise ValueError("Invalid ensemble member index.")

# %%
scenarios = ['Paris1.5C', 'Paris2C',  'RCP4.5', 'RCP8.5']
scenario = scenarios[0]  # Change this to select a different scenario
print(f"Selected scenario: {scenario}")

base_dir = '../Data/Naughten_2023/'
temp_ds, u_ds, v_ds = load_data(base_dir, scenario)
plot_trends(temp_ds, u_ds, v_ds, scenario, member_idx=None)

# %%
