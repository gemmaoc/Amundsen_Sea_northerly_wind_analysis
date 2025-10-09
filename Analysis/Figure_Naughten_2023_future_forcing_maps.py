#%%

import os
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.path as mpath

#%% Functions for loading data-----------------------------------------------------------------------
def load_data(base_dir, scenario):
    """
    Load temperature, u-wind, and v-wind trend files for a given scenario.
    """
    scenario_dir = os.path.join(base_dir, scenario)

    files = {
        'temp': [f for f in os.listdir(scenario_dir) if 'air_temperature' in f][0],
        'u': [f for f in os.listdir(scenario_dir) if 'eastward_wind' in f][0],
        'v': [f for f in os.listdir(scenario_dir) if 'northward_wind' in f][0],
        'sw_temp': [f for f in os.listdir(scenario_dir) if 'potential_temperature' in f][0]
    }

    temp_ds = xr.open_dataset(os.path.join(scenario_dir, files['temp']))
    u_ds = xr.open_dataset(os.path.join(scenario_dir, files['u']))
    v_ds = xr.open_dataset(os.path.join(scenario_dir, files['v']))
    sw_temp_ds = xr.open_dataset(os.path.join(scenario_dir, files['sw_temp']))

    return temp_ds, u_ds, v_ds, sw_temp_ds

def calc_on_shelf_trends(ds, vname):
    """
    Calculate trends on the continental shelf for a given variable, provided a 2d datasets 
    (e.g. spatial sw temp trends averaged over 200-700m, or spatial uwind trends).

    returns an xr data array of shape (n_ens,) with the average trend over the shelf box for each ens member
    """
    #get variable data
    var_data = ds[vname]

    # define box over which  to average
    lat1, lat2, lon1, lon2 = -70.8,-76,-115,-100 #naughten 2022 shelf box

    # get regional data
    reg_data = var_data.sel(Y=slice(lat2, lat1), X=slice(lon1, lon2))
    # mask out placeholder values greater than 1e36
    reg_data = reg_data.where(reg_data < 1e30)
    # take mean over region, skipping nans
    reg_avg = reg_data.mean(dim=['X','Y'], skipna=True) #shape (n_ens)
    # put in units of per cent
    reg_avg = reg_avg * 100

    return reg_avg
    

#%% Functions for plotting-----------------------------------------------------------------------

def plot_trends_single(temp_ds, u_ds, v_ds, scenario):
    """
    Plot separate temperature trend and wind trend (quivers) on a map for each ensemble member.
    Option to plot air or ocean temperature. 

    """
    # Access coordinates
    lon = temp_ds['X']
    lat = temp_ds['Y']

    # Determine available member indices
    available_members = temp_ds['time'].values

    # Helper function to plot for a single member
    def plot_for_member(idx):
        # Ensemble slice, and convert from per year to per century
        try:
            temp = temp_ds['air_temperature'].sel(time=idx) * 100
            temp_type = "Air"
        except:
            temp = temp_ds['sea_water_potential_temperature'].sel(time=idx) * 100
            temp_type = "Ocean"
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
        clevels = np.linspace(-3, 3, 17)
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

        plt.title(f"{scenario} - Ensemble Member {idx+1}: "+temp_type+" Temp and Wind Trends (2006–2100)")
        plt.tight_layout()
        plt.show()

    # Plot for all members 
    for idx in available_members:
        print(f"Plotting ensemble member {idx}...")
        plot_for_member(idx)

def plot_trends_multi(temp_ds, u_ds, v_ds, scenario, subtitles=None):
    """
    Plot temperature trend and wind trend (quivers) for all ensemble members
    in a single figure with subplots.

    Optional parameter 'subtitles' is a list of strings to use as subtitles for each subplot. Must match number of ensemble members.
    """
    lon = temp_ds['X'].values
    lat = temp_ds['Y'].values
    available_members = temp_ds['time'].values
    n_members = len(available_members)

    # Determine grid size for subplots (2 rows × 5 columns for 10 members)
    ncols = 2
    nrows = n_members // ncols + (n_members % ncols > 0)
    fig_height = nrows * 1.9
    fig = plt.figure(figsize=(6,fig_height))

    for i, idx in enumerate(available_members):
        ax = plt.subplot(nrows, ncols, i+1, projection=ccrs.SouthPolarStereo(central_longitude=-110)) 

        # Slice ensemble member and convert to per century
        try:
            temp = temp_ds['air_temperature'].sel(time=idx) * 100
            temp_type = "Air"
            # Temperature contour levels
            clevels = np.linspace(-4, 4, 17)
        except:
            temp = temp_ds['sea_water_potential_temperature'].sel(time=idx) * 100
            temp_type = "Ocean"
            # Temperature contour levels
            clevels = np.linspace(-2, 2, 17)
        u = u_ds['eastward_wind'].sel(time=idx) * 100
        v = v_ds['northward_wind'].sel(time=idx) * 100
        lon_np = np.array(lon)
        lat_np = np.array(lat)

        # Mask extreme values
        temp_masked = temp.where(temp < 1e30)
        u_masked = u.where(u < 1e30)
        v_masked = v.where(v < 1e30)

        # Contourf temperature trend
        cf = ax.contourf(lon, lat, temp_masked, levels=clevels, transform=ccrs.PlateCarree(),
                         cmap='coolwarm', extend='both')

        # Subsample wind vectors
        step = 40
        lon_sub = lon[::step]
        lat_sub = lat[::step]
        u_sub = u_masked.values[::step, ::step]
        v_sub = v_masked.values[::step, ::step]

        q = ax.quiver(lon_sub, lat_sub, u_sub, v_sub,
                  transform=ccrs.PlateCarree(),
                  scale=10, width=0.01,
                  headlength=4, headaxislength=4, minshaft=2,
                  pivot='middle', edgecolor='black', facecolor='black', zorder=200)
        if i == 9:
            ax.quiverkey(q, X=-0.07, Y=0.35, U=1,
                  label='1 m/s\nper century', labelpos='S', coordinates='axes', fontproperties={'size':10})

        # Add map features
        ax.coastlines()
        ax.add_feature(cfeature.LAND, zorder=100, facecolor='lightgray')
        ax.gridlines(draw_labels=False)

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

        if subtitles is not None:
            ax.set_title(subtitles[i], fontsize=10)
        else:
            ax.set_title(f"Member {idx+1}", fontsize=10)

    # Add a single colorbar for all subplots
    cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.02])  # [left, bottom, width, height]
    fig.colorbar(cf, cax=cbar_ax, orientation='horizontal', label="Temperature Trend [degK/cent]")

    plt.suptitle(f"{scenario} - "+temp_type+" Temperature and Wind Trends", fontsize=14)
    plt.subplots_adjust(hspace=0.25, wspace=0.1, top=0.92, bottom=0.09)
    if nrows < 5:
        plt.subplots_adjust(hspace=0.25, wspace=0.1, top=0.86, bottom=0.09)
    plt.show()

# %% Call plotting function -----------------------------------------------------------------------
scenarios = ['Historical','Paris1.5C', 'Paris2C',  'RCP4.5', 'RCP8.5']
scenario = input(f"Select a scenarior {scenarios}: ").strip()# Change this to select a different scenario
print(f"Selected scenario: {scenario}")

base_dir = '/Users/gemma/Documents/Data/Model/Ocean/Naughten_2023/'

# get atmos and ocean datasets
temp_ds, u_ds, v_ds, sw_temp_ds = load_data(base_dir, scenario)


# Plot trends with air temperature
# plot_trends_multi(temp_ds, u_ds, v_ds, scenario)

# plot trends with 200-700m seawater temperature
# calculate on-shelf trends for all ensemble members
shelf_sw_temp_trends = calc_on_shelf_trends(sw_temp_ds, 'sea_water_potential_temperature')
# calculate on-shelf northerly trends for all ensemble members
shelf_vwind_trends = calc_on_shelf_trends(v_ds, 'northward_wind')
subtitles = [
    f"Ens {i+1}\n{shelf_sw_temp_trends.values[i]:.2f}°C/cent; {shelf_vwind_trends.values[i]:.2f}m/s/cent"
    for i in range(len(shelf_sw_temp_trends))
    ]
# plot trend maps with on-shelf temp trends in subtitles
plot_trends_multi(sw_temp_ds, u_ds, v_ds, scenario, subtitles = subtitles)

# %% Plot trends for the ensemble mean of a given scenario------------------------------------------------------------------------

def plot_trends_ensemble_mean(temp_ds, u_ds, v_ds, scenario, trend_str=None):
    """
    Plot temperature trend and wind trend (quivers) for the ensemble mean of a given scenario.
    """
    lon = temp_ds['X'].values
    lat = temp_ds['Y'].values

    # Calculate ensemble mean and convert to per century
    try:
        temp_mean = temp_ds['air_temperature'].mean(dim='time') * 100
        temp_type = "Air"
        # Temperature contour levels
        clevels = np.linspace(-4, 4, 17)
    except:
        temp_mean = temp_ds['sea_water_potential_temperature'].mean(dim='time') * 100
        temp_type = "Ocean"
        # Temperature contour levels
        clevels = np.linspace(-1, 1, 17)
    u_mean = u_ds['eastward_wind'].mean(dim='time') * 100
    v_mean = v_ds['northward_wind'].mean(dim='time') * 100

    # Mask extreme values
    temp_masked = temp_mean.where(temp_mean < 1e30)
    u_masked = u_mean.where(u_mean < 1e30)
    v_masked = v_mean.where(v_mean < 1e30)

    # Convert masked data to numpy arrays for plotting
    lon_np = np.array(lon)
    lat_np = np.array(lat)
    temp_np = np.array(temp_masked)
    u_np = np.array(u_masked)
    v_np = np.array(v_masked)

    # Set up plot
    fig = plt.figure(figsize=(8,5))
    ax = plt.axes(projection=ccrs.SouthPolarStereo(central_longitude=-110))

    # Set extent to match the data subregion
    lon_min = np.nanmin(lon_np)
    lon_max = np.nanmax(lon_np)
    lat_min = np.nanmin(lat_np)
    lat_max = np.nanmax(lat_np)
    # lon_min, lon_max, lat_min, lat_max =-120,-95,-70,-76  # For ASE domain
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
    ax.gridlines(draw_labels=False)
    cf = ax.contourf(lon_np, lat_np, temp_np, 
                     levels=clevels, transform=ccrs.PlateCarree(), 
                     cmap='coolwarm', extend='both') 
    # Subsample data to reduce arrow density
    step = 60  # Increase for fewer arrows
    lon_sub = lon_np[::step]
    lat_sub = lat_np[::step]
    u_sub = u_np[::step, ::step]
    v_sub = v_np[::step, ::step]
    # Quiver plot for masked wind trends
    q = ax.quiver(
        lon_sub, lat_sub, u_sub, v_sub,
        transform=ccrs.PlateCarree(),
        scale=10, width=0.01,#scale 10 for full model domain, 3 for ASE
        headlength=4, headaxislength=4, minshaft=2,
        pivot='middle', edgecolor='black', facecolor='black',
        zorder=200
    )
    ax.quiverkey(q, X=0.09, Y=0.25, U=0.4,
                  label='0.4 m/s\nper century', labelpos='S', 
                  coordinates='axes', fontproperties={'size':10})
    # Colorbar
    cbar = plt.colorbar(cf, orientation='horizontal', pad=0.05,
                        shrink=0.7)
    cbar.set_label("Temperature Trend [degK/cent]")
    plt.title(f"{scenario} Ensemble Mean: " + temp_type + " Temp and Wind Trends\n"+trend_str)
    # plt.tight_layout()
    plt.show()

#%% -----------------------------------------------------------------------
scenario = input(f"Select a scenarior {scenarios}: ").strip()# Change this to select a different scenario
print(f"Selected scenario: {scenario}")

# get atmos and ocean datasets
temp_ds, u_ds, v_ds, sw_temp_ds = load_data(base_dir, scenario)

# calculate on-shelf trends for all ensemble members
shelf_sw_temp_trends = calc_on_shelf_trends(sw_temp_ds, 'sea_water_potential_temperature')

# calculate on-shelf northerly trends for all ensemble members
shelf_vwind_trends = calc_on_shelf_trends(v_ds, 'northward_wind')

#calc ens mean trends
ens_mean_sw_temp_trend = shelf_sw_temp_trends.mean().values
ens_mean_vwind_trend = shelf_vwind_trends.mean().values
trend_str = f"On-shelf SW temp trend = {ens_mean_sw_temp_trend:.2f}°C/cent; On-shelf Vwind trend = {ens_mean_vwind_trend:.2f}m/s/cent"


plot_trends_ensemble_mean(sw_temp_ds, u_ds, v_ds, scenario, trend_str = trend_str)


# %%
