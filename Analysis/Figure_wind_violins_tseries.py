#%%

import os
from scipy import stats
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

#%% USER SETTINGS------------------------------------------------------

vname = 'us'  # variable name: 'vs' for northward wind, 'us' for eastward wind, 'tas' for temperature
region = 'Amundsen_Sea_shelf'


#%% Define data directories and regions--------------------------------------

base_dir = "/glade/campaign/univ/uwas0134/Gemma/Data/"

regions = {
    "Amundsen_Sea_shelf": [-115, -100, -76, -70], 
    "ASE_shelf_break": [-115, -102, -72, -70], 
}


#%% define func to get ensemble trends for a single variable and single simulation------------------------------------

def load_sim_ensemble(sim, variable, region_bounds):
    """
    Load ensemble climate variable timeseries and trends for a single simulation.
    
    Parameters:
    sim (str): Simulation name (e.g., 'CESM2_LENS').
    variable (str): Variable name (e.g., 'V').
    region_bounds (list): List of [lon1, lon2, lat1, lat2] defining the region.

    Returns:
    ensemble_tseries (xr data array): Array of annual mean timeseries for each ensemble member.
    ensemble_trends (np.array): Array of annunal-mean trends for each ensemble member.
    ensemble_pvales (np.array): Array of p-values for each ensemble member's trend
    """

    sim_dict = {'CESM2_SSP370': 'CESM2_SSP370/',
                    'CESM2_SSP245': 'CESM2_SSP245/',
                    'CESM2_SSP585': 'CESM2_SSP585/'}
    # var_map shows variable directory and variable name in netCDF file
    var_map = {
                "tas": ["TS",'TS'],
                "us": ["U", 'u1000'],
                "vs": ["V", 'v1000']
            }
    sim_dir = base_dir + 'Model/' + sim_dict[sim] + var_map[variable][0] + "/"
    print(sim_dir)

    # Get all .nc files in the directory
    nc_files = [f for f in os.listdir(sim_dir) if f.endswith('.nc') and var_map[variable][1] in f]
    print(f"Found {len(nc_files)} ensemble files for simulation {sim} and variable {variable}.")

    # Load all files and collect variable data

    # get number of years from first file name (fname format example: '*v1000.201501-210012.nc')
    f0 = nc_files[0]
    start_yr = f0.split(var_map[variable][1])[1][1:5]
    end_yr = f0.split(var_map[variable][1])[1][8:12]
    n_yrs = int(end_yr) - int(start_yr) + 2 # +2 to account for inclusive range and resampling to year end
    ensemble_tseries = np.zeros((len(nc_files), n_yrs))  # 87 years from 2015 to 2100 in CESM2 sims. will need to change for CESM1 MENS which stops in 2080
    ensemble_trends = np.zeros(len(nc_files))
    ensemble_pvalues = np.zeros(len(nc_files))

    for i, fname in enumerate(nc_files):

        ds = xr.open_dataset(os.path.join(sim_dir, fname))
        var_data = ds[var_map[variable][1]]
        
        # Clip to region
        lon1, lon2, lat1, lat2 = region_bounds
        if ds.lon.max() > 180:
            lon1 = (lon1 + 360) % 360
            lon2 = (lon2 + 360) % 360
        data_region = var_data.sel(lon=slice(lon1, lon2), lat=slice(lat1, lat2))
        
        # Average spatially over region for this member
        member_avg = data_region.mean(dim=["lat", "lon"], skipna=True)

        # Calculate annual mean
        member_avg = member_avg.resample(time='YE').mean()


        # Calculate trend in per century
        trend = stats.linregress(member_avg['time'].dt.year[0:-1], member_avg.values[0:-1]) #exclude last yr bc unusually large..LOOK INTO
        trend_per_cent = trend.slope * 100  # Convert to m/s per century
        trend_pval = trend.pvalue

        # Store in arrays
        ensemble_tseries[i, :] = member_avg
        ensemble_trends[i] = trend_per_cent
        ensemble_pvalues[i] = trend_pval
        ds.close()

    print(f"Ensemble mean trend: {np.mean(ensemble_trends):.2f} m/s per century, "
        f"std dev = {np.std(ensemble_trends, ddof=1):.2f} m/s per century")

    return ensemble_tseries, ensemble_trends, ensemble_pvalues



#%% Define func to load proxy recon trends-----------------------------------------------------

def get_proxy_trends(recon, variable, region_bounds, time_per):
    """Load proxy northward wind trends."""

    recon_dict = {'cesm2_lens': 'CESM2_LENS_recon_1850_2005/LENS2_super_GKO1_all_bilinPSM_1mc_1850_2005_GISBrom_',
                  'cesm2_pace': 'CESM2_PAC_PACE_recon_1850_2005/PAC_PACE2_super_GKO1_all_bilinPSM_1mc_1850_2005_GISBrom_'}
    proxy_dir = base_dir + "Proxy_reconstructions/"
    proxy_file = proxy_dir + recon_dict[recon] +  variable + ".nc"

    ds_proxy = xr.open_dataset(proxy_file)
    ds_proxy = ds_proxy.squeeze()
    lon1, lon2, lat1, lat2 = region_bounds
    start,stop = time_per
    ds_proxy_reg = ds_proxy.sel(time=slice(start,stop),lat=slice(lat1,lat2), 
                            lon=slice((lon1 + 360) % 360, (lon2 + 360) % 360))
    # Assuming the dataset has a variable named 'northward_wind_trend'
    proxy_reg_avg = ds_proxy_reg[variable].mean(dim=['lat', 'lon'], skipna=True)
    proxy_trend = stats.linregress(proxy_reg_avg['time'], proxy_reg_avg.values)

    trend_per_cent = proxy_trend.slope * 100  # Convert to m/s per century
    trend_pval = proxy_trend.pvalue

    return proxy_reg_avg, trend_per_cent, trend_pval

#%% Load sim data-----------------------------------------------------

sims = ['CESM2_SSP370', 'CESM2_SSP245', 'CESM2_SSP585']
sim_tseries_dict = {}
sim_trends_dict = {}
sim_pvals_dict = {}
for sim in sims:
    tseries, trends, pvals = load_sim_ensemble(sim, vname, regions[region])
    sim_tseries_dict[sim] = tseries
    sim_trends_dict[sim] = trends
    sim_pvals_dict[sim] = pvals


#%% Load proxy trends -----------------------------------------------------

# (note this populates an array, which will be turned into a single proxy recon list for plotting)

recons = ['cesm2_lens','cesm2_pace']
time_per = [1900,2005]
recon_tseries_list = [] #list of 1d data arrays with time and var data
recon_trends_arr = np.zeros(len(recons))
recon_pvals_arr = np.zeros(len(recons))
for i in range(len(recons)):
    tseries, trend, pval = get_proxy_trends(recons[i], vname, regions[region], time_per)
    recon_tseries_list.append(tseries)
    recon_trends_arr[i] = trend
    recon_pvals_arr[i] = pval

#%% ---- PLOT ----

# Specify the order of simulations for plotting
sim_order = ['CESM2_SSP245', 'CESM2_SSP370', 'CESM2_SSP585']

plot_data = [recon_trends_arr] + [sim_trends_dict[sim] for sim in sim_order]

# choose colors
reds = plt.cm.YlOrRd(np.linspace(0.4, 0.9, 4))
colors = ['darkgray',reds[1],reds[2],reds[3]]

# set up plot
fig, ax = plt.subplots(figsize=(8, 6))

x_positions = np.arange(len(plot_data))
ax.set_xticks(x_positions)
x_labels = ['Proxy recons\n1900-2005'] + sim_order
ax.set_xticklabels(x_labels)

parts = ax.violinplot(plot_data, positions=x_positions, widths=0.5, showmeans=False, showmedians=False,
                        showextrema=False)
for pc, color in zip(parts['bodies'], colors):
    pc.set_facecolor(color)
    pc.set_edgecolor('black')
    pc.set_alpha(0.7)
    pc.set_linewidth(1.5)
        

# Set style of violin plots
for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
    if partname in parts:
        vp = parts[partname]
        if isinstance(vp, list):  
            for line in vp:
                line.set_color("black")
                line.set_linewidth(1.5)
        else:  # single LineCollection
            vp.set_color("black")
            vp.set_linewidth(2)

# annotate the number of ensemble members/proxy reconstructions
for i, data in enumerate(plot_data):
    n = len(data)
    ax.text(x_positions[i], np.mean(data),
            f'n={n}', ha='center', va='center', fontsize=12)

# Adjust plot aesthetics
plt.axhline(0, color='k', linestyle='--', linewidth=2,zorder=0)
plt.axvline(0.5, color='gray', linestyle='--', linewidth=2,zorder=0)
plt.grid(True, axis='y')

plt.title(f'{vname} trends over {region.replace("_", " ")} (2015-2100)')
plt.ylabel(vname + " trend (m/s per century)")
plt.rcParams.update({'font.size': 14,
                        'axes.titlesize': 14,
                        'axes.labelsize': 14})
plt.tight_layout()

plt.show()


#%% Plot time series -----------------------------------------------------

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)

# plot proxy recons
ax.plot(recon_tseries_list[0]['time'], recon_tseries_list[0], color='darkgray',
        label=f'CESM2 LENS Proxy Recon\n{np.mean(recon_trends_arr[0]):.2f} m/s/cent, p = {np.mean(recon_pvals_arr[0]):.3f}')
ax.plot(recon_tseries_list[1]['time'], recon_tseries_list[1], color='black',
        label=f'CESM2 PACE Proxy Recon\n{np.mean(recon_trends_arr[1]):.2f} m/s/cent, p = {np.mean(recon_pvals_arr[1]):.3f}')

# Plot sim ensemble means
for sim, color in zip(sim_order, reds[1:]):

    sim_mean = np.mean(sim_tseries_dict[sim], axis=0)
    years = np.arange(2015, 2015 + sim_mean.shape[0])

    #offset timeseries so that the first year matches the proxy recon end year (2005)
    recon_mean_2005 = np.mean([recon_tseries_list[0].sel(time=2005).values,
                                        recon_tseries_list[1].sel(time=2005).values])
    offset = recon_mean_2005 - sim_mean[0]
    sim_mean += offset
    lab = f"{sim} Ensemble Mean\n{np.mean(sim_trends_dict[sim]):.2f} m/s/cent, p = {np.mean(sim_pvals_dict[sim]):.3f}"
    ax.plot(years[:-1], sim_mean[:-1], label=lab, color=color)

    # # plot interquartile range shading
    sim_lower = np.percentile(sim_tseries_dict[sim], 25, axis=0) + offset
    sim_upper = np.percentile(sim_tseries_dict[sim], 75, axis=0) + offset
    ax.fill_between(years[:-1], sim_lower[:-1], sim_upper[:-1], color=color, alpha=0.3)#,
                    # label=f'{sim} Interquartile Range')

    # print trend, p-val, and std dev of timeseries
    print(f"{sim} trend: {np.mean(sim_trends_dict[sim]):.2f} m/s/cent, p-val: {np.mean(sim_pvals_dict[sim]):.3f}")

# Adjust plot aesthetics
plt.grid(True, axis='y')
plt.xlabel('Year')
plt.ylabel(vname + " anomaly (m/s)")
plt.title(f'{vname} over {region.replace("_", " ")}')
plt.legend(loc='lower left',ncol=2)
plt.rcParams.update({'font.size': 14,
                        'axes.titlesize': 14,
                        'axes.labelsize': 14,
                        'legend.fontsize': 10})
plt.tight_layout()
plt.show()



# %%
