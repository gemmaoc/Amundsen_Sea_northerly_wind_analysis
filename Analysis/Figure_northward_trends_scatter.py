#%%

import os
from scipy import stats
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

#%% USER SETTINGS
base_dir = "../Data/Naughten_2023/"  # directory containing subdirs for each scenario

regions = {
    "Amundsen_Sea": [-115, -100, -76, -70], 
    "ASE_shelf_break": [-115, -102, -72, -70],  # Amundsen Sea shelf break
}
var_map = {
    "temp": "temperature",
    "u": "eastward_wind",
    "v": "northward_wind"
}


#%%
def load_and_avg(scenario_dir, variable, region_bounds):
    """Load all ensemble members for a variable and average over region."""
    fname = [f for f in os.listdir(scenario_dir) if var_map[variable] in f][0]
    ds = xr.open_dataset(os.path.join(scenario_dir, fname))

    # Clip to region
    lon1, lon2, lat1, lat2 = region_bounds
    ds_region = ds.sel(X=slice(lon1, lon2), Y=slice(lat1, lat2))
    print(ds_region)

    # mask large values (placeholders for Nans where boundaries are)
    ds_region_masked = ds_region.where(ds_region[var_map[variable]] < 1e35)
    # Average spatially over region (shape (10 members))
    members = ds_region_masked[var_map[variable]].mean(dim=["X", "Y"], skipna=True)

    # calculate ensemble mean trend
    ens_mean = members.mean()

    return members, ens_mean

#%%
# Get proxy reconstruction northward wind trends
def get_proxy_trends(recon, region_bounds):
    """Load proxy northward wind trends."""

    recon_dict = {'cesm1_lme': 'iCESM_LME_GKO1_linPSM_1mc_1800_2005_v10_',
                  'cesm1_lens': 'LENS_super_GKO1_all_linPSM_1mc_1800_2005_GISBrom_1880_2019_',
                  'cesm1_pace': 'PACE_super_GKO1_all_linPSM_1mc_1900_2005_GISBrom_1880_2019_v10_sst_',
                  'cesm2_lens': 'LENS2_super_GKO1_all_bilinPSM_1mc_1850_2005_GISBrom_',
                  'cesm2_pace': 'PAC_PACE2_super_GKO1_all_bilinPSM_1mc_1850_2005_GISBrom_'}
    proxy_dir = "../../../Research/LMR_analysis/LMR_output/"
    proxy_file = proxy_dir + recon_dict[recon] +  "v10.nc"

    ds_proxy = xr.open_dataset(proxy_file)
    ds_proxy = ds_proxy.squeeze()
    lon1, lon2, lat1, lat2 = region_bounds
    ds_proxy_reg = ds_proxy.sel(time=slice(1920,2005),lat=slice(lat1,lat2), 
                            lon=slice((lon1 + 360) % 360, (lon2 + 360) % 360))
    # Assuming the dataset has a variable named 'northward_wind_trend'
    ds_proxy_reg_avg = ds_proxy_reg['v10'].mean(dim=['lat', 'lon'], skipna=True)
    proxy_trend = stats.linregress(ds_proxy_reg_avg['time'], ds_proxy_reg_avg.values)
    
    trend_per_cent = proxy_trend[0] * 100  # Convert to m/s per century

    return trend_per_cent

#%%
def main():
    # ---- USER INPUT ----
    variable = input(f"Select variable {list(var_map.keys())}: ").strip()
    # variable = "u"
    region_name = input(f"Select region {list(regions.keys())}: ").strip()
    # region_name = "Amundsen_Sea" #for on-shelf
    # region_name = 'ASE_shelf_break'  
    region_bounds = regions[region_name]
    # --------------------

    scenarios = ['Historical', 'Paris1.5C', 'Paris2C', 'RCP4.5', 'RCP8.5']
    all_data = {}
    ens_mean_trends = {}
    for scenario in scenarios:
        scenario_path = os.path.join(base_dir, scenario)
        print(scenario)
        members, ens_mean = load_and_avg(scenario_path, variable, region_bounds)
        all_data[scenario] = members*100 # convert to m/s/cent
        ens_mean_trends[scenario] = ens_mean * 100 # convert to m/s/cent

    proxy_trends = {}
    recons = ['cesm1_lme', 'cesm1_lens', 'cesm2_lens', 'cesm2_pace'] #'cesm1_pace', 
    for recon in recons:
        proxy_trend = get_proxy_trends(recon, region_bounds)
        proxy_trends[recon] = proxy_trend
        print(f"{recon} proxy trend: {proxy_trend:.2f} m/s per century")

    # ---- PLOT ----
    reds = plt.cm.YlOrRd(np.linspace(0.4, 0.9, 4))
    model_colors = {
        "Historical": "darkgray",
        "Paris1.5C": reds[0],
        "Paris2C": reds[1],
        "RCP4.5": reds[2],
        "RCP8.5": reds[3]
    }


    # pick evenly spaced shades between light gray and black
    grayscale = plt.cm.Blues(np.linspace(0.3, 0.9, len(recons)))

    proxy_colors = {name: grayscale[i] for i, name in enumerate(recons)}

    plt.figure(figsize=(8, 6))

    # Plot proxy trends
    # Plot proxy values, each with its own legend entry
    for recon, trend in proxy_trends.items():
        print(recon, trend, proxy_colors[recon])
        plt.scatter(["Historical"], [trend], 
                color=proxy_colors[recon], label=recon+' reconstruction', 
                s=150, edgecolor="k", zorder=4, linewidths=2)

    for scen_key, members in all_data.items():
        mean_val = ens_mean_trends[scen_key].item()
        # Plot ensemble mean as solid color scatter
        plt.scatter([scen_key], [mean_val], color=model_colors[scen_key], 
                    s=150, label=f"{scen_key} simulations", edgecolor='k', zorder=4,
                    linewidths=2)
        # Plot other ensemble members as shaded scatter
        other_members = members.values
        plt.scatter([scen_key]*len(other_members), other_members, 
                    color=model_colors[scen_key], alpha=0.5, s=150, zorder=3)

    
    
    plt.ylabel(f"{var_map[variable].replace('_', ' ').title()} Trend (m/s per century)")
    plt.title(f"Ensemble Trends for {region_name} - {var_map[variable].replace('_', ' ').title()}")
    plt.grid(True, axis='y')
    plt.axhline(0, color='k', linestyle='--', linewidth=2,zorder=2)
    plt.rcParams.update({'font.size': 14,
                         'legend.fontsize': 9,
                         'axes.titlesize': 14,
                         'axes.labelsize': 14})
    plt.tight_layout()

    # Only show one mean label per scenario
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),ncol=2, loc='lower right')

    plt.show()

if __name__ == "__main__":
    main()

# %%
