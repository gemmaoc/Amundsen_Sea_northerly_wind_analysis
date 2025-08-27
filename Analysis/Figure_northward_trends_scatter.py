#%%

import os
from scipy import stats
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

#%% USER SETTINGS
base_dir = "Data/Naughten_2023/"  # directory containing subdirs for each scenario

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
def load_model_ens(scenario_dir, variable, region_bounds):
    """
    Load all ensemble members for a variable and average over region.
    Returns:
        members: xarray DataArray of shape (n_members,)
        ens_mean: float, ensemble mean trend
    """

    print(os.getcwd())
    fname = [f for f in os.listdir(scenario_dir) if var_map[variable] in f][0]
    ds = xr.open_dataset(os.path.join(scenario_dir, fname))

    # Clip to region
    lon1, lon2, lat1, lat2 = region_bounds
    ds_region = ds.sel(X=slice(lon1, lon2), Y=slice(lat1, lat2))

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

    trend_per_cent = proxy_trend.slope * 100  # Convert to m/s per century
    trend_pval = proxy_trend.pvalue

    return trend_per_cent, trend_pval


#%%
def main():
    # ---- USER INPUT ----
    variable = input(f"Select variable {list(var_map.keys())}: ").strip()
    # variable = "v"
    region_name = input(f"Select region {list(regions.keys())}: ").strip()
    # region_name = "Amundsen_Sea" #for on-shelf
    # region_name = 'ASE_shelf_break'  
    region_bounds = regions[region_name]
    bias_correction = input("Apply bias correction? (y/n): ").strip().lower() == 'y'
    # --------------------

    # Load simulated data
    scenarios = ['Historical', 'Paris1.5C', 'Paris2C', 'RCP4.5', 'RCP8.5']
    sim_data = {}
    for scenario in scenarios:
        scenario_path = os.path.join(base_dir, scenario)
        print(scenario)
        members, ens_mean = load_model_ens(scenario_path, variable, region_bounds)
        sim_data[scenario] = members*100 # convert to m/s/cent

    # Load proxy data
    proxy_trends = {}
    recons = ['cesm1_lme', 'cesm1_lens', 'cesm2_lens', 'cesm2_pace'] #'cesm1_pace', 
    for recon in recons:
        proxy_trend, pval = get_proxy_trends(recon, region_bounds)
        proxy_trends[recon] = proxy_trend
        print(f"{recon} proxy trend: {proxy_trend:.2f} m/s per century, p-value: {pval:.3f}")

    # propogate data for easy plotting: proxy_data then simulated data
    proxy_values = list(proxy_trends.values())
    sim_values = [list(sim_data[scen].values) for scen in sim_data]
    plot_data = [proxy_values] + sim_values

    # calculate means and standard errors for each group
    means = [] # order is proxy, historical, paris1.5, paris2, rcp4.5, rcp8.5
    std_errs = []
    for i, values in enumerate(plot_data, 0):
        mean = np.mean(values)
        se = np.std(values, ddof=1) / np.sqrt(len(values))
        means.append(mean)
        std_errs.append(se)

    # calculate bias and uncertainty from historical sim and proxy
    bias = means[1]- means[0]  # model historical - proxy mean
    print(f"Bias (Hist model - Proxies) = {bias:.2f}")


    # ---- PLOT ----

    # choose colors
    reds = plt.cm.YlOrRd(np.linspace(0.4, 0.9, 4))
    colors = ['tab:blue','darkgray',reds[0],reds[1],reds[2],reds[3]]

    # set up plot
    fig, ax = plt.subplots(figsize=(8, 6))

    x_positions = np.arange(len(scenarios) + 1)
    ax.set_xticks(x_positions)
    x_labels = ['Proxies'] + scenarios
    ax.set_xticklabels(x_labels)

    parts = ax.violinplot(plot_data, positions=x_positions, widths=0.5, showmeans=False, showmedians=False,
                          showextrema=False)
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
        pc.set_linewidth(1.5)

    # plot mean + standard error
    for i, values in enumerate(plot_data, 0):
        mean, se = means[i], std_errs[i]

        # Plot mean as a black dot
        ax.scatter(i, mean, color="k", zorder=3)
        # Plot error bar for standard error
        ax.errorbar(i, mean, yerr=se, color="k", capsize=5, linewidth=1.5, zorder=2)

        # Print them out
        print(f"{x_labels[i]}: mean = {mean:.2f}, SE = {se:.2f}")
        # Add text annotation above violin
        ax.text(i, max(values) + 0.01, f"{mean:.2f} ± {se:.2f}",
            ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        if bias_correction and i > 1:

            corrected_trend = mean - bias
            sim_hist_se = std_errs[1]  # standard error of historical sim
            proxy_hist_se = std_errs[0]  # standard error of historical proxy
            # propagate uncertainties in quadrature: sqrt(se_future^2 + se_sim_hist^2 + se_proxy_hist^2)
            corrected_se = np.sqrt(se**2 + sim_hist_se**2 + proxy_hist_se**2)

            # plot bias-corrected mean + propagated uncertainty
            ax.scatter(i, corrected_trend, color="c", zorder=3)
            ax.errorbar(i, corrected_trend, yerr=corrected_se, color="c", capsize=5, linewidth=2, zorder=2)
            # Add text annotation below corrected error bar
            ax.text(i, corrected_trend - corrected_se - 0.01, f"{corrected_trend:.2f} ± {corrected_se:.2f}",
                ha='center', va='top', fontsize=10, color='k', fontweight='bold')
            

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
    
    # Adjust plot aesthetics
    plt.axhline(0, color='k', linestyle='--', linewidth=2,zorder=2)
    plt.axvline(1.5, color='gray', linestyle='--', linewidth=1)
    plt.grid(True, axis='y')

    ylim_dict = {'v': [-0.7, 0.5], 'u': [-1,1], 'temp': [-2,2]}
    plt.ylim(ylim_dict[variable])
    plt.ylabel(f"{var_map[variable].replace('_', ' ').title()} Trend (m/s per century)")
    plt.title(f"Ensemble Trends for {region_name} - {var_map[variable].replace('_', ' ').title()}")
    plt.rcParams.update({'font.size': 14,
                         'axes.titlesize': 14,
                         'axes.labelsize': 14})
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()

# %%
