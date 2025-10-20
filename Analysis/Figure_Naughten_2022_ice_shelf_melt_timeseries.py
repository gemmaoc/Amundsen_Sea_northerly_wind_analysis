#%% Load packages
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats


#%% User parameters--------------------------------------------------------------
data_dir = '/Users/gemma/Documents/Data/Model/Ocean/Naughten_2022/'

warm_composite_members = [1,11,15,17] #from O'Connor et al., 2025, determined by on-shelf CDW trends
# based on ice shelf melt, the warmest members are 15, 17, 2, 12

#%% Load Naughten 2022 data-------------------------------------------------------------

# # Load Naughten ice shelf flux ensemble timeseries
fname = 'MITgcm_IceShelvesFromDotsonToCosgrove_1920-2013_ice_shelf_basal_melt_flux_PACE'
ens_list = [f'{i:02}' for i in range(1, 21)]
n_ens = len(ens_list)

# Load times for plotting all members
ds = xr.open_dataset(data_dir + fname + ens_list[0]+ '.nc')
ens_ds = xr.open_dataset(data_dir + fname + ens_list[0] + '.nc')
ens_melt = ds.ice_shelf_basal_melt_flux
time = ens_melt.time.values
# Convert cftime.DatetimeNoLeap to float years
years = np.array([t.year + (t.month - 1) / 12 for t in time])

#%% Calc trends and plot full ensemble of ice shelf flux timeseries----------------------------------------------------

plt.figure(figsize=(12,6))
cent_melt_trends = []

for i in range(n_ens):
    ens_ds = xr.open_dataset(data_dir + fname + ens_list[i] + '.nc')
    ens_melt = ens_ds.ice_shelf_basal_melt_flux

    # apply 2-year running mean for plotting
    ens_melt_rm = ens_melt.rolling(time=24, center=True).mean()

    plt.plot(years, ens_melt_rm, label=ens_list[i],color='tab:blue')
    # calc trend
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, ens_melt)
    print(ens_list[i] + ': {:.2f} Gt/yr/cent'.format(slope*100))
    cent_melt_trends.append(slope*100)

# Plot ensemble mean
ens_melt_list = [xr.open_dataset(data_dir + fname + ens_list[i] + '.nc').ice_shelf_basal_melt_flux for i in range(n_ens)]
ens_mean = xr.concat(ens_melt_list, dim='ensemble').mean(dim='ensemble')
ens_mean_rm = ens_mean.rolling(time=24, center=True).mean()
plt.plot(years, ens_mean_rm, label='Ensemble mean', color='navy', linewidth=2)
print('Ensemble mean trend:', np.mean(cent_melt_trends), 'Gt/yr/cent')

plt.legend(ncol=10)
plt.title('Naughten et al., 2022 ice-shelf basal melt')
plt.ylabel('[Gt/yr]')

print('Ensemble mean trend:', np.mean(cent_melt_trends))


#%% Plot just warm composite members----------------------------------------------------

plt.figure(figsize=(6.5,5))

for i in warm_composite_members:

    warm_comp_ens_i = f'{i:02}'
    ens_ds = xr.open_dataset(data_dir + fname + warm_comp_ens_i + '.nc')
    ens_melt = ens_ds.ice_shelf_basal_melt_flux

    # apply 2-year running mean for plotting
    ens_melt_rm = ens_melt.rolling(time=24, center=True).mean()

    plt.plot(years, ens_melt_rm, label='Ens '+warm_comp_ens_i,color='dodgerblue')
    # calc trend
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, ens_melt)
    print(warm_comp_ens_i + ': {:.2f} Gt/yr/cent'.format(slope*100))
    cent_melt_trends.append(slope*100)

# calculate mean trend and plot as trend line
# Use the same 2-year running mean as the timeseries for trend calculation and plotting
warm_comp_melt_list = [xr.open_dataset(data_dir + fname + f'{i:02}' + '.nc').ice_shelf_basal_melt_flux for i in warm_composite_members]
warm_comp_mean = xr.concat(warm_comp_melt_list, dim='ensemble').mean(dim='ensemble')
warm_comp_mean_rm = warm_comp_mean.rolling(time=24, center=True).mean()

# Remove NaNs from running mean for regression
valid = ~np.isnan(warm_comp_mean_rm)
slope, intercept, r_value, p_value, std_err = stats.linregress(years[valid], warm_comp_mean_rm[valid])
warm_comp_mean_trend = slope * 100
print('Warm composite mean trend:', warm_comp_mean_trend)

# Plot trend line using running mean
plt.plot(years, slope * years + intercept, label='Mean trend', color='navy', linewidth=2)

plt.legend(ncol=3,loc='upper left',fontsize=10)
# plt.title('Naughten et al., 2022 ice-shelf basal melt')
plt.ylabel('[Gt/yr]',fontsize=16)
plt.ylim(0,250)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Year',fontsize=16)


# %%
