
# %% Load packages
import os
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.path as mpath

base_dir = "/glade/campaign/univ/uwas0134/Gemma/Data/"

#%% Functions for loading data-----------------------------------------------------------------------

simulation = 'CESM2_SSP370'
vname = 'V' #V, U, TS, PSL

"""
Load temperature, psl, u-wind, and v-wind datasets for a given simulation.
"""
scenario_dir = os.path.join(simulation, vname)

sim_dir = base_dir + 'Model/' + simulation + '/' + vname + "/"
print(sim_dir)

f_vname_dir = {'V': 'v1000',
               'U': 'u1000',
               'TS': 'TS',
               'PSL': 'PSL'}
f_vname = f_vname_dir[vname]

# Get all .nc files in the directory
nc_files = [f for f in os.listdir(sim_dir) if f.endswith('.nc') and f_vname in f]
print(f"Found {len(nc_files)} ensemble files for simulation {simulation} and variable {vname}.")

# calc annual means for each ensemble member
# create array with shape (num_ensembles, num_years)
annual_means = np.zeros((len(nc_files), 87, 192, 288))  # assuming 87 years from 2015 to 2100 incl., 192 lats, 288 lons
cent_trends = np.zeros((len(nc_files), 192, 288))
for i, nc_file in enumerate(nc_files):

    print(i)

    # open dataset
    ds = xr.open_dataset(os.path.join(sim_dir, nc_file))
    v_data = ds[f_vname]

    # calculate annual mean at each latlon grid point
    annual_mean = v_data.resample(time='1Y').mean(dim='time').values  # shape (87, 192, 288)
    annual_means[i,:,:] = annual_mean

    # calculate trend per century at each latlon grid point
    years = np.arange(2015, 2101)
    for lat in range(192):
        for lon in range(288):
            y = annual_mean[:, lat, lon]
            # perform linear regression to get trend
            A = np.vstack([years, np.ones(len(years))]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            cent_trends[i, lat, lon] = m * 100  # trend per century

    ds.close()


# %%
