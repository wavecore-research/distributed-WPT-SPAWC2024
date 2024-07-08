# %%

from techtile_tools import ESL_positions
import numpy as np

# read all info Files 
import glob
import yaml
import re
import pandas as pd
import os
import tikzplotlib

# Define the directory path
directory_path = "../measurements/meas_data/"

# Define the pattern to match files with the format "*_info.yml"
pattern = directory_path + "*_info.yml"


# %%
# Use glob to find all files matching the pattern
file_list = glob.glob(pattern)

data_per_M = {2: [], 8:[], 84:[]}
TX_powers = {2: [], 8:[], 84: []}
# Loop through the list of files
for file_path in file_list:
    # Read each file
    with open(file_path, 'r') as file:
        # Process the file content here

        file_id = os.path.basename(file_path).split("_")[0]
        print(file_id)
        info = yaml.safe_load(file)
        x_offset = info["info"]["xaxisoffset"]
        y_offset = info["info"]["yaxisoffset"]
        z_offset = info["info"]["zaxisoffset"]

        conf_name = info["usrp"]["external_configuration_file"]

        num_ant = int(re.search(r"_M(\d+)\.csv", conf_name).group(1))

        # Read the CSV file into a pandas DataFrame
        data_path = os.path.join(
            directory_path, f"{file_id}_esl_paper_meas.csv")
        print(data_path)
        data = pd.read_csv(data_path)
        # Extract the specified columns from the DataFrame
        data = data[['x', 'y', 'z', 'dbm']]

        data['x'] += x_offset
        data['y'] += y_offset
        data['z'] += z_offset

        data_per_M[num_ant].append(data)
        
# hstack all data now a list of list
# %%
for M in data_per_M.keys():
    data_per_M[M] = np.vstack(data_per_M[M])

    # read tx powers from partial_csv_tx_powers_M2.csv
    file_path = os.path.join(
        "../measurements", f"partial_csv_tx_powers_M{M}.csv")

    TX_powers[M].extend(pd.read_csv(file_path)["gain_dB"])
    

# %%
# find the most close location per ESL device
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
alpha = 0.16145203
NT = 12*60*60

for M in data_per_M.keys():
    sampled_locations = data_per_M[M][:, 0:3]
    powers = data_per_M[M][:, -1]

    a = []

    for esl in ESL_positions:
        dist = np.linalg.norm(esl - sampled_locations, axis=1)
        closest_diff = np.min(dist)
        closest_idx = np.argmin(dist)

        print(closest_diff)
        print(closest_idx)
        a.append(powers[closest_idx])

    a = np.asarray(a)
    a_lin = (10**(a/10))
    x = np.sort(alpha*a_lin*1e3)
    y = np.linspace(0, 1, len(a), endpoint=False)
    pwr_dBm = np.array(TX_powers[M]) - 67.4
    pwr = np.sum((10**(pwr_dBm/10)))

    pwr_rx = np.sum(alpha*a_lin)

    ax.plot(
        x, y, label=f'{M} ({pwr:0.2f}mW)', linewidth=2)

ax.set_xlabel('Received DC power [\si{\micro\watt}]')
ax.set_ylabel('\Gls{cdf}')
ax.legend()
ax.grid(True)

# Save plot as Tikz
tikzplotlib.save("../figures/techtile/measurements.tex")

plt.show()

# %%
