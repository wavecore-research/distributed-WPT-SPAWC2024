
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
 
NUM_SHELVES = 5
fc = 0.917  # in GHz
K_per_shelf = 24 # in example 60

techtile_antennas = []

import requests
import yaml

# Retrieve the file content from the URL
response = requests.get("https://raw.githubusercontent.com/techtile-by-dramco/plotter/main/positions.yml", allow_redirects=True)
# Convert bytes to string
content = response.content.decode("utf-8")
# Load the yaml
techtile_positions = yaml.safe_load(content)

for tile in techtile_positions["antennes"]:
    for ch in tile["channels"]:
        if ch["z"] == 2.4:
            # antenna is in ceiling
            techtile_antennas.append((ch["x"],ch["y"],ch["z"]))
print(f"Found {len(techtile_antennas)} antennas")




def euclidean_distance(v1, v2) -> np.ndarray:
    """
    calculate the euclidean distance between two vectors v1 and v2
    """
    return np.sqrt(np.sum((np.array(v1) - np.array(v2))**2))


def distance_matrix(v1, v2) -> np.ndarray:
    """
    create a distance matrix between every element of vector v1 and v2
    """
    matrix = np.zeros((len(v1), len(v2)))
    for i, v1i in  enumerate(v1):
        for j, v2j in enumerate(v2):
            matrix[i, j] = euclidean_distance(v1i, v2j)
    return matrix

def find_closest_coordinate(positions, target): 
    distances = np.linalg.norm(positions - target, axis=1)
    closest_index = np.argmin(distances)
    closest_coordinate = positions[closest_index]
    return closest_index, closest_coordinate


# %%
def find_mass_center(coordinates, x):
    from sklearn.cluster import KMeans

    coordinates = np.array(coordinates)

    # Extracting only the X and Y coordinates
    xy_coordinates = np.array([[coord[0], coord[1]] for coord in coordinates])

    # Applying K-means clustering
    kmeans = KMeans(n_clusters=x)
    kmeans.fit(xy_coordinates)

    # Getting the centers of the clusters
    centers = kmeans.cluster_centers_

    assert len(centers) == x

    best_locations = []
    bext_loc_indx = []
    for center in centers:
        # return the closest antenna to that center
        center_xyz = np.array([center[0], center[1], 2.4])
        idx, loc = find_closest_coordinate(coordinates, center_xyz)
        best_locations.append(loc)
        bext_loc_indx.append(idx)

    # just plot one to show how it is done 
    if x == 6:
        cmap = mpl.colormaps['viridis']

        center_colors = [cmap(float(l)/x) for l in range(x)]

        selected_coordinate = np.array(best_locations)
        xy_kmeans = kmeans.predict(xy_coordinates)
        labels = kmeans.labels_

        fig, ax = plt.subplots()

        

        for _i in range(x):

            ax.plot(selected_coordinate[_i, 0],
                    selected_coordinate[_i, 1], linestyle='None', marker='o', markersize=np.sqrt(30),
                    markeredgecolor=center_colors[_i], markerfacecolor=center_colors[_i], linewidth=1.0
                    )


            ax.plot(xy_coordinates[labels == _i, 0],
                    xy_coordinates[labels == _i, 1], linestyle='None', marker='o', markersize=np.sqrt(30), c=center_colors[_i],
                    markeredgecolor=center_colors[_i], markerfacecolor=center_colors[_i], linewidth=1.0, alpha=.2
                    )
            
            ax.plot(centers[_i, 0], centers[_i, 1], linestyle='None', marker='p', markersize=np.sqrt(30), c=center_colors[_i],
                    markeredgecolor='black', markerfacecolor=center_colors[_i], linewidth=1.0, alpha=.5
                    )


            # plt.scatter(xy_coordinates[labels == _i, 0],
            #             xy_coordinates[labels == _i, 1], c=center_colors[_i])
            # plt.scatter(centers[_i, 0], centers[_i, 1],
            #             c=center_colors[_i], marker='x', s=200, linewidths=2)

        # plt.scatter(xy_coordinates[:,0], xy_coordinates[:,1],
        #             c=xy_kmeans, s=50, cmap='viridis')
        # plt.scatter(centers[:, 0], centers[:, 1],
        #             c=center_colors, s=200, alpha=0.2)
        
        
        ax.set_xlabel('Length (m)')
        ax.set_ylabel('Width (m)')
        ax.grid(True)
        plt.show()
    


    return bext_loc_indx, best_locations

#%%

ESL_positions = []

for x in np.linspace(0, 8.4, K_per_shelf):
    for z in np.linspace(0.1, 2.1, NUM_SHELVES):
        # wall left
        ESL_positions.append((x, 3.5, z)) # 50cm away from the wall
        # wall right
        ESL_positions.append((x, 0.5, z)) # 50cm away from the wall

K = len(ESL_positions)  # number of ESLs
M = len(techtile_antennas)


def get_K():
    return K

def get_M():
    return M

def get_d_MK():
    return distance_matrix(techtile_antennas, ESL_positions)


def get_L_MK(M_vals):
    L_MK = []
    for iM, M in enumerate(M_vals):
        # pick the M antennas based on k-means clustering

        antenna_idx, antenna_positions = find_mass_center(techtile_antennas, M) 

        d_MK = distance_matrix(antenna_positions, ESL_positions)

        h_dB = 31.84 + 21.50 * np.log10(d_MK) + 19.00 * np.log10(fc) # InF LoS model from TR 138 901

        assert np.all(h_dB > 0)

        l = 10.0 ** (-h_dB / 10.0)
        L_MK.append(l)
    return L_MK





# # %% Plot the selection of each antenna per iterations going from M=1->84 antennas
# import matplotlib.animation as animation
# # Initialize the figure and axis
# fig, ax = plt.subplots()
# sc = ax.scatter([], [])

# plt.xlim(0, 8)
# plt.ylim(0, 4)
# plt.xlabel('X')
# plt.ylabel('Y')


# # Update function for the animation
# def update(i):
#     ant_coords = np.asarray(antenna_pos_per_loop[i])
#     X = np.c_[ant_coords[:,0], ant_coords[:,1]]
#     sc.set_offsets(X)
#     return sc,

# # Set up the animation
# anim = animation.FuncAnimation(fig, update, frames=len(antenna_pos_per_loop), blit=True)

# plt.rcParams['animation.ffmpeg_path'] = r'C:\Program Files\ffmeg\bin\ffmpeg.exe'
# from IPython.display import HTML
# HTML(anim.to_html5_video())
        


