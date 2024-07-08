
#%%
import cvxpy as cp
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
# needs matplotlib 3.7 (not higher), have to solve this also: https://github.com/nschloe/tikzplotlib/pull/579 (manually)
import tikzplotlib
from techtile_tools import K, M, get_d_MK, get_L_MK

import os
file_dir = os.path.join(os.path.abspath(''), "..",
                        "figures", "techtile", "non-coherent")

# THIS script is specifically for 1 N as the number of N does not affect the outcome

#%%
E = 500  # 500mJ
alpha = 0.16145203
beta = 0.0158 # mW 
NUM_SHELVES = 5
fc = 0.917  # in GHz
K_per_shelf = 24 # in example 60
P_max = 4000

 
M_vals = [2,8,84] #np.arange(2, M+1, 1)
N_vals =  [1]


obj_sol_mJ = np.zeros((len(N_vals), len(M_vals)))
x_vals_mW = []

# #%%
# def compute_constr(x,h):
#     contrs = []
#     for k in range(K):
#         constr = 0
#         for n in range(N):
#             constr += T * alpha * np.sum(np.multiply((h[n, :, k]) ** 2, x[n, :]))
#         assert constr+0.01>= E+N*T*beta, f"constraint not met {constr} from user {k}!"
#         contrs.append(constr)
#         print(f"user {k}: {constr:0.4f}mJ (< {E+N*T*beta:0.4f}mJ)")
#     return contrs

#%%
def compute_rx_energy_J(x,h):
    energy = np.zeros((K))
    for k in range(K):
        for n in range(N):
            energy[k] += T * alpha * np.sum(np.multiply((h[n, :, k]) ** 2, x[n, :]))
        assert energy[k]+0.01>= E+N*T*beta, f"constraint not met {constr} from user {k}!"
        print(f"user {k}: {energy[k]:0.4f}mJ (< {E+N*T*beta:0.4f}mJ)")
    return energy/1e3 # to convert from mJ to joule

#%%
def compute_rx_power_mW(x,h, _N):
    res_KN = compute_rx_power_per_slot_mW(x, h, _N)
    return np.sum(res_KN, axis=1)/N


def compute_rx_power_per_slot_mW(x, h, _N):
    # this one is without the scaling factor
    T = 12 * 60 * 60 / _N  # NT = 12h
    res = np.zeros((K,_N))
    for k in range(K):
        for n in range(_N):
            res[k,n] = T * (alpha * (np.sum(np.multiply(h[n, :, k] ** 2, x[n, :]))) - beta)
    return res

#%%
L_MMK = get_L_MK(M_vals)
h_MMK = []
for L_MK in L_MMK:
    h_MMK.append(np.sqrt(L_MK))


def to_hnmk(h, N):
    return np.tile(h, (N,1,1))



#%% Solve optimization problem
x_vals_mW = [[[0]*m for m in M_vals] for n in N_vals] # generate NMM matrix to hold the Tx power values in mw

for iM, M in enumerate(M_vals):
    for iN, N in enumerate(N_vals):
        T = 12 * 60 * 60 / N  # NT = 12h

        # already generated above

        h_nmk = to_hnmk(h_MMK[iM] , N) # same for for all N time slots

        assert np.alltrue(h_nmk < 1)

        # try to find a scaling function so x is close to 10
        # scale_factor = 10 *  (M/np.linalg.norm(h))
        # print(f"Scaling by {scale_factor:0.2f}")
        # h = h*scale_factor

        x = cp.Variable((N, M), name="p_nm", nonneg=True)

        # The objective function
        cost = cp.sum(x)

        # A contraint per user to have the required Energy E
        # One constraint to enforce all transmit powers >= 0 at all time
        constraints = []
        for k in range(K):
            constr = 0
            for n in range(N):
                constr += cp.sum(cp.multiply(h_nmk[n, :, k]**2, x[n, :]))
            constraints.append(E -  T*constr*alpha + N*T*beta <=0)

        # max power constraint

        for m in range(M):
            for n in range(N):
                constraints.append(x[n,m]<=P_max)

       
        prob = cp.Problem(cp.Minimize(cost), constraints)
    
        prob.solve(verbose=True) # verbose=True, save_file="output.txt"
        assert prob.status not in ["infeasible", "unbounded"], "Problem is infeasible"

        assert np.all(x.value <= P_max)

        # compute_constr(x.value,h_nmk) #TODO check if correct, to speed-up we are not checking this
        obj_sol_mJ[iN, iM] = (T*np.sum(x.value))
        print(x.value)
        print(f"Obj fun: {obj_sol_mJ[iN, iM]/1000:0.4f}J")

        x_vals_mW[iN][iM] = x.value



# %% Check if same power at all N time slots
# Assuming matrix is your N x M matrix
# Check if all values per row are approximately the same across all columns
matrix = np.array(x_vals_mW[-1][-1])
for iN, n in enumerate(N_vals):
    for iM, m in enumerate(M_vals):
        p_NM = x_vals_mW[iN][iM]
        for iim in range(p_NM.shape[-1]):
            if not np.allclose(p_NM[:,iim], p_NM[0,iim]):
                plt.imshow(np.expand_dims(p_NM[:,iim],axis=0), aspect='auto')
                plt.colorbar()
                plt.show()
                print(f"Not all transmit powers are the same over the timeslots for antenna {M_vals[iim]} for {n} timeslots")

# %% 

# plot transmit as matrix of MxN
plt.imshow(x_vals_mW[-1][-1], aspect='auto')
plt.show()

# plot receive power as matrix KxN
rx_KN = compute_rx_power_per_slot_mW(x_vals_mW[-1][-1], to_hnmk(h_MMK[-1] , N), N_vals[-1])
plt.imshow(rx_KN.T, aspect='auto')
plt.colorbar()

# %% Plot power of each antenna
import matplotlib.animation as animation
import matplotlib as mpl
# Initialize the figure and axis
fig, ax = plt.subplots()
sc = ax.scatter([], [])

plt.xlim(0, 8)
plt.ylim(0, 4)
plt.xlabel('X')
plt.ylabel('Y')


# vmin=values[np.isfinite(values)].min()


# m = ScalarMappable(norm=Normalize(vmin=-20, vmax=36), cmap='viridis')
# cbar = fig.colorbar(m, ax=ax)
# cbar.set_label('Tx [dBm]')


# # Update function for the animation
# def update(i):
#     ant_coords = np.asarray(antenna_pos_per_loop[i])
#     tx_powers = 10*np.log10(x_vals_mW[0][i]).flatten()
#     X = np.c_[ant_coords[:,0], ant_coords[:,1]]
#     sc.set_facecolor(m.to_rgba(tx_powers))
#     sc.set_clim(vmin=min(tx_powers), vmax=max(tx_powers))
#     sc.set_offsets(X)
#     return sc,

# # Set up the animation
# anim = animation.FuncAnimation(fig, update, frames=len(antenna_pos_per_loop), blit=True, interval=1000)

# plt.rcParams['animation.ffmpeg_path'] = r'C:\Program Files\ffmeg\bin\ffmpeg.exe'
# from IPython.display import HTML
# HTML(anim.to_html5_video())

# %% Consumed energy for each M

plt.xlabel('M')
plt.ylabel('Energy kJ')
for iN, N in enumerate(N_vals):
    plt.plot(M_vals, obj_sol_mJ[iN, :]/1e6)
plt.legend()
plt.show()

# %% Consumed power for each M in W

plt.xlabel('M')
plt.ylabel('Total Power W')
for iN, N in enumerate(N_vals):
    plt.plot(M_vals, (obj_sol_mJ[iN, :]/1e3)/(12*60*60))
plt.legend()
plt.show()

# %% Consumed power for each M in dBm

plt.xlabel('M')
plt.ylabel('Total Power dBm')
for iN, N in enumerate(N_vals):
    plt.plot(M_vals, 10*np.log10((obj_sol_mJ[iN, :]/(12*60*60))))
plt.legend()
plt.show()

# %%
# # Plot the CDF
fig = go.Figure()
is_pass = [1,2,3,4,5,10,30,50,84]
for iM, M in enumerate(M_vals):
    for iN, _N in enumerate(N_vals):
        if M in is_pass:
            a = compute_rx_power_mW(x_vals_mW[iN][iM], to_hnmk(h_MMK[iM] , N), _N)/1e3
            x = np.sort(a)
            y = np.linspace(0, 1, len(a), endpoint=False)
            fig.add_trace(go.Scatter(
                x=x, y=y, mode='lines', name=f'CDF M{M} N{_N}'))

fig.update_layout(title='Cumulative Distribution Function (CDF)',
                  xaxis_title='Values',
                  yaxis_title='% above x',
                  template='plotly_dark')
fig.show()

# %%
# Plot the CDF in matplotlib to tikz

fig, ax = plt.subplots()

i = 0
is_pass = [2,8,84]
for iM, M in enumerate(M_vals):
    for iN, _N in enumerate(N_vals):
        if M in is_pass:
            a = compute_rx_energy_J(x_vals_mW[iN][iM], to_hnmk(h_MMK[iM], _N))
            x = np.sort(a)
            y = np.linspace(0, 1, len(a), endpoint=False)
            ax.plot(x, y, label=f'{M} ({10*np.log10((np.sum(x_vals_mW[iN][iM]))/_N):.2f}dBm)', linewidth=2)
        i +=1

ax.set_xlabel('Received DC energy [\si{\joule}]')
ax.set_ylabel('\Gls{cdf}')
ax.legend()
ax.grid(True)

# Save plot as Tikz
tikzplotlib.save(os.path.join(file_dir, "cdf-dc-rx.tex"))

plt.show()


# %%
i = 0
y = [0]*len(M_vals)
for iM, M in enumerate(M_vals):
    for iN, _N in enumerate(N_vals):
        num_tx = (np.array(10*np.log10(x_vals_mW[iN][iM])) > -20).sum()
        y[iM] = num_tx
        i += 1
plt.plot(M_vals, y/M_vals)
plt.show()

# %%
i = 0
y = [0]*len(M_vals)
for iM, M in enumerate(M_vals):
    for iN, _N in enumerate(N_vals):
        num_tx = (np.array(10*np.log10(x_vals_mW[iN][iM])) > -20).sum()
        y[iM] = num_tx
        i += 1
plt.bar(M_vals, y)
plt.show()

# %%
fig = go.Figure()
y = obj_sol_mJ[0,:]
x = M_vals
fig.add_trace(go.Scatter(x=x, y=y/1e6, mode='lines', name=f'CDF M{M}'))

fig.update_layout(title='Energy consumption',
                  xaxis_title='M',
                  yaxis_title='Energy [kJ]',
                  template='plotly_dark')
fig.show()

# %%
fig, ax = plt.subplots()

y = obj_sol_mJ[0, :]/1e3 # in W
x = M_vals

ax.plot(x, y/(12*60*60))

# output as table rather than tikz picture
a = np.zeros((len(x), 2))
a[:,0] = x
a[:,1] = y/(12*60*60)
np.savetxt("../figures/techtile/partial-power-tx.dat", a, delimiter=" ", header='x y', comments="", fmt="%.10f")

ax.set_title('Power consumption')
ax.set_xlabel('M')
ax.set_ylabel('Power [W]')
ax.grid(True)

# Save plot as Tikz
# manually adapted it in overleaf, so do not export it anymore.
# tikzplotlib.save("../figures/techtile/partial-power-tx.tex") 

plt.show()
# %% Save the transmit powers for the experiments
import csv

MAX_GAIN = 80 #dB
# scale everything to max power 

allpowers = []

for iM, M in enumerate(M_vals):
    allpowers.extend(x_vals_mW[0][iM][0]) # iN,iM = NxM matrix

MAX_POWER = np.max(10*np.log10(allpowers))

for im, m in enumerate(M_vals):

    print(m)
    print(obj_sol_mJ[0, im]/T)
    print(np.sum(x_vals_mW[0][im][0]))


    filename = f"../measurements/partial_csv_tx_powers_M{m}.csv"
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(("tile","ch","gain_dB")) # header

        ids = selected_antenna_ids[im]
        powers = 10*np.log10(np.asarray(x_vals_mW[0][im][0]))
        powers = powers + (MAX_GAIN-MAX_POWER)
        powers[powers<0] = 0

        rel_powers = []

        for iid, m_id in enumerate(ids):
            x,y,z = techtile_antennas[m_id]
            csvwriter.writerow((tiles_techtile_antennas[m_id],channel_techtile_antennas[m_id], powers[iid]))
            rel_powers.append(powers[iid])

        # check the total power
        rel_powers = np.asarray(rel_powers)
        print(np.sum(10**(rel_powers/10)))
# %%
