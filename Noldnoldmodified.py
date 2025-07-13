import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import diags
import scipy.sparse as sp
import math
import os.path
import pickle
import time

#Global variables:
p = None
q = None
x_unitcells = None
y_unitcells = None
flux = None
dkx = None
dky = None
nux_list = None
nuy_list = None
kx_list = None
ky_list = None
matrix_size = None
filename = None

def OneParticle(p_val, q_val, x_unitcells_val, y_unitcells_val):
        #To use the variables globally we need to introduce them all, after deleting every self. operation
        global p, q, x_unitcells, y_unitcells, flux, dkx, dky, nux_list, nuy_list, kx_list, ky_list, matrix_size, filename
        p = p_val
        q = q_val
        x_unitcells = x_unitcells_val
        y_unitcells = y_unitcells_val

        flux = p / q

        dkx = 2 * np.pi / x_unitcells
        dky = 2 * np.pi / y_unitcells

        nux_list = np.arange(-x_unitcells / 2, x_unitcells / 2, dtype=int)
        nuy_list = np.arange(-y_unitcells / 2, y_unitcells / 2, dtype=int)

        kx_list = dkx * nux_list
        ky_list = dky * nuy_list

        matrix_size = q
        filename = f"/p={p}_q={q}_lx={x_unitcells}_ly={y_unitcells}.pkl"

def hamiltonian( kx, ky):
        if matrix_size == 1:
            matrix = 2 * (np.cos(kx) + np.cos(ky))
        elif matrix_size == 2:
            matrix = [[-2 * np.cos(kx), 1 + np.exp(-1j * ky)],
                      [1 + np.exp(1j * ky), 2 * np.cos(kx)]]
        else:
            main_diagonal = np.array([2 * np.cos(kx + 2 * np.pi * flux * r) for r in range(1, q+1)])
            upper_off_diagonal = np.full(matrix_size, 1)
            corner_diagonal = np.exp(-1j * ky)

            diagonal_list = [main_diagonal, upper_off_diagonal, np.conjugate(upper_off_diagonal), corner_diagonal, np.conjugate(corner_diagonal)]
            matrix = diags(diagonal_list, [0, 1, -1, q - 1, -q + 1])
        return -matrix.toarray()

def energy_line( nux, dictionary):
        kx = dkx * nux
        for nuy in nuy_list:
            ky = dky * nuy
            values, vectors = np.linalg.eigh(hamiltonian(kx, ky))
            sorting = np.argsort(values)
            values, vectors = values[sorting].real, vectors[sorting]
            dictionary[int(nux), int(nuy)] = values, vectors.T

def energy_spectrum( calculate_new=False):
        filepath = "./dictionaries/one_particle/" + filename

        if os.path.exists(filepath) and not calculate_new:
            existing_file = open(filepath, "rb")
            energy_dictionary = pickle.load(existing_file)
        else:
            print("A new dictionary is to be calculated.")
            energy_dictionary = {}
            for nux in nux_list:
                energy_line(nux, energy_dictionary)
            new_file = open(filepath, "wb")
            pickle.dump(energy_dictionary, new_file)
            new_file.close()
            print("Calculation done!")
        return energy_dictionary

def plot3d():
        full_energy_list = []
        for r in range(matrix_size):
            full_energy_list.append(np.zeros((len(kx_list), len(ky_list))))
        full_energy_list = np.transpose(np.array(full_energy_list), axes=(1, 2, 0))

        energies = energy_spectrum(calculate_new=True)

        for i in range(len(nux_list)):
            for j in range(len(nuy_list)):
                full_energy_list[i, j] = energies[nux_list[i], nuy_list[j]][0]
        full_energy_list = np.transpose(np.array(full_energy_list), axes=(2, 0, 1))

        kxs, kys = np.meshgrid(kx_list / np.pi, ky_list / np.pi)

        ax = plt.axes(projection="3d")
        palette = sns.color_palette("hls", matrix_size)
        for r in range(matrix_size):
            ax.plot_surface(kxs, kys, full_energy_list[r].T, color=palette[r])

        font_size = 12
        ax.set_xlabel(r"$k_x$/$\pi$", fontsize=font_size)
        ax.set_ylabel(r"$k_y$/$\pi$", fontsize=font_size)
        ax.set_zlabel(r"E/t", fontsize=font_size)
        plt.title(f"p/q = {p}/{q}")
        plt.tight_layout()
        plt.show()

def plot_state(nux, nuy, band_index=0):
        energies = energy_spectrum()
        plotted_state = np.abs(energies[nux, nuy][1][band_index]) ** 2

        state_labels = np.arange(1, matrix_size + 1)

        plt.bar(state_labels, plotted_state, width=0.8)
        plt.title(
            rf"probability density for kx/$\pi$ = {round(dkx*nux/np.pi, 2)} and ky/$\pi$= "
            rf"{round(dky*nuy/np.pi, 2)} and q={q}")
        plt.ylim(0, 1)
        plt.xlabel(r"site $\alpha$")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    start = time.time()
    system = OneParticle(1, 3, 180, 230)
    print(q)
    plot3d()
    plot_state(0, 0,1)
    end = time.time()
    print(f"the code took {end - start:.4f} seconds to complete.")
