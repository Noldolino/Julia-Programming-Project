import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import diags
import scipy.sparse as sp
import math
import os.path
import pickle


class OneParticle:
    # pre-definitions
    def __init__(self, p, q, x_unitcells, y_unitcells):
        self.p = p
        self.q = q
        self.x_unitcells = x_unitcells
        self.y_unitcells = y_unitcells

        self.flux = p / q

        self.dkx = 2 * np.pi / self.x_unitcells
        self.dky = 2 * np.pi / self.y_unitcells

        self.nux_list = np.arange(-x_unitcells / 2, x_unitcells / 2, dtype=int)
        self.nuy_list = np.arange(-y_unitcells / 2, y_unitcells / 2, dtype=int)

        self.kx_list = self.dkx * self.nux_list
        # von -pi bis pi
        self.ky_list = self.dky * self.nuy_list

        self.matrix_size = self.q
        self.filename = f"/p={self.p}_q={self.q}_lx={self.x_unitcells}_ly={self.y_unitcells}.pkl" #

    def hamiltonian(self, kx, ky):
        if self.matrix_size == 1:
            matrix = 2 * (np.cos(kx) + np.cos(ky))
        elif self.matrix_size == 2:
            matrix = [[-2 * np.cos(kx), 1 + np.exp(-1j * ky)],
                      [1 + np.exp(1j * ky), 2 * np.cos(kx)]]
        else:
            main_diagonal = np.array([2 * np.cos(kx + 2 * np.pi * self.flux * r) for r in range(1, self.q+1)])
            upper_off_diagonal = np.full(self.matrix_size, 1)
            corner_diagonal = np.exp(-1j * ky)

            diagonal_list = [main_diagonal, upper_off_diagonal, np.conjugate(upper_off_diagonal), corner_diagonal,
                             np.conjugate(corner_diagonal)]
            matrix = diags(diagonal_list, [0, 1, -1, self.q - 1, -self.q + 1])
        return -matrix.toarray()

    def energy_line(self, nux, dictionary):
        kx = self.dkx * nux
        for nuy in self.nuy_list:
            ky = self.dky * nuy
            values, vectors = np.linalg.eigh(self.hamiltonian(kx, ky))
            sorting = np.argsort(values)
            values, vectors = values[sorting].real, vectors[sorting]
            dictionary[int(nux), int(nuy)] = values, vectors.T

    def energy_spectrum(self, calculate_new=False):

        filepath = "./dictionaries/one_particle/" + self.filename

        if os.path.exists(filepath) and not calculate_new:
            existing_file = open(filepath, "rb")
            energy_dictionary = pickle.load(existing_file)
        else:
            print("A new dictionary is to be calculated.")
            energy_dictionary = {}
            # can be multi-processed
            for nux in self.nux_list:
                self.energy_line(nux, energy_dictionary)
            new_file = open(filepath, "wb")
            pickle.dump(energy_dictionary, new_file)
            new_file.close()
            print("Calculation done!")
        return energy_dictionary

    def plot3d(self):

        full_energy_list = []
        for r in range(self.matrix_size):
            full_energy_list.append(np.zeros((len(self.kx_list), len(self.ky_list))))
        full_energy_list = np.transpose(np.array(full_energy_list), axes=(1, 2, 0))

        energies = self.energy_spectrum(calculate_new=True)

        for i in range(len(self.nux_list)):
            for j in range(len(self.nuy_list)):
                full_energy_list[i, j] = energies[self.nux_list[i], self.nuy_list[j]][0]
        full_energy_list = np.transpose(np.array(full_energy_list), axes=(2, 0, 1))

        kxs, kys = np.meshgrid(self.kx_list / np.pi, self.ky_list / np.pi)

        ax = plt.axes(projection="3d")
        palette = sns.color_palette("hls", self.matrix_size)
        for r in range(self.matrix_size):
            ax.plot_surface(kxs, kys, full_energy_list[r].T, color=palette[r])

        font_size = 12
        ax.set_xlabel(r"$k_x$/$\pi$", fontsize=font_size)
        ax.set_ylabel(r"$k_y$/$\pi$", fontsize=font_size)
        ax.set_zlabel(r"E/t", fontsize=font_size)
        plt.title(f"p/q = {self.p}/{self.q}")
        plt.tight_layout()

    def plot_state(self, nux, nuy, band_index=0):

        energies = self.energy_spectrum()
        plotted_state = np.abs(energies[nux, nuy][1][band_index]) ** 2

        state_labels = np.arange(1, self.matrix_size + 1)

        plt.bar(state_labels, plotted_state, width=0.8)
        plt.title(
            rf"probability density for kx/$\pi$ = {round(self.dkx*nux/np.pi, 2)} and ky/$\pi$= "
            rf"{round(self.dky*nuy/np.pi, 2)} and q={self.q}")
        plt.ylim(0, 1)
        plt.xlabel(r"site $\alpha$")
        plt.tight_layout()


if __name__ == "__main__":

    system = OneParticle(1, 4, 100, 100)

    print(system.q)




