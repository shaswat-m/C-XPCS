# --------------------------------------------------------
# intensity_util.py
# by Shaswat Mohanty, shaswatm@stanford.edu
#
# Objectives
# Library of functions for computing the g(r), s(q) and I(q)
# Contains first principle calculation functions which are not eventually used in the FFT implementation
#
# Cite: (https://doi.org/10.1088/1361-651X/ac860c)
# --------------------------------------------------------
from numba import jit, prange
from numba import set_num_threads
import itertools
from functools import partial
import multiprocessing as mp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from md_util import *


################### structure and intensity computation ######################
def g_r_verlet(pos, bins, rc, h, a=None, nnlist=None, bin_range=None):
    '''RDF computing function

            Parameters
            ----------
            pos : float, dimension (natoms, 3)
                Position of all atoms
            rc : float,
                Cutoff radius for g(r)
            h : float, dimension (3, 3)
                Periodic box size h = (c1|c2|c3)
            bins : int
                Number of bins
            bin_range : float, dimension(bin,)
                specific bin range instead of bins
            nnlist : List,
                List of neighbors -- similar nindex

            Returns
            -------
            bin_centers : float, dimension (bins,)
                Centers of all bins
            hist_normalized : float, dimension (bins,)
                Normalized histograms to be equivalent to the g(r)
    '''
    if rc>h.max()/2:
        newpos,newh = config_repeater(pos,h)
        pos = newpos.copy()
        h = newh.copy()
    if nnlist != None:
        index = nnlist
        nn = np.array([len(l) for l in index], dtype=int)
    else:
        nn, index = verletlist(pos, h, rc)

    nparticles = pos.shape[0]
    hinv = np.linalg.inv(h)
    dist = []
    for i in range(nparticles):
        ri = np.array([pos[i, :]])
        ind = index[i]
        rj = pos[ind, :]
        drij = pbc(rj - np.repeat(ri, len(ind), axis=0), h, hinv)
        dist.extend(np.linalg.norm(drij, axis=1).tolist())
    if bin_range is None:
        hist, bin_edges = np.histogram(dist, bins)
    else:
        hist, bin_edges = np.histogram(dist, bins, range=(bin_range[0], bin_range[1]))
    # print("bin_edges = [%g : %g : %g]"%(bin_edges[0], bin_edges[1]-bin_edges[0], bin_edges[-1]))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    cnt = (4.0 / 3.0) * np.pi * (np.power(bin_edges[1:], 3) - np.power(bin_edges[:-1], 3)) * nparticles / np.linalg.det(
        h)
    hist_normalized = np.divide(hist, cnt) / nparticles

    return bin_centers, hist_normalized

def g_r_verlet_binary(pos_a, pos_b, bins, rc, a, h):
    '''Partial RDF computing function between A-B types

                        Parameters
                        ----------
                        pos_a : float, dimension (natoms_a, 3)
                            Position of all atoms of type A
                        pos_b : float, dimension (natoms_b, 3)
                            Position of all atoms of type B
                        rc : float,
                            Cutoff radius for g(r)
                        h : float, dimension (3, 3)
                            Periodic box size h = (c1|c2|c3)
                        bins : int
                            Number of bins

                        Returns
                        -------
                        bin_centers : float, dimension (bins,)
                            Centers of all bins
                        hist_normalized : float, dimension (bins,)
                            Normalized histograms to be equivalent to the g(r)
    '''
    nn_a, index_a, nn_b, index_b = verletlist_binary(pos_a, pos_b, h, rc)

    atoms_a = pos_a.shape[0]
    atoms_b = pos_b.shape[0]
    nparticles = atoms_a + atoms_b
    hinv = np.linalg.inv(h)
    dist = []
    for i in range(atoms_a):
        ri = np.array([pos_a[i, :]])
        ind = index_a[i]
        rj = pos_b[ind, :]
        drij = pbc(rj - np.repeat(ri, len(ind), axis=0), h, hinv)
        dist.extend(np.linalg.norm(drij, axis=1).tolist())

    for i in range(atoms_b):
        ri = np.array([pos_b[i, :]])
        ind = index_b[i]
        rj = pos_a[ind, :]
        drij = pbc(rj - np.repeat(ri, len(ind), axis=0), h, hinv)
        dist.extend(np.linalg.norm(drij, axis=1).tolist())

    hist, bin_edges = np.histogram(dist, bins)
    print("bin_edges = [%g : %g : %g]" % (bin_edges[0], bin_edges[1] - bin_edges[0], bin_edges[-1]))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    cnt = (4.0 / 3.0) * np.pi * (np.power(bin_edges[1:], 3) - np.power(bin_edges[:-1], 3)) * nparticles / np.linalg.det(
        h)
    hist_normalized = 2.0 * np.divide(hist, cnt) / nparticles

    return bin_centers, hist_normalized

def s_q_from_g_r(q_array, r_array, g_array, r0, rho):
    '''Numerically integrate the g(r) (Fourier Transform) to obtain s(q)

            Parameters
            ----------
            q_array : float, dimension (N-q, 3)
                Array of all wave-vectors for s(q) computation
            r_array : float, dimension (N_pairs)
                Pairwise distances/vectors for all valid pairs within a cutoff
            r0 : float,
                Lowest value in r-space, where analytical integral is to be computed
            g_array : float, dimension (bins, )
                g(r) along r_array
            rho : float,
                Atomic density

            Returns
            -------
            s_array : float, dimension (N-q,)
                s(q) computed for given q_array
    '''
    s_array = np.zeros(q_array.shape)
    for i in range(q_array.shape[0]):
        q = q_array[i]
        r_sinqr = np.multiply(r_array, np.sin(q * r_array))
        int_0_r0 = (r0 * q * np.cos(r0 * q) - np.sin(r0 * q)) / np.power(q, 3)  # integral from 0 to r0
        s_array[i] = 1.0 + 4.0 * np.pi * rho * (int_0_r0 + np.trapz(np.multiply(g_array - 1.0, r_sinqr), x=r_array) / q)
    return s_array

def s_q_from_pos(q_array, pos, h, rc, rho, nnlist=None):
    '''Obtain s(q) from pointwise computation of Fourier transform of atomic positions

            Parameters
            ----------
            q_array : float, dimension (N-q, 3)
                Array of all wave-vectors for s(q) computation
            pos : float, dimension (natoms, 3)
                Position of all atoms
            rc : float,
                Cutoff radius for g(r)
            h : float, dimension (3, 3)
                Periodic box size h = (c1|c2|c3)
            rho : float,
                Atomic density
            nnlist : List,
                List of neighbors -- similar nindex

            Returns
            -------
            s_array : float, dimension (N-q,)
                s(q) computed for given q_array
    '''
    if nnlist != None:
        index = nnlist
        nn = np.array([len(l) for l in index], dtype=int)
    else:
        nn, index = verletlist(pos, h, rc)

    # construct list of interatomic distances
    nparticles = pos.shape[0]
    hinv = np.linalg.inv(h)
    dist = []
    for i in range(nparticles):
        ri = np.array([pos[i, :]])
        ind = index[i]
        rj = pos[ind, :]
        drij = pbc(rj - np.repeat(ri, len(ind), axis=0), h, hinv)
        dist.extend(np.linalg.norm(drij, axis=1).tolist())

    r_array = np.array(dist)
    #    r_array= f_weight_vec(r_array,rc)
    s_array = np.zeros(q_array.shape)
    for i in range(q_array.shape[0]):
        q = q_array[i]
        sinqr_div_r = np.divide(np.sin(q * r_array), r_array)  # np.exp(-1j * r_array* q)
        int_0_rc = (rc * q * np.cos(rc * q) - np.sin(rc * q)) / np.power(q, 3)  # integral from 0 to rc
        s_array[i] = 1.0 + 4.0 * np.pi * rho * int_0_rc + np.sum(sinqr_div_r) / (
                    nparticles * q)  # np.sum(sinqr_div_r)/(nparticles)
    return s_array.real

def get_q3(k0, ky_relative, kz_relative):
    '''Get a stacked 2D of kx, ky, and kz values on a detector grid

            Parameters
            ----------
            k_0 : float,
                Wave-vector magnitude
            ky_relative : float,
                Non-dimensional wave-vector in y direction
            kz_relative : float,
                Non-dimensional wave-vector in z direction

            Returns
            -------
            q3_array : float, dimension (N-q, N-q, 3)
                3d q_array of wavevector in all direction on a 2D dectector grid
    '''
    ky, kz = np.meshgrid(ky_relative, kz_relative)
    kx = np.sqrt(1.0 - np.square(ky) - np.square(kz))
    qx, qy, qz = kx - 1.0, ky, kz
    q3_array = np.stack((qx, qy, qz), axis=-1) * k0
    return q3_array

def get_r_array(pos, h, rc, nnlist=None, atoms_add=None):
    '''Get array of pairwise distances

            Parameters
            ----------
            pos : float, dimension (natoms, 3)
                Position of all atoms
            rc : float,
                Cutoff radius for g(r)
            h : float, dimension (3, 3)
                Periodic box size h = (c1|c2|c3)
            nnlist : List,
                List of neighbors -- similar nindex
            atoms_add : int,
                Number of atoms to consider for getting r_array

            Returns
            -------
            r_array : float, dimension (arbitrary,)
                array of all pairwise distances for the specified cutoff
    '''

    if nnlist != None:
        index = nnlist
        nn = np.array([len(l) for l in index], dtype=int)
    else:
        nn, index = verletlist(pos, h, rc)

    # construct list of interatomic distances
    if atoms_add != None:
        nparticles = atoms_add
    else:
        nparticles = pos.shape[0]
    hinv = np.linalg.inv(h)
    r_list = []
    for i in range(nparticles):
        ri = np.array([pos[i, :]])
        ind = index[i]
        rj = pos[ind, :]
        drij = pbc(rj - np.repeat(ri, len(ind), axis=0), h, hinv)
        dis = np.linalg.norm(drij, axis=1)
        indo = np.where(dis > rc)
        drij = np.delete(drij, indo, axis=0)
        r_list.extend(drij.tolist())

    r_array = np.array(r_list)
    return r_array

def get_r_array_ref(pos, pos_ref, h, rc, nnlist=None):
    '''Get array of pairwise distances

            Parameters
            ----------
            pos : float, dimension (natoms, 3)
                Position of all atoms
            rc : float,
                Cutoff radius for g(r)
            h : float, dimension (3, 3)
                Periodic box size h = (c1|c2|c3)
            pos_ref : float, dimension (natoms, 3)
                Position of all atoms in first/reference configuration
            nnlist : List,
                List of neighbors -- similar nindex

            Returns
            -------
            r_array : float, dimension (arbitrary,)
                array of all pairwise distances for the specified cutoff
    '''

    if nnlist != None:
        index = nnlist
        nn = np.array([len(l) for l in index], dtype=int)
    else:
        nn, index = verletlist(pos, h, rc)

    if pos.shape[0] != pos_ref.shape[0]:
        raise TypeError(
            'Initial configuration and the desired configuration do not have the same number of atoms/particles')
    # construct list of interatomic distances
    nparticles = pos.shape[0]
    hinv = np.linalg.inv(h)
    r_list = []
    for i in range(nparticles):
        ri = np.array([pos[i, :]])
        ind = index[i]
        rj = pos_ref[ind, :]
        drij = pbc(rj - np.repeat(ri, len(ind), axis=0), h, hinv)
        r_list.extend(drij.tolist())

    r_array = np.array(r_list)
    return r_array

#@jit(nopython=True, parallel=True)
def s_q3_from_pos(q3_flat, pos, nparticles):
    '''Get s(q) from positions over the entire 2D grid

            Parameters
            ----------
            pos : float, dimension (natoms, 3)
                Position of all atoms
            nparticles : int,
                Total number of atoms
            q3_flat: float, dimension(N-q*N-q, 3)
                Flatted array of wave-vectors

            Returns
            -------
            s_flat : float, dimension (N-q*N-q,)
                array of all s(q) over the q3_flat array
    '''

    s_flat = np.zeros(q3_flat.shape[0], dtype=np.complex64)
    for i in range(q3_flat.shape[0]):  # note: range -> prange
        q3 = q3_flat[i, :]
        q = np.linalg.norm(q3)
        if q < 1e-10:
            continue
        adj_ISF = np.exp(-1j * np.dot(pos, q3.T)).sum()
        s_flat[i] = adj_ISF * np.conj(adj_ISF) / nparticles
    return s_flat.real

#@jit(nopython=True, parallel=True)
def s_q3_from_pos_par(q3_flat, r_array, rc, rho, nparticles, smear=False, ddq=0, r_sq=None):
    '''Get s(q) from positions over the entire 2D grid

            Parameters
            ----------
            rc : float,
                Cutoff radius for g(r)
            rho : float,
                Atomic density
            nparticles : int,
                Total number of atoms
            q3_flat: float, dimension(N-q*N-q, 3)
                Flatted array of wave-vectors
            r_array : float, dimension (N_pairs)
                Pairwise distances/vectors for all valid pairs within a cutoff
            smear :  boolean
                If smearing of density field is being considered
            ddq : float,
                Width of the gaussian smear being used
            r_sq : float, dimension (N_pairs,)
                Squared distances from r_array

            Returns
            -------
            s_flat : float, dimension (N-q*N-q,)
                array of all s(q) over the q3_flat array
    '''
    s_flat = np.zeros(q3_flat.shape[0], dtype=np.complex64)
    for i in range(q3_flat.shape[0]):  # note: range -> prange
        q3 = q3_flat[i, :]
        q = np.linalg.norm(q3)
        if q < 1e-10:
            continue
        if smear:
            exp_miqr = np.exp(-1j * np.dot(r_array, q3.T)).flatten() * np.exp(-(0.5 * ddq ** 2) * r_sq ** 2)
            s_flat[i] = (1.0 + np.sum(exp_miqr) / nparticles)
        else:
            exp_miqr = np.exp(-1j * np.dot(r_array, q3.T))
            int_0_rc = (rc * q * np.cos(rc * q) - np.sin(rc * q)) / np.power(q, 3)  # integral from 0 to rc
            s_flat[i] = (1.0 + 4.0 * np.pi * rho * int_0_rc + np.sum(exp_miqr) / nparticles)
    return s_flat.real

def s_q_position_par(q3_pos):
    '''Get s(q) from positions on the q3_pos vector

            Parameters
            ----------
            q3_pos: float, dimension(natoms+1, 3)
                array of first row as wave-vector and the remaining rows as position of atoms
                -- function created for use in conjunction with multiprocessing

            Returns
            -------
            s_positon : float
                s(q) over the q3_pos wave-vector
    '''
    q3_array = q3_pos[0:1, :]
    pos = q3_pos[1:, :]
    N_atoms = q3_pos.shape[0] - 1
    s_position = np.zeros(1, dtype=np.complex)
    q3 = q3_array[0:1, :]
    q = np.linalg.norm(q3)
    adj_ISF = np.exp(-1j * np.dot(pos, q3.T)).sum()
    s_position = adj_ISF * np.conj(adj_ISF) / N_atoms
    return s_position

@jit(nopython=True, parallel=True)
def I_q3_from_pos_par(q3_flat, r_array, rc, rho, nparticles, ff, smear=False, ddq=0, r_sq=None):
    '''Get s(q) from positions over the entire 2D grid

            Parameters
            ----------
            rc : float,
                Cutoff radius for g(r)
            rho : float,
                Atomic density
            nparticles : int,
                Total number of atoms
            q3_flat: float, dimension(N-q*N-q, 3)
                Flatted array of wave-vectors
            r_array : float, dimension (N_pairs)
                Pairwise distances/vectors for all valid pairs within a cutoff
            smear :  boolean
                If smearing of density field is being considered
            ddq : float,
                Width of the gaussian smear being used
            r_sq : float, dimension (N_pairs,)
                Squared distances from r_array
            ff : float,
                form factor over k-space for a given scatterer

            Returns
            -------
            s_flat : float, dimension (N-q*N-q,)
                array of all s(q) over the q3_flat array
    '''

    s_flat = np.zeros(q3_flat.shape[0])
    for i in prange(q3_flat.shape[0]):  # note: range -> prange
        q3 = q3_flat[i, :]
        q = np.linalg.norm(q3)
        if q < 1e-10:
            continue
        if smear:
            exp_miqr = (((2 * np.pi) ** 0.5 * ddq) ** 3) * np.exp(-1j * np.dot(r_array, q3.T)).flatten() * np.exp(
                -(0.5 * ddq ** 2) * r_sq ** 2)
            int_0_rc = 0 * (-rc * ((2 * np.pi) ** 0.5) * np.cos(rc * q) * np.exp(
                -0.5 * (ddq ** 2) * rc ** 2) / ddq)  # integral from 0 to rc
            s_flat[i] = ((ddq * (2 * np.pi) ** 0.5) ** 3 + 4.0 * np.pi * rho * int_0_rc + np.sum(
                exp_miqr) / nparticles) * ff[i] ** 2
        else:
            exp_miqr = np.exp(-1j * np.dot(r_array, q3.T))
            int_0_rc = (rc * q * np.cos(rc * q) - np.sin(rc * q)) / np.power(q, 3)  # integral from 0 to rc
            s_flat[i] = (1.0 + 4.0 * np.pi * rho * int_0_rc + np.sum(exp_miqr) / nparticles) * ff[i] ** 2
    return s_flat.real

def ISF_from_pos_par(posit, s_ref=np.zeros((10, 10, 10)), N=200, wdt=500, cs=3, ms=30, ind_need=np.array([0, 2]),
                     dump=False, grid=False):
    '''Get I(q) from positions on desired indexes of the FFT grid

            Parameters
            ----------
            posit : float, dimension (natoms+2,3)
                Box + atoms in one array -- used in multiprocessing
            N : int,
                Number of grid points to dicretize the box into
            wdt : int,
                Number of grid points that specify gaussian smear width
            cs: int,
                Number of neighboring grids to consider in density field generation
            ms : int
                mini FFT grid size
            ind_need :  List
                List of all inidces in the minigrid over which the intensity is to be computed
            dump : boolean
                If dumpfile is being read
            grid : boolean
                Return whole minigrid instead of just indices

            Returns
            -------
            s_r : float, dimension (ms, ms, ms) or (len(ind_need))
                I(q) or s(q) results depending on need
    '''
    h = posit[:2, :].T
    pos = posit[2:, :]
    if dump:
        x = np.linspace(0, -h[0, 0] + h[0, 1], N + 1)
        y = np.linspace(0, -h[1, 0] + h[1, 1], N + 1)
        z = np.linspace(0, -h[2, 0] + h[2, 1], N + 1)
    else:
        x = np.linspace(h[0, 0], h[0, 1], N + 1)
        y = np.linspace(h[1, 0], h[1, 1], N + 1)
        z = np.linspace(h[2, 0], h[2, 1], N + 1)
    boxl = np.mean(h[:, 1] - h[:, 0])
    hx = h[0, 1] - h[0, 0]
    hy = h[1, 1] - h[1, 0]
    hz = h[2, 1] - h[2, 0]
    [X, Y, Z] = np.meshgrid(x, y, z)
    n_of_r = np.zeros(X.shape)
    NP = pos.shape[0]
    box_diag = h[:, 1] - h[:, 0]
    for i in range(NP):
        if dump:
            s_pos = (pos[i, :]) / box_diag
        else:
            s_pos = (pos[i, :] - h[:, 0]) / box_diag
        kx = int(np.floor(s_pos[0] * N))
        ky = int(np.floor(s_pos[1] * N))
        kz = int(np.floor(s_pos[2] * N))
        if (kx >= cs) and (kx < N - cs):
            indx = np.linspace(kx - cs, kx + cs, 2 * cs + 1, dtype=int)
        elif kx < cs:
            indx = np.append(np.linspace(N - (cs - kx), N - 1, cs - kx, dtype=int),
                             np.linspace(0, kx + cs, 2 * cs + 1 - (cs - kx), dtype=int))
        elif kx >= N - cs:
            indx = np.append(np.linspace(kx - cs, N - 1, cs + N - kx, dtype=int),
                             np.linspace(0, cs + kx - N, cs + kx - N + 1, dtype=int))
        if (ky >= cs) and (ky < N - cs):
            indy = np.linspace(ky - cs, ky + cs, 2 * cs + 1, dtype=int)
        elif ky < cs:
            indy = np.append(np.linspace(N - (cs - ky), N - 1, cs - ky, dtype=int),
                             np.linspace(0, ky + cs, 2 * cs + 1 - (cs - ky), dtype=int))
        elif ky >= N - cs:
            indy = np.append(np.linspace(ky - cs, N - 1, cs + N - ky, dtype=int),
                             np.linspace(0, cs + ky - N, cs + ky - N + 1, dtype=int))
        if (kz >= cs) and (kz < N - cs):
            indz = np.linspace(kz - cs, kz + cs, 2 * cs + 1, dtype=int)
        elif kz < cs:
            indz = np.append(np.linspace(N - (cs - kz), N - 1, cs - kz, dtype=int),
                             np.linspace(0, kz + cs, 2 * cs + 1 - (cs - kz), dtype=int))
        elif kz >= N - cs:
            indz = np.append(np.linspace(kz - cs, N - 1, cs + N - kz, dtype=int),
                             np.linspace(0, cs + kz - N, cs + kz - N + 1, dtype=int))
        indexes = np.ix_(indy, indx, indz)
        [XX, YY, ZZ] = np.meshgrid(x[indx], y[indy], z[indz])
        n_of_r[indexes] += norm3d(pos[i, :], XX, YY, ZZ, boxl / wdt, hx, hy, hz)

    s_r = np.fft.fftshift(np.fft.fftn(n_of_r))
    if grid:
        return s_r
    else:
        s_r = s_ref * np.flip(np.flip(np.flip(s_r, axis=0), axis=1), axis=2)
        xxx = np.arange(N // 2 - ms - 1, N // 2 + ms)
        NN = len(xxx)
        index2 = np.ix_(xxx, xxx, xxx)
        s_val = np.zeros((NN) ** 3)
        s_val[:] = np.reshape(s_r[index2], (1, -1))

        return s_val[ind_need]

def f_weight_array(g,r,rc):
    '''Get scaling weight for tailing g(r) smoothly to 1

            Parameters
            ----------
            rc : float,
                Cutoff radius for g(r)
            r: float, dimension (N_pairs,)
                Squared distances from r_array
            g: float, dimension (N_pairs,)
                g(r) over all the r_array distances

            Returns
            -------
            Weight : float, dimension (N_pairs,)
                Weight to be multiplied to generate a tail towards g(r)
    '''
    ri=0.95*rc
    weight=np.zeros(np.size(r))
    weight[r<ri]=1.0+(g[r<ri]-1.0)
    weight[(r>=ri) & (r<=rc)]=1.0+0.5*(1.0+np.cos(np.pi*((r[(r>=ri) & (r<=rc)]-ri)/(rc-ri))))*(g[(r>=ri) & (r<=rc)]-1.0)
    return weight

def f_weight_vec(r,rc):
    '''Get scaling weight for tailing g(r) smoothly to 1

            Parameters
            ----------
            rc : float,
                Cutoff radius for g(r)
            r: float, dimension (N_pairs,)
                Squared distances from r_array

            Returns
            -------
            Weight : float, dimension (N_pairs,)
                Weight to be multiplied to generate a tail towards g(r)
    '''
    ri=0.95*rc
    weight=np.zeros(np.size(r))
    weight[r<ri]=(r[r<ri])
    weight[(r>=ri) & (r<=rc)]=0.5*(1.0+np.cos(np.pi*((r[(r>=ri) & (r<=rc)]-ri)/(rc-ri))))*(r[(r>=ri) & (r<=rc)])
    return weight

def f_weight(r,rc):
    '''Get scaling weight for tailing g(r) smoothly to 1

            Parameters
            ----------
            rc : float,
                Cutoff radius for g(r)
            r: float, dimension (N_pairs,)
                Squared distances from r_array

            Returns
            -------
            Weight : float, dimension (N_pairs,)
                Weight to be multiplied to generate a tail towards g(r)
    '''
    ri=0.95*rc
    if r<ri:
        weight=1
    elif r>rc:
        weight=0
    else:
        weight=0.5*(1.0+cos(np.pi*((r-ri)/(rc-ri))))
    return weight

def auto_corr(s_cor,frames,g_2=True):
    '''Auto-correlation computation

            Parameters
            ----------
            s_cor : float, dimension (N-q,)
                Cutoff radius for g(r)
            frames: int,
                Frames or time upto which to compute the time auto-correlation
            g_2 : boolean
                True if g_2(q) normalization is required; False if F(q,t) normalization is required

            Returns
            -------
            Correlation : float, dimension (frames,)
                Array of correlation values
    '''
    ts_cor=np.zeros(frames)
    for i in range(frames):
        if i==0:
            f=s_cor
            g=s_cor
        else:
            f=s_cor[i:]
            g=s_cor[:-i]
        ts_cor[i]=np.mean(f*g)
    if g_2:
        return ts_cor/s_cor.mean()**2
    else:
        return ts_cor/ts_cor[0]

def ISF_corr(s_cor,frames,normed=True):
    '''Auto-correlation computation

            Parameters
            ----------
            s_cor : float, dimension (N-q,)
                Cutoff radius for g(r)
            frames: int,
                Frames or time upto which to compute the time auto-correlation
            normed : boolean
                Requirement of normalization (used mainly for F(q,t)

            Returns
            -------
            Correlation : float, dimension (frames,)
                Array of correlation values
    '''
    ts_cor=np.zeros(frames)
    for i in range(frames):
        if i==0:
            f=s_cor
            g=s_cor
        else:
            f=s_cor[i:]
            g=s_cor[:-i]
        ts_cor[i]=np.mean(f*np.conj(g))
    if normed:
        return ts_cor/ts_cor[0]
    else:
        return ts_cor

def norm3d(arr, x, y, z, sig, hx, hy, hz):
    '''Density field generation

            Parameters
            ----------
            arr : float, dimension (3,)
                Array of mean width density field in the x, y , z directions respectively
            x: float, dimension (Nx, Nx, Nx)
                mesh grid of x coordinates
            y: float, dimension (Nx, Nx, Nx)
                mesh grid of y coordinates
            z: float, dimension (Nx, Nx, Nx)
                mesh grid of z coordinates
            sig: float
                Smear width of density gaussian
            hx: float
                box length in x direction
            hy: float
                box length in y direction
            hz: float
                box length in z direction

            Returns
            -------
            gaussian density : float, dimension (Nx, Nx, Nx)
                Gaussian density on meshgrid defined by x, y, z
    '''
    mux, muy, muz = arr[0], arr[1], arr[2]
    return (1 / (np.sqrt(2 * np.pi * sig ** 2) ** 3)) * (np.exp(-0.5 * (pbc1d(x - mux, hx) / sig) ** 2) * np.exp(
        -0.5 * (pbc1d(y - muy, hy) / sig) ** 2) * np.exp(-0.5 * (pbc1d(z - muz, hz) / sig) ** 2))

@jit(nopython=True, parallel=True)
def convert_density(X, Y, Z, NP, pos, boxl, wdt, hx, hy, hz):
    '''Convert to atomic density using a gaussian smear

            Parameters
            ----------
            arr : float, dimension (3,)
                Array of mean width density field in the x, y , z directions respectively
            X: float, dimension (Nx, Nx, Nx)
                mesh grid of x coordinates
            Y: float, dimension (Nx, Nx, Nx)
                mesh grid of y coordinates
            Z: float, dimension (Nx, Nx, Nx)
                mesh grid of z coordinates
            sig: float
                Smear width of density gaussian
            boxl: float
                box length in x direction
            hx: float
                box length in x direction
            hy: float
                box length in y direction
            hz: float
                box length in z direction
            NP : int,
                Total number of atoms
            pos : float, dimension (NP, 3)
                Position of all atoms

            Returns
            -------
            gaussian density : float, dimension (Nx, Nx, Nx)
                Gaussian density on meshgrid defined by x, y, z
    '''
    n_of_r = np.zeros(X.shape)
    sig = boxl / wdt
    a = 0
    for i in prange(NP):
        #	n_of_r+=norm3d(X,Y,Z,pos[i,0],pos[i,1],pos[i,2],boxl/wdt,hx,hy,hz)
        pbcx = ((pos[i, 0] - X) / hx - np.round_((pos[i, 0] - X) / hx, 0, a)) * hx
        pbcy = ((pos[i, 1] - Y) / hy - np.round_((pos[i, 1] - Y) / hy, 0, a)) * hy
        pbcz = ((pos[i, 2] - Z) / hz - np.round_((pos[i, 2] - Z) / hz, 0, a)) * hz
        n_of_r += (1 / (np.sqrt(2 * np.pi * sig ** 2) ** 3)) * (
                    np.exp(-0.5 * (pbcx / sig) ** 2) * np.exp(-0.5 * (pbcy / sig) ** 2) * np.exp(
                -0.5 * (pbcz / sig) ** 2))

    return n_of_r

def s_q_from_pos_smear_analytical(q_array, pos, h, rc, rho, nnlist=None):
    '''Obtain s(q) from analytical expression -- internally computes r_array from g(r)

            Parameters
            ----------
            q_array : float, dimension (N-q, 3)
                Array of all wave-vectors for s(q) computation
            pos : float, dimension (natoms, 3)
                Position of all atoms
            rc : float,
                Cutoff radius for g(r)
            h : float, dimension (3, 3)
                Periodic box size h = (c1|c2|c3)
            rho : float,
                Atomic density
            nnlist : List,
                List of neighbors -- similar nindex

            Returns
            -------
            s_array : float, dimension (N-q,)
                s(q) computed for given q_array
    '''
    if nnlist != None:
        index = nnlist
        nn = np.array([len(l) for l in index], dtype=int)
    else:
        nn, index = verletlist(pos, h, rc)

    # construct list of interatomic distances
    nparticles = pos.shape[0]
    hinv = np.linalg.inv(h)
    dist = []
    for i in range(nparticles):
        ri = np.array([pos[i, :]])
        ind = index[i]
        rj = pos[ind, :]
        drij = pbc(rj - np.repeat(ri, len(ind), axis=0), h, hinv)
        dist.extend(np.linalg.norm(drij, axis=1).tolist())

    r_array = np.array(dist)
    #    r_array= f_weight_vec(r_array,rc)
    s_array = np.zeros(q_array.shape)
    for i in range(q_array.shape[0]):
        q = q_array[i]
        sinqr_div_r = np.exp(-1j * r_array * q)
        int_0_rc = (rc * q * np.cos(rc * q) - np.sin(rc * q)) / np.power(q, 3)  # integral from 0 to rc
        s_array[i] = 1.0 + 4.0 * np.pi * rho * int_0_rc + np.sum(sinqr_div_r) / (
            nparticles)  # np.sum(sinqr_div_r)/(nparticles)
    return s_array.real

def accum_np(accmap, a, func=np.mean):
    indices = np.where(np.ediff1d(accmap, to_begin=[1],
                                  to_end=[1]))[0]
    vals = np.zeros(len(indices) - 1)
    for i in range(len(indices) - 1):
        vals[i] = func(a[indices[i]:indices[i + 1]])
    return vals

def s_q_from_pos_smear(posit, N=200, wdt=500, cs=3, ms=30, dump=False, intensity=False, movie_plot=False, ISF=False,
                       structure_factor=False, correction=False, correction_grid=None, q_magnitude=None, ind_need=None):
    '''Get I(q) from positions on desired indexes of the FFT grid

            Parameters
            ----------
            posit : float, dimension (natoms+2,3)
                Box + atoms in one array -- used in multiprocessing
            N : int,
                Number of grid points to dicretize the box into
            wdt : int,
                Number of grid points that specify gaussian smear width
            cs: int,
                Number of neighboring grids to consider in density field generation
            ms : int
                mini FFT grid size
            ind_need :  List
                List of all inidces in the minigrid over which the intensity is to be computed
            dump : boolean
                If dumpfile is being read
            intensity : boolean
                If I(q) needs to be computed
            ISF : boolean
                If F(q) needs to be computed
            structure : boolean
                If s(q) needs to be computed
            correction_grid : float, dimension (N,N,N)
                Correction grid value
            correction : boolean
                If correction factor is needed
            q_magnitude : float, dimension (N,N,N)
                Needeed to compute I(q) from s(q)
            Returns
            -------
            s_r : float, dimension (ms, ms, ms) or (len(ind_need))
                I(q) or s(q) results depending on need
    '''
    if correction is False:
        if (int(intensity) + int(ISF) + int(structure_factor)) != 1:
            raise TypeError('Choose one and only one output entity')
    h = posit[:2, :].T
    pos = posit[2:, :]
    boxl = np.mean(h[:, 1] - h[:, 0])
    hx = h[0, 1] - h[0, 0]
    hy = h[1, 1] - h[1, 0]
    hz = h[2, 1] - h[2, 0]
    s = pos / boxl
    s = s - 0.5 - np.round(s - 0.5)
    pos = boxl * (s + 0.5)
    delx = hx / N
    if dump:
        x = np.linspace(0, hx - delx, N)
        y = np.linspace(0, hy - delx, N)
        z = np.linspace(0, hz - delx, N)
    else:
        x = np.linspace(h[0, 0], h[0, 1] - delx, N)
        y = np.linspace(h[1, 0], h[1, 1] - delx, N)
        z = np.linspace(h[2, 0], h[2, 1] - delx, N)
    [X, Y, Z] = np.meshgrid(x, y, z)
    n_of_r = np.zeros(X.shape)
    s_r = np.zeros(X.shape, dtype=np.complex)
    NP = pos.shape[0]
    box_diag = h[:, 1] - h[:, 0]
    for i in range(NP):
        if dump:
            s_pos = (pos[i, :]) / box_diag
        else:
            s_pos = (pos[i, :] - h[:, 0]) / box_diag
        kx = int(np.floor(s_pos[0] * N))
        ky = int(np.floor(s_pos[1] * N))
        kz = int(np.floor(s_pos[2] * N))
        if (kx >= cs) and (kx < N - cs):
            indx = np.linspace(kx - cs, kx + cs, 2 * cs + 1, dtype=int)
        elif kx < cs:
            indx = np.append(np.linspace(N - (cs - kx), N - 1, cs - kx, dtype=int),
                             np.linspace(0, kx + cs, 2 * cs + 1 - (cs - kx), dtype=int))
        elif kx >= N - cs:
            indx = np.append(np.linspace(kx - cs, N - 1, cs + N - kx, dtype=int),
                             np.linspace(0, cs + kx - N, cs + kx - N + 1, dtype=int))
        if (ky >= cs) and (ky < N - cs):
            indy = np.linspace(ky - cs, ky + cs, 2 * cs + 1, dtype=int)
        elif ky < cs:
            indy = np.append(np.linspace(N - (cs - ky), N - 1, cs - ky, dtype=int),
                             np.linspace(0, ky + cs, 2 * cs + 1 - (cs - ky), dtype=int))
        elif ky >= N - cs:
            indy = np.append(np.linspace(ky - cs, N - 1, cs + N - ky, dtype=int),
                             np.linspace(0, cs + ky - N, cs + ky - N + 1, dtype=int))
        if (kz >= cs) and (kz < N - cs):
            indz = np.linspace(kz - cs, kz + cs, 2 * cs + 1, dtype=int)
        elif kz < cs:
            indz = np.append(np.linspace(N - (cs - kz), N - 1, cs - kz, dtype=int),
                             np.linspace(0, kz + cs, 2 * cs + 1 - (cs - kz), dtype=int))
        elif kz >= N - cs:
            indz = np.append(np.linspace(kz - cs, N - 1, cs + N - kz, dtype=int),
                             np.linspace(0, cs + kz - N, cs + kz - N + 1, dtype=int))
        indexes = np.ix_(indy, indx, indz)
        [XX, YY, ZZ] = np.meshgrid(x[indx], y[indy], z[indz])
        n_of_r[indexes] += norm3d(pos[i, :], XX, YY, ZZ, boxl / wdt, hx, hy, hz)
    s_r = np.fft.fftshift(np.fft.fftn(n_of_r)) * (boxl / (N)) ** 3
    p_r = s_r / NP
    s_r = s_r * np.conj(s_r) / NP
    if correction:
        if ISF:
            return (p_r / NP) ** -1
        else:
            return s_r ** -1
    else:
        xxx = np.linspace(N // 2 - ms, N // 2 + ms, 2 * ms + 1, dtype=np.int)
        NN = len(xxx)
        index2 = np.ix_(xxx, xxx, xxx)
        [XXX, YYY, ZZZ] = np.meshgrid(xxx, xxx, xxx)
        if correction_grid is None:
            if ind_need is None:
                if intensity:
                    if q_magnitude is None:
                        raise TypeError("magnitude of q-space is necessary for computing the intensity")
                    ret_val = s_r * form_factor_analytical(q_magnitude) ** 2
                    return ret_val
                elif ISF is True:
                    return p_r
                elif structure_factor is True:
                    return s_r
            else:
                if intensity:
                    if q_magnitude is None:
                        raise TypeError("magnitude of q-space is necessary for computing the intensity")
                    ret_val = s_r * form_factor_analytical(q_magnitude) ** 2
                    I_q = ret_val[index2].reshape(-1, 1)
                    return I_q[ind_need].flatten()
                elif ISF is True:
                    p_r = p_r[index2].reshape(-1, 1)
                    return p_r[ind_need].flatten()
                elif structure_factor is True:
                    s_r = s_r[index2].reshape(-1, 1)
                    return s_r[ind_need].flatten()

def s_q_from_pos_smear_array(pos, h, N=200, wdt=500, cs=3, ms=30, uniform_density=True, dump=False,
                             correction_grid=None):
    '''Get I(q) from positions on a line of spherically averaged wave-vectors

            Parameters
            ----------
            posit : float, dimension (natoms+2,3)
                Box + atoms in one array -- used in multiprocessing
            N : int,
                Number of grid points to dicretize the box into
            wdt : int,
                Number of grid points that specify gaussian smear width
            cs: int,
                Number of neighboring grids to consider in density field generation
            ms : int
                mini FFT grid size
            dump : boolean
                If dumpfile is being read
            correction_grid : float, dimension (N,N,N)
                Correction grid value
            Returns
            -------
            r_un : float, dimension (arb,)
                q_value array
            signa : float, dimension (arb,)
                angle averaged s(q) on the r_un line of wavevectors
    '''
    h = pos[:2, :].T
    boxl = np.mean(h[:, 1] - h[:, 0])
    s_r = s_q_from_pos_smear(pos, N=N, wdt=wdt, cs=cs, ms=ms, dump=dump, structure_factor=True)
    s_r = correction_grid * s_r
    xxx = np.linspace(N // 2 - ms, N // 2 + ms, 2 * ms + 1, dtype=np.int)
    NN = len(xxx)
    index2 = np.ix_(xxx, xxx, xxx)
    [XXX, YYY, ZZZ] = np.meshgrid(xxx, xxx, xxx)
    s_val = np.zeros([(NN) ** 3, 2])
    posit = np.zeros([(NN) ** 3, 3])
    [XXX, YYY, ZZZ] = np.meshgrid(xxx, xxx, xxx)
    posit[:, 0] = np.reshape(XXX, (1, -1))
    posit[:, 1] = np.reshape(YYY, (1, -1))
    posit[:, 2] = np.reshape(ZZZ, (1, -1))
    s_val[:, 0] = np.linalg.norm((posit - N // 2) / boxl, axis=1)
    s_val[:, 1] = np.reshape(s_r[index2], (1, -1))
    s_val = s_val[np.argsort(s_val[:, 0])]
    r_un, ia, idx = np.unique(s_val[:, 0], return_index=True, return_inverse=True)
    signa = accum_np(idx, s_val[:, 1])
    return r_un, signa

def local_sq(pos, h=None, N=200, wdt=500, cs=3, ms=30, dump=False, correction_grid=None):
    '''Parallelize s_q_from_pos_smear_array for multiprocessing

            Parameters
            ----------
            pos : float, dimension (natoms,3)
                atoms in one array -- used in multiprocessing
            N : int,
                Number of grid points to dicretize the box into
            wdt : int,
                Number of grid points that specify gaussian smear width
            cs: int,
                Number of neighboring grids to consider in density field generation
            ms : int
                mini FFT grid size
            ind_need :  List
                List of all inidces in the minigrid over which the intensity is to be computed
            dump : boolean
                If dumpfile is being read
            correction_grid : float, dimension (N,N,N)
                Correction grid value
            Returns
            -------
            array_q : float, dimension (arb,)
                angle averaged s(q) on the r_un line of wavevectors
    '''
    _, array_sq = s_q_from_pos_smear_array(pos, h=h, N=N, wdt=wdt, cs=cs, ms=ms, dump=dump,
                                           correction_grid=correction_grid)
    return array_sq

def s_q_from_pos_smear_par(posit, N=200, wdt=500, cs=3, ms=30, ind_need=np.array([0, 2]), dump=False):
    '''Get I(q) from positions on desired indexes of the FFT grid

            Parameters
            ----------
            posit : float, dimension (natoms+2,3)
                Box + atoms in one array -- used in multiprocessing
            N : int,
                Number of grid points to dicretize the box into
            wdt : int,
                Number of grid points that specify gaussian smear width
            cs: int,
                Number of neighboring grids to consider in density field generation
            ms : int
                mini FFT grid size
            ind_need :  List
                List of all inidces in the minigrid over which the intensity is to be computed
            dump : boolean
                If dumpfile is being read

            Returns
            -------
            s_r : float, dimension (ms, ms, ms) or (len(ind_need))
                I(q) or s(q) results depending on need
    '''
    h = posit[:2, :].T
    pos = posit[2:, :]
    if dump:
        x = np.linspace(0, hx - delx, N)
        y = np.linspace(0, hy - delx, N)
        z = np.linspace(0, hz - delx, N)
    else:
        x = np.linspace(h[0, 0], h[0, 1] - delx, N)
        y = np.linspace(h[1, 0], h[1, 1] - delx, N)
        z = np.linspace(h[2, 0], h[2, 1] - delx, N)
    boxl = np.mean(h[:, 1] - h[:, 0])
    hx = h[0, 1] - h[0, 0]
    hy = h[1, 1] - h[1, 0]
    hz = h[2, 1] - h[2, 0]
    s = pos / boxl
    s = s - 0.5 - np.round(s - 0.5)
    pos = boxl * (s + 0.5)
    [X, Y, Z] = np.meshgrid(x, y, z)
    n_of_r = np.zeros(X.shape)
    NP = pos.shape[0]
    box_diag = h[:, 1] - h[:, 0]
    for i in range(NP):
        if dump:
            s_pos = (pos[i, :]) / box_diag
        else:
            s_pos = (pos[i, :] - h[:, 0]) / box_diag
        kx = int(np.floor(s_pos[0] * N))
        ky = int(np.floor(s_pos[1] * N))
        kz = int(np.floor(s_pos[2] * N))
        if (kx >= cs) and (kx < N - cs):
            indx = np.linspace(kx - cs, kx + cs, 2 * cs + 1, dtype=int)
        elif kx < cs:
            indx = np.append(np.linspace(N - (cs - kx), N - 1, cs - kx, dtype=int),
                             np.linspace(0, kx + cs, 2 * cs + 1 - (cs - kx), dtype=int))
        elif kx >= N - cs:
            indx = np.append(np.linspace(kx - cs, N - 1, cs + N - kx, dtype=int),
                             np.linspace(0, cs + kx - N, cs + kx - N + 1, dtype=int))
        if (ky >= cs) and (ky < N - cs):
            indy = np.linspace(ky - cs, ky + cs, 2 * cs + 1, dtype=int)
        elif ky < cs:
            indy = np.append(np.linspace(N - (cs - ky), N - 1, cs - ky, dtype=int),
                             np.linspace(0, ky + cs, 2 * cs + 1 - (cs - ky), dtype=int))
        elif ky >= N - cs:
            indy = np.append(np.linspace(ky - cs, N - 1, cs + N - ky, dtype=int),
                             np.linspace(0, cs + ky - N, cs + ky - N + 1, dtype=int))
        if (kz >= cs) and (kz < N - cs):
            indz = np.linspace(kz - cs, kz + cs, 2 * cs + 1, dtype=int)
        elif kz < cs:
            indz = np.append(np.linspace(N - (cs - kz), N - 1, cs - kz, dtype=int),
                             np.linspace(0, kz + cs, 2 * cs + 1 - (cs - kz), dtype=int))
        elif kz >= N - cs:
            indz = np.append(np.linspace(kz - cs, N - 1, cs + N - kz, dtype=int),
                             np.linspace(0, cs + kz - N, cs + kz - N + 1, dtype=int))
        indexes = np.ix_(indy, indx, indz)
        [XX, YY, ZZ] = np.meshgrid(x[indx], y[indy], z[indz])
        n_of_r[indexes] += norm3d(pos[i, :], XX, YY, ZZ, boxl / wdt, hx, hy, hz)

    s_r = np.fft.fftshift(np.fft.fftn(n_of_r))
    s_r = s_r * np.conj(s_r)
    xxx = np.arange(N // 2 - ms - 1, N // 2 + ms)
    NN = len(xxx)
    index2 = np.ix_(xxx, xxx, xxx)
    s_val = np.zeros((NN) ** 3)
    s_val[:] = np.reshape(s_r[index2], (1, -1))
    return s_val[ind_need]

def I_q_from_pos_smear_par(posit, N=200, wdt=500, cs=3, ms=30, ind_need=np.array([0, 2]), dump=False, movie_plot=False,
                           ISF=False, coarse_grain=False, grid_type='3D'):
    '''Get I(q) from positions on desired indexes of the FFT grid

            Parameters
            ----------
            posit : float, dimension (natoms+2,3)
                Box + atoms in one array -- used in multiprocessing
            N : int,
                Number of grid points to dicretize the box into
            wdt : int,
                Number of grid points that specify gaussian smear width
            cs: int,
                Number of neighboring grids to consider in density field generation
            ms : int
                mini FFT grid size
            ind_need :  List
                List of all inidces in the minigrid over which the intensity is to be computed
            dump : boolean
                If dumpfile is being read
            ISF : boolean
                If F(q) needs to be computed or s(q)
            movie_plot : boolean
                If a 2D frame needs to be generated for movie of 2D snapshots
            coarse_grain : int
                number of grid points to combine to coarse grain the signal
            grid_type : str
                '3D' or 2D'

            Returns
            -------
            s_r : float, dimension (ms, ms, ms) or (len(ind_need))
                I(q) or s(q) results depending on need
    '''
    h = posit[:2, :].T
    pos = posit[2:, :]
    boxl = np.mean(h[:, 1] - h[:, 0])
    hx = h[0, 1] - h[0, 0]
    hy = h[1, 1] - h[1, 0]
    hz = h[2, 1] - h[2, 0]
    s = pos / boxl
    s = s - 0.5 - np.round(s - 0.5)
    pos = boxl * (s + 0.5)
    delx = hx / N
    if dump:
        x = np.linspace(0, hx - delx, N)
        y = np.linspace(0, hy - delx, N)
        z = np.linspace(0, hz - delx, N)
    else:
        x = np.linspace(h[0, 0], h[0, 1] - delx, N)
        y = np.linspace(h[1, 0], h[1, 1] - delx, N)
        z = np.linspace(h[2, 0], h[2, 1] - delx, N)
    [X, Y, Z] = np.meshgrid(x, y, z)
    n_of_r = np.zeros(X.shape)
    s_r = np.zeros(X.shape, dtype=np.complex)
    NP = pos.shape[0]
    box_diag = h[:, 1] - h[:, 0]
    for i in range(NP):
        if dump:
            s_pos = (pos[i, :]) / box_diag
        else:
            s_pos = (pos[i, :] - h[:, 0]) / box_diag
        kx = int(np.floor(s_pos[0] * N))
        ky = int(np.floor(s_pos[1] * N))
        kz = int(np.floor(s_pos[2] * N))
        if (kx >= cs) and (kx < N - cs):
            indx = np.linspace(kx - cs, kx + cs, 2 * cs + 1, dtype=int)
        elif kx < cs:
            indx = np.append(np.linspace(N - (cs - kx), N - 1, cs - kx, dtype=int),
                             np.linspace(0, kx + cs, 2 * cs + 1 - (cs - kx), dtype=int))
        elif kx >= N - cs:
            indx = np.append(np.linspace(kx - cs, N - 1, cs + N - kx, dtype=int),
                             np.linspace(0, cs + kx - N, cs + kx - N + 1, dtype=int))
        if (ky >= cs) and (ky < N - cs):
            indy = np.linspace(ky - cs, ky + cs, 2 * cs + 1, dtype=int)
        elif ky < cs:
            indy = np.append(np.linspace(N - (cs - ky), N - 1, cs - ky, dtype=int),
                             np.linspace(0, ky + cs, 2 * cs + 1 - (cs - ky), dtype=int))
        elif ky >= N - cs:
            indy = np.append(np.linspace(ky - cs, N - 1, cs + N - ky, dtype=int),
                             np.linspace(0, cs + ky - N, cs + ky - N + 1, dtype=int))
        if (kz >= cs) and (kz < N - cs):
            indz = np.linspace(kz - cs, kz + cs, 2 * cs + 1, dtype=int)
        elif kz < cs:
            indz = np.append(np.linspace(N - (cs - kz), N - 1, cs - kz, dtype=int),
                             np.linspace(0, kz + cs, 2 * cs + 1 - (cs - kz), dtype=int))
        elif kz >= N - cs:
            indz = np.append(np.linspace(kz - cs, N - 1, cs + N - kz, dtype=int),
                             np.linspace(0, cs + kz - N, cs + kz - N + 1, dtype=int))
        indexes = np.ix_(indy, indx, indz)
        [XX, YY, ZZ] = np.meshgrid(x[indx], y[indy], z[indz])
        n_of_r[indexes] += norm3d(pos[i, :], XX, YY, ZZ, boxl / wdt, hx, hy, hz)

    s_r = np.fft.fftshift(np.fft.fftn(n_of_r)) * (boxl / (N + 1)) ** 3
    if ISF:
        p_temp = s_r
    s_r = s_r * np.conj(s_r) / NP
    if coarse_grain:
        s_r_old = s_r.copy()
        N = N // 2
        s_r = np.zeros((N, N, N))
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    s_r[i, j, k] = np.sum(
                        np.sum(np.sum(s_r_old[2 * i:2 * i + 2, 2 * j:2 * j + 2, 2 * k:2 * k + 2], axis=0), axis=1))
    xxx = np.linspace(N // 2 - ms, N // 2 + ms, 2 * ms + 1, dtype=np.int)
    NN = len(xxx)
    index2 = np.ix_(xxx, xxx, xxx)
    [XXX, YYY, ZZZ] = np.meshgrid(xxx, xxx, xxx)
    if grid_type == '3D':
        posit = np.zeros([(NN) ** 3, 3])
        posit[:, 0] = np.reshape(XXX, (1, -1))
        posit[:, 1] = np.reshape(YYY, (1, -1))
        posit[:, 2] = np.reshape(ZZZ, (1, -1))
        s_val = np.zeros((NN) ** 3)
        s_val[:] = np.reshape(s_r[index2], (1, -1))
    else:
        posit = np.zeros([(NN) ** 2, 3])
        [XXX, YYY, ZZZ] = np.meshgrid(xxx, xxx, xxx)
        posit[:, 0] = np.reshape(XXX[:, NN // 2, :], (1, -1))
        posit[:, 1] = np.reshape(YYY[:, NN // 2, :], (1, -1))
        posit[:, 2] = np.reshape(ZZZ[:, NN // 2, :], (1, -1))
        ts_val = s_r[index2]
        s_val = np.zeros((NN) ** 2)
        s_val[:] = np.reshape(ts_val[:, NN // 2, :], (1, -1))
    if coarse_grain:
        d_val = np.linalg.norm(2.0 * (posit - N // 2) / boxl, axis=1)
    else:
        d_val = np.linalg.norm((posit - N // 2) / boxl, axis=1)
    # ret_val=np.exp((2*np.pi*d_val)**2*(boxl/wdt)**2)*s_val*form_factor_analytical(2*np.pi*d_val)**2
    s_val = np.exp((2 * np.pi * d_val) ** 2 * (boxl / wdt) ** 2) * s_val
    ret_val = s_val * form_factor_analytical(2 * np.pi * d_val) ** 2
    # ret_val = ret_val.reshape(XXX.shape)
    s_val = s_val.reshape(-1, 1)
    if ISF:
        if grid_type == '3D':
            p_val = np.zeros((NN) ** 3, dtype=np.complex)
            p_val[:] = np.reshape(p_temp[index2], (1, -1))
        else:
            ps = p_temp[index2]
            p_val = np.zeros((NN) ** 2, dtype=np.complex)
            p_val[:] = np.reshape(ps[:, NN // 2, :], (1, -1))
        p_r = np.exp((2 * np.pi * d_val) ** 2 * (boxl / wdt) ** 2) * p_val  # *form_factor_analytical(2*np.pi*d_val)
        p_r = p_r * np.conj(p_r) / NP
        if movie_plot:
            ret_grid = ret_val.reshape((NN, NN, NN))
            return ret_grid[NN // 2, :, :]
        else:
            return p_r[ind_need]
    else:
        if movie_plot:
            ret_grid = ret_val.reshape((NN, NN, NN))
            return ret_grid[NN // 2, :, :]
        else:
            return s_val[ind_need].flatten()

def form_factor_analytical(q_val, atom_type="Ar"):
    '''Computing form factor at a given wave-vector magnitude

            Parameters
            ----------
            q_val : float, dimension (N-q)
                Array of all wave-vectors for form factor computation
            atom_type : str
                Atom type for form factor estimation

            Returns
            -------
            f_val : float, dimension (q_val,)
                form factor computed for given q_val array
    '''
    if atom_type == "Ar":
        a = np.array([7.4845,6.7723,0.6539,1.6442])
        b = np.array([0.9072,14.8407,43.8983,33.3929])
        c = 1.4445
    elif atom_type == "C":
        a = np.array([2.31,1.02,1.5886,0.865])
        b = np.array([20.8439,10.2075,0.5687,51.6512])
        c = 0.2156
    elif atom_type == "H":
        a = np.array([0.489918,0.262003,0.196767,0.049879])
        b = np.array([20.6593,7.74039,49.5519,2.20159])
        c = 0.001305
    elif atom_type == "O":
        a = np.array([3.0485,2.2868,1.5463,0.867])
        b = np.array([13.2771,5.7011,0.3239,32.9089])
        c = 0.2508
    elif atom_type == "Cs":
        a = np.array([20.3892,19.1062,10.662,1.4953])
        b = np.array([3.569,0.3107,24.3879,213.904])
        c = 3.3352
    elif atom_type == "Pb":
        a = np.array([31.0617,13.0637,18.442,5.9696])
        b = np.array([0.6902,2.3576,8.618,47.2579])
        c = 13.4118
    elif atom_type == "I":
        a = np.array([20.1472,18.9949,7.5138,2.2735])
        b = np.array([4.347,0.3814,27.766,66.8776])
        c = 4.0712
    else:
        a = np.array([0,0,0,0])
        b = np.array([0,0,0,0])
        c = 0
    f_val = c 
    for i in range(4):
        f_val += a[i]*np.exp(-b[i]*(q_val/(4*np.pi))**2)

    return f_val

def optical_contrast(s):
    '''Compute optical contrast from an s(q) or I(q) array

            Parameters
            ----------
            s : float, dimension (N,)
                Array of s(q) or I(q) in either q-space or t-space

            Returns
            -------
            contrast : float,
                optical contrast
    '''
    return ((s ** 2).mean() - (s.mean()) ** 2) / (s.mean()) ** 2

def cross_corr(s, frames):
    '''Time cross-correlation (two-time correlation) computation

            Parameters
            ----------
            s : float, dimension (N-q,)
                Cutoff radius for g(r)
            frames: int,
                Frames or time upto which to compute the time auto-correlation

            Returns
            -------
            Correlation : float, dimension (frames,)
                Array of time cross-correlation grid
    '''
    ts = np.zeros((frames, frames))
    for i in range(frames):
        for j in range(i, frames):
            if i == j:
                f = s[i:]
                g = s[i:]
            else:
                f = s[j:]
                g = s[i:-(j - i)]
            ts[i, j] = (np.mean(f * g) - np.mean(f) * np.mean(g)) / (((np.mean(f ** 2) - np.mean(f) ** 2) ** 0.5) * (
            (np.abs(np.mean(g ** 2) - np.mean(g) ** 2) ** 0.5)))
            if i != j:
                ts[j, i] = ts[i, j]
    return ts

def end_to_end_pos_to_g2(pos, q_val, frames, pool, save_file = 'sample', atoms_add = 40, h_in = None, total_atoms = 200):
    '''Get I(q) from positions on desired indexes of the FFT grid

            Parameters
            ----------
            pos : float, dimension (natoms,3)
                atoms in one array -- used in multiprocessing
            frames: int,
                Frames or time upto which to compute the time auto-correlation
            pool : multiprocessing object,
                Pool of cpus for multiprocessing
            q_val : float, dimension (N-q, 3)
                Array of all wave-vectors for form factor computation
            save_file : str
                Name of npz file to save s(q), g2(q,t) and F(q) array to
            atoms_add :  int
                Number of atoms to compute the XPCS signal over
            h_in : float, dimension (3,2)
                hi and lo values in x, y, and z direction
            total_atoms : int
                If position array doesn't encode all atoms

            Returns
            -------
            s_r : float, dimension (ms, ms, ms) or (len(ind_need))
                I(q) or s(q) results depending on need
    '''
    if len(pos.shape) == 3:
        total_steps = pos.shape[2]
        N_atoms = pos.shape[0]
        pos = pos.reshape(pos.shape[0]*pos.shape[2],pos.shape[1])
    else:
        N_atoms = total_atoms
        total_steps = pos.shape[0]//N_atoms
    if h_in is None:
        boxl = 17.0
    else:
        boxl = h_in
    q3_eff = q_val ** 2
    t_size = 0.05
    lags = np.linspace(0, t_size * (total_steps - 1), total_steps)
    sigma_grid = 400
    N_grid = 400
    mini_grid = 40
    density_cutoff = 5
    q_tol = 0.05
    I_Q = True
    position = 'time'
    s = pos / boxl
    s = s - 0.5
    s = s - np.round(s)
    pos = boxl * (s + 0.5)
    scale_pos = np.zeros((3, 3))
    h = np.array([[0.0,boxl],[0.0,boxl],[0.0,boxl]])
    scale_pos[:2, :] = h.T
    scale_pos[2, :] = boxl * np.array([0.0, 0.0, 0.0])
    rescale_factor = s_q_from_pos_smear(scale_pos, N=N_grid, wdt=sigma_grid,
                                        cs=density_cutoff, ms=mini_grid, dump=True, structure_factor=True,
                                        correction=True)
    rescale_factor_ISF = s_q_from_pos_smear(scale_pos, N=N_grid, wdt=sigma_grid,
                                            cs=density_cutoff, ms=mini_grid, dump=True, ISF=True, correction=True)
    if position == 'time':
        if atoms_add is None:
            atoms_add = N_atoms
        else:
            atoms_add = atoms_add
        pos_input = np.zeros((atoms_add + 2, 3, total_steps))
        for i in range(total_steps):
            temp_pos = pos[i * N_atoms:i * N_atoms + atoms_add, :]
            pos_input[:2, :, i] = h.T
            pos_input[2:, :, i] = temp_pos
    x_grid = np.linspace(N_grid // 2 - mini_grid, N_grid // 2 + mini_grid, 2 * mini_grid + 1, dtype=np.int)
    mini_grid_index = np.ix_(x_grid, x_grid, x_grid)
    q_grid = 2 * np.pi * np.linspace(-(N_grid // 2), N_grid // 2 - 1, N_grid, dtype=np.int) / boxl
    [Qx, Qy, Qz] = np.meshgrid(q_grid, q_grid, q_grid)
    Q_line = np.zeros((N_grid ** 3, 3))
    Q_line[:, 0] = Qx.reshape(-1, 1).flatten()
    Q_line[:, 1] = Qy.reshape(-1, 1).flatten()
    Q_line[:, 2] = Qz.reshape(-1, 1).flatten()
    q_grid_magnitude = np.linalg.norm(Q_line, axis=1).reshape(Qx.shape)
    q_probe = q_grid_magnitude[mini_grid_index]
    correct_grid = rescale_factor[mini_grid_index].reshape(-1, 1)
    correct_grid_ISF = rescale_factor_ISF[mini_grid_index].reshape(-1, 1)
    probe_index = np.where(abs(q_probe.reshape(-1, 1) - q_val) < q_tol)[0]
    if len(probe_index) == 0:
        raise TypeError('No points selected')
    compute_partial = partial(s_q_from_pos_smear, N=N_grid, wdt=sigma_grid,
                              cs=density_cutoff, ms=mini_grid, dump=True, structure_factor=True,
                              q_magnitude=q_grid_magnitude, ind_need=probe_index)
    s_time = pool.map(compute_partial, [pos_input[:, :, r] for r in range(total_steps)])
    s_time = np.asarray(s_time)
    s_time = s_time.T.real
    s_time = np.dot(np.diag(correct_grid[probe_index].flatten()), s_time)
    compute_partial_ISF = partial(s_q_from_pos_smear, N=N_grid, wdt=sigma_grid,
                                  cs=density_cutoff, ms=mini_grid, dump=True, ISF=True, q_magnitude=q_grid_magnitude,
                                  ind_need=probe_index)
    s_time_ISF = pool.map(compute_partial_ISF, [pos_input[:, :, r] for r in range(total_steps)])
    s_time_ISF = np.asarray(s_time_ISF)
    s_time_ISF = s_time_ISF.T.real
    s_time_ISF = np.dot(np.diag(correct_grid_ISF[probe_index].flatten()), s_time_ISF)
    correl = partial(auto_corr, frames=frames)
    correl_ISF = partial(ISF_corr, frames=frames)
    items = s_time.shape[0]
    g2 = pool.map(correl, [s_time[i, :] for i in range(items)])
    g2 = np.asarray(g2)
    f_qt = pool.map(correl_ISF, [s_time_ISF[i, :] for i in range(items)])
    f_qt = np.asarray(f_qt)
    F_qt = np.mean(f_qt, axis=0) ** 2
    beta = g2[:, 0] - 1.0
    g2_exp_fit = np.dot(np.diag(1.0 / (beta)), (g2 - 1.0))
    g2_exp_fit_mean = np.mean(g2_exp_fit, axis=0)
    g2_mean = np.mean(g2, axis=0)
    np.savez('%s_%.2f_%d.npz' % (save_file, q_val, atoms_add), g2_m = g2_mean, g2_em = g2_exp_fit_mean, F = F_qt)

def gr_MD(pos,bins,rc,h):
    _, g_array = g_r_verlet(pos,bins,rc,h)
    return g_array
