# --------------------------------------------------------
# md_util.py
# by Shaswat Mohanty, shaswatm@stanford.edu
#
# Objectives
# Utility functions for analysis of MD trajectory
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
from load_util import *


def objective(y,a):
    return a*y

def objective_lin(x,a,b):
    return a*x+b

################### boundary conditions ######################
def calculate_strains(h, load = True):
    '''calculate strains in the box

            Considering periodic boundary conditions (PBC)

            Parameters
            ----------
            h : float, dimension (N, 3)
                Box length for N timesteps in x, y, and z directions
            load : boolean
                either during loading or unloading half of the cycle

            Returns
            -------
            ex, ey, ez : float, dimension (N,)
                Strains in all 3 directions

        '''
    if load:
        ex = h[:,0]/h[0,0]-1.0
        ey = h[:,1]/h[0,1]-1.0
        ez = h[:,2]/h[0,2]-1.0
    else:
        ex = h[:,0]/h[-1,0]-1.0
        ey = h[:,1]/h[-1,1]-1.0
        ez = h[:,2]/h[-1,2]-1.0
    return ex, ey, ez

def pbc(drij, h, hinv=None):
    '''calculate distance vector between i and j
    
        Considering periodic boundary conditions (PBC)
    
        Parameters
        ----------
        drij : float, dimension (npairs, 3)
            distance vectors of atom pairs (Angstrom)
        h : float, dimension (3, 3)
            Periodic box size h = (c1|c2|c3)
        hinv : optional, float, dimension (3, 3)
            inverse matrix of h, if None, it will be calculated

        Returns
        -------
        drij : float, dimension (npairs, 3)
            modified distance vectors of atom pairs considering PBC (Angstrom)
            
    '''
    # Check the input
    if len(drij.shape) == 1:         # Only one pair
        drij = drij.reshape(1, -1)
    if (len(drij.shape) != 2):
        raise ValueError('pbc: drij shape not correct, must be (npairs, nd), (nd = 2,3)')
    npairs, nd = drij.shape 
    if len(h.shape) != 2 or h.shape[0] != h.shape[1] or nd != h.shape[0]:
        raise ValueError('pbc: h matrix shape not consistent with drij')
    # Calculate inverse matrix of h
    if hinv is None:
        hinv = np.linalg.inv(h)

    dsij = np.dot(hinv, drij.T).T
    dsij = dsij - np.round(dsij)
    drij = np.dot(h, dsij.T).T
    
    return drij

def pbc_msd(pos_new, pos_old, h):
    '''calculate the offset vector to add to avoid periodic boundary jump
    
        Considering periodic boundary conditions (PBC)
    
        Parameters
        ----------
        pos_new : float, dimension (npairs, 3)
            distance vectors of atom pairs (Angstrom)
        pos_old : float, dimension (npairs, 3)
            distance vectors of atom pairs (Angstrom)
        h : float, dimension (3, 3)
            Periodic box size h = (c1|c2|c3)

        Returns
        -------
        off_vec : float, dimension (npairs, 3)
            modified distance vectors of atom pairs considering PBC (Angstrom)
            
    '''
    # Check the input
    if len(pos_new.shape) == 1:         # Only one pair
        npairs = 1 
        nd = 3
        off_vec = np.zeros(3)
    else:
        npairs, nd = pos_new.shape
        off_vec = np.zeros((npairs,nd))
    h_f = np.diag(h)
    if npairs == 1:
        diff = pos_new-pos_old
        siff = np.rint(diff/h_f)
        off_vec = -siff*h_f
        #print(diff, siff, off_vec)
        '''
    ind_plus = np.where(diff<-h_f/2.0)[0]
        ind_minus = np.where(diff>h_f/2.0)[0]
        if len(ind_plus)>0:
           off_vec[ind_plus] = h_f[ind_plus]
        if len(ind_minus)>0:
           off_vec[ind_minus] = -h_f[ind_minus]
        '''
    else:
        for i in range(npairs):
            diff = pos_new[i,:]-pos_old[i,:]
            siff = np.rint(diff/h_f)
            off_vec[i,:] = -siff*h_f
            '''
            ind_plus = np.where(diff<-h_f/2.0)[0]
            ind_minus = np.where(diff>h_f/2.0)[0]
            if len(ind_plus)>0:
                off_vec[i,ind_plus] = h_f[ind_plus]
            if len(ind_minus)>0:
                off_vec[i,ind_minus] = -h_f[ind_minus]
            '''
    
    return off_vec
################### neighbor lists ######################

def celllist(r, h, Ns, Ny=None, Nz=None):
    '''Construct cell list in 3D
    
        This function takes the **real coordinates** of atoms `r` and the
        simulation box size `h`. Grouping atoms into Nx x Ny x Nz cells
        
        Parameters
        ----------
        r : float, dimension (nparticles, nd)
            *real* coordinate of atoms
        h : float, dimension (nd, nd)
            Periodic box size h = (c1|c2|c3)
        Ns : tuple, dimension (nd, )
            number of cells in x, y, z direction
        Ny : int
            if not None, represent number of cells in y direction, use with Nx = Ns
        Nz : int
            if not None, represent number of cells in z direction
            
        Returns
        -------
        cell : list, dimension (Nx, Ny, Nz)
            each element cell[i][j][k] is also a list recording all 
            the indices of atoms within the cell[i][j][k].
            (0 <= i < Nx, 0 <= j < Ny, 0 <= k < Nz)
        cellid : int, dimension (nparticles, nd)
            for atom i:
            ix, iy, iz = (cellid[i, 0], cellid[i, 1], cellid[i, 2])
            atom i belongs to cell[ix][iy][iz]

    '''
    if Ny is not None:
        if Nz is None:
            Ns = (Ns, Ny)
        else:
            Ns = (Ns, Ny, Nz)

    nparticle, nd = r.shape
    if nd != 3 or len(Ns) != 3:
        raise TypeError('celllist: only support 3d cell')

    # create empty cell list of size Nx x Ny x Nz
    cell = np.empty(Ns, dtype=object)
    for i, v in np.ndenumerate(cell):
        cell[i] = []

    # find reduced coordinates of all atoms
    s = np.dot(np.linalg.inv(h), r.T).T
    # fold reduced coordinates into [0, 1) as scaled coordinates
    s = s - np.floor(s)

    # create cell list and cell id list
    cellid = np.floor(s*np.array(Ns)[np.newaxis, :]).astype(np.int)
    for i in range(nparticle):
        cell[tuple(cellid[i, :])].append(i)

    return cell, cellid

def verletlist(r, h, rv, atoms = None, near_neigh = None, vectorization = True):
    '''Construct Verlet List (neighbor list) in 3D (vectorized)
    
        Uses celllist to achieve O(N)
    
        Parameters
        ----------
        r : float, dimension (nparticles, 3)
            *real* coordinate of atoms
        h : float, dimension (3, 3)
            Periodic box size h = (c1|c2|c3)
        rv : float
            Verlet cut-off radius
            
        Returns
        -------
        nn : int, dimension (nparticle, )
            nn[i] is the number of neighbors for atom i
        nindex : list, dimension (nparticle, nn)
            nindex[i][j] is the index of j-th neighbor of atom i,
            0 <= j < nn[i].
            
    '''
    nparticles, nd = r.shape
    if nd != 3:
        raise TypeError('celllist: only support 3d cell')
    if atoms is not None:
        nparticles = atoms
    # first determine the size of the cell list
    c1 = h[:, 0]; c2 = h[:, 1]; c3 = h[:, 2];
    V = np.abs(np.linalg.det(h))
    hx = np.abs( V / np.linalg.norm(np.cross(c2, c3)))
    hy = np.abs( V / np.linalg.norm(np.cross(c3, c1)))
    hz = np.abs( V / np.linalg.norm(np.cross(c1, c2)))
    
    # Determine the number of cells in each direction
    Nx = np.floor(hx/rv).astype(np.int)
    Ny = np.floor(hy/rv).astype(np.int)
    Nz = np.floor(hz/rv).astype(np.int)
    if Nx > 50:
        Nx = 12 #(Nx/20).astype(np.int)
    if Ny > 50:
        Ny = 12 #(Ny/20).astype(np.int)
    if Nz > 50:
        Nz = 12 #(Nz/20).astype(np.int)
    
    if Nx < 2 or Ny < 2 or Nz < 2:
        raise ValueError("Number of cells too small! Increase simulation box size.")

    # Inverse of the h matrix
    hinv = np.linalg.inv(h);
    cell, cellid = celllist(r, h, Nx, Ny, Nz)
    
    # initialize Verlet list
    nn = np.zeros(nparticles, dtype=int)
    nindex = [[] for i in range(nparticles)]
    if near_neigh is not None:
        global_nbr = []
    for i in range(nparticles):
        # position of atom i
        ri = r[i, :].reshape(1, 3)
        if near_neigh is not None:
            nbr_inds = []
            nbr_dist = []
        # find which cell (ix, iy, iz) that atom i belongs to
        ix, iy, iz = (cellid[i, 0], cellid[i, 1], cellid[i, 2])
        
        # go through all neighboring cells
        ixr = ix+1
        iyr = iy+1
        izr = iz+1
        if Nx < 3:
            ixr = ix
        if Ny < 3:
            iyr = iy
        if Nz < 3:
            izr = iz
        
        for nx in range(ix-1, ixr+1):
            for ny in range(iy-1, iyr+1):
                for nz in range(iz-1, izr+1):
                    # apply periodic boundary condition on cell id nnx, nny, nnz
                    nnx, nny, nnz = (nx%Nx, ny%Ny, nz%Nz)

                    # extract atom id in this cell
                    ind = cell[nnx][nny][nnz].copy()
                    nc = len(ind)

                    # vectorized implementation
                    if vectorization:
                        if i in ind:
                            ind.remove(i)
                        rj = r[ind,:]
                        drij = pbc(rj - np.repeat(ri, len(ind), axis=0), h, hinv)

                        ind_nbrs = np.where(np.linalg.norm(drij, axis=1) < rv)[0].tolist()
                        if near_neigh is None:
                            if len(ind_nbrs) > 0:
                                nn[i] += len(ind_nbrs)
                                nindex[i].extend([ind[j] for j in ind_nbrs])
                        else:
                            if len(ind_nbrs) > 0:
                                nbr_inds.extend([ind[j] for j in ind_nbrs])
                                nbr_dist.extend(np.linalg.norm(drij[ind_nbrs], axis=1).tolist())
                    else:
                        for k in range(nc):
                            j = ind[k]
                            # update nn[i] and nindex[i]
                            if i == j:
                                continue
                            else:
                                rj = r[j, :].reshape(1, 3)

                                # obtain the distance between atom i and atom j
                                drij = pbc(rj - ri, h, hinv)
                                if near_neigh is None:
                                    if np.linalg.norm(drij) < rv:
                                        nn[i] += 1
                                        nindex[i].append(j)
                                else:
                                    if np.linalg.norm(drij) < rv:
                                        nbr_dist.extend(np.linalg.norm(drij).reshape(-1,))
                                        nbr_inds.append(j)
        if near_neigh is not None:
            if len(nbr_dist)>=near_neigh:
                nbr_dist = np.array(nbr_dist)
                nbr_inds = np.array(nbr_inds)
                nbr_inds = nbr_inds[np.argsort(nbr_dist)].tolist()
                to_add = []
                ct = 0
                for ll in range(len(nbr_inds)):
                    if (nbr_inds[ll] not in global_nbr) and ct < near_neigh:
                        to_add.append(nbr_inds[ll])
                        global_nbr.append(nbr_inds[ll])
                        ct += 1
                if ct == near_neigh:
                    nn[i] = near_neigh
                    nindex[i].append(to_add)

    return nn, nindex

def verletlist_binary(r, rr, h, rv, vectorization = True, near_neigh = None):
    '''Construct Verlet List (neighbor list) in 3D (vectorized)
    
        Uses celllist to achieve O(N)
    
        Parameters
        ----------
        r : float, dimension (nparticles, 3)
            *real* coordinate of atoms
        h : float, dimension (3, 3)
            Periodic box size h = (c1|c2|c3)
        rv : float
            Verlet cut-off radius
            
        Returns
        -------
        nn : int, dimension (nparticle, )
            nn[i] is the number of neighbors for atom i
        nindex : list, dimension (nparticle, nn)
            nindex[i][j] is the index of j-th neighbor of atom i,
            0 <= j < nn[i].
            
    '''
    np_a, nd = r.shape
    np_b, nd = rr.shape
    if nd != 3:
        raise TypeError('celllist: only support 3d cell')

    # first determine the size of the cell list
    c1 = h[:, 0]; c2 = h[:, 1]; c3 = h[:, 2];
    V = np.abs(np.linalg.det(h))
    hx = np.abs( V / np.linalg.norm(np.cross(c2, c3)))
    hy = np.abs( V / np.linalg.norm(np.cross(c3, c1)))
    hz = np.abs( V / np.linalg.norm(np.cross(c1, c2)))
    
    # Determine the number of cells in each direction
    Nx = np.floor(hx/rv).astype(np.int) 
    Ny = np.floor(hy/rv).astype(np.int)
    Nz = np.floor(hz/rv).astype(np.int)
    if Nx > 100:
        Nx = (Nx/20).astype(np.int)
    if Ny > 100:
        Ny = (Ny/20).astype(np.int)
    if Nz > 100:
        Nz = (Nz/20).astype(np.int)
    if Nx < 2 or Ny < 2 or Nz < 2:
        raise ValueError("Number of cells too small! Increase simulation box size.")
    
    # Inverse of the h matrix
    hinv = np.linalg.inv(h);
    cell_a, cellid_a = celllist(r, h, Nx, Ny, Nz)
    cell_b, cellid_b = celllist(rr, h, Nx, Ny, Nz)
    
    # Find the number of atoms
    np_a = r.shape[0]
    np_b = rr.shape[0]
    
    # initialize Verlet list
    nn_a = np.zeros(np_a, dtype=int)
    nindex_a = [[] for i in range(np_a)]
    nn_b = np.zeros(np_b, dtype=int)
    nindex_b = [[] for i in range(np_b)]
    if near_neigh is not None:
        global_nbr = []
    for i in range(np_a):
        # position of atom i
        ri = r[i, :].reshape(1, 3)
        if near_neigh is not None:
            nbr_inds = []
            nbr_dist = []
        # find which cell (ix, iy, iz) that atom i belongs to
        ix, iy, iz = (cellid_a[i, 0], cellid_a[i, 1], cellid_a[i, 2])
        
        # go through all neighboring cells
        ixr = ix+1
        iyr = iy+1
        izr = iz+1
        if Nx < 3:
            ixr = ix
        if Ny < 3:
            iyr = iy
        if Nz < 3:
            izr = iz
        
        for nx in range(ix-1, ixr+1):
            for ny in range(iy-1, iyr+1):
                for nz in range(iz-1, izr+1):
                    # apply periodic boundary condition on cell id nnx, nny, nnz
                    nnx, nny, nnz = (nx%Nx, ny%Ny, nz%Nz)

                    # extract atom id in this cell
                    ind = cell_b[nnx][nny][nnz].copy()
                    nc = len(ind)
                    if vectorization:			
                    # vectorized implementation
                        if i in ind:
                            ind.remove(i)
                        rj = rr[ind,:]
                        drij = pbc(rj - np.repeat(ri, len(ind), axis=0), h, hinv)
                        ind_nbrs = np.where(np.linalg.norm(drij, axis=1) < rv)[0].tolist()
                        if near_neigh is None:
                            if len(ind_nbrs) > 0:
                                nn_a[i] += len(ind_nbrs)
                                nindex_a[i].extend([ind[j] for j in ind_nbrs])
                        else:
                            if len(ind_nbrs) > 0:
                                nbr_inds.extend([ind[j] for j in ind_nbrs])
                                nbr_dist.extend(np.linalg.norm(drij[ind_nbrs],axis=1).tolist()) 
 
                    else:
                        for k in range(nc):
                            j = ind[k]
                            # update nn[i] and nindex[i]
                            if i == j:
                                continue
                            else:
                                rj = rr[j, :].reshape(1, 3)

                            # obtain the distance between atom i and atom j
                                drij = pbc(rj - ri, h, hinv)
                                if near_neigh is None:
                                    if np.linalg.norm(drij) < rv:
                                        nn_a[i] += 1
                                        nindex_a[i].append(j)
                                else:
                                    if np.linalg.norm(drij) < rv:
                                        nbr_dist.extend(np.linalg.norm(drij).reshape(-1,))
                                        nbr_inds.append(j)
        
        if near_neigh is not None:
            if len(nbr_dist)>=near_neigh:
                nbr_dist = np.array(nbr_dist)
                nbr_inds = np.array(nbr_inds)
                nbr_inds = nbr_inds[np.argsort(nbr_dist)].tolist()
                to_add = []
                ct = 0 
                for ll in range(len(nbr_inds)):
                    if (nbr_inds[ll] not in global_nbr) and ct < near_neigh:
                        to_add.append(nbr_inds[ll])
                        global_nbr.append(nbr_inds[ll])
                        ct += 1
                if ct == near_neigh:
                    nn_a[i] = near_neigh
                    nindex_a[i].append(to_add)

    for i in range(np_b):
        # position of atom i
        ri = rr[i, :].reshape(1, 3)
        
        # find which cell (ix, iy, iz) that atom i belongs to
        ix, iy, iz = (cellid_b[i, 0], cellid_b[i, 1], cellid_b[i, 2])
        
        # go through all neighboring cells
        ixr = ix+1
        iyr = iy+1
        izr = iz+1
        if Nx < 3:
            ixr = ix
        if Ny < 3:
            iyr = iy
        if Nz < 3:
            izr = iz
        
        for nx in range(ix-1, ixr+1):
            for ny in range(iy-1, iyr+1):
                for nz in range(iz-1, izr+1):
                    # apply periodic boundary condition on cell id nnx, nny, nnz
                    nnx, nny, nnz = (nx%Nx, ny%Ny, nz%Nz)

                    # extract atom id in this cell
                    ind = cell_a[nnx][nny][nnz].copy()
                    nc = len(ind)
                    if vectorization:
                    # vectorized implementation
                        if i in ind:
                            ind.remove(i)
                        rj = r[ind,:]
                        drij = pbc(rj - np.repeat(ri, len(ind), axis=0), h, hinv)
  
                        ind_nbrs = np.where(np.linalg.norm(drij, axis=1) < rv)[0].tolist()
                        if len(ind_nbrs) > 0:
                            nn_b[i] += len(ind_nbrs)
                            nindex_b[i].extend([ind[j] for j in ind_nbrs])
                    else:
                        for k in range(nc):
                            j = ind[k]
                            # update nn[i] and nindex[i]
                            if i == j:
                                continue
                            else:
                                rj = r[j, :].reshape(1, 3)

                            # obtain the distance between atom i and atom j
                                drij = pbc(rj - ri, h, hinv)

                                if np.linalg.norm(drij) < rv:
                                    nn_b[i] += 1
                                    nindex_b[i].append(j)

    return nn_a, nindex_a, nn_b, nindex_b

def verletlist_old(r, h, rv):
    '''Construct Verlet List (neighbor list) in 3D
    
        Uses celllist to achieve O(N)
    
        Parameters
        ----------
        r : float, dimension (nparticles, 3)
            *real* coordinate of atoms
        h : float, dimension (3, 3)
            Periodic box size h = (c1|c2|c3)
        rv : float
            Verlet cut-off radius
            
        Returns
        -------
        nn : int, dimension (nparticle, )
            nn[i] is the number of neighbors for atom i
        nindex : list, dimension (nparticle, nn)
            nindex[i][j] is the index of j-th neighbor of atom i,
            0 <= j < nn[i].
            
    '''
    nparticle, nd = r.shape
    if nd != 3:
        raise TypeError('celllist: only support 3d cell')

    # first determine the size of the cell list
    c1 = h[:, 0]; c2 = h[:, 1]; c3 = h[:, 2];
    V = np.abs(np.linalg.det(h))
    hx = np.abs( V / np.linalg.norm(np.cross(c2, c3)))
    hy = np.abs( V / np.linalg.norm(np.cross(c3, c1)))
    hz = np.abs( V / np.linalg.norm(np.cross(c1, c2)))
    
    # Determine the number of cells in each direction
    Nx = np.floor(hx/rv).astype(np.int)
    Ny = np.floor(hy/rv).astype(np.int)
    Nz = np.floor(hz/rv).astype(np.int)
    if Nx > 100:
        Nx = (Nx/20).astype(np.int)
    if Ny > 100:
        Ny = (Ny/20).astype(np.int)
    if Nz > 100:
        Nz = (Nz/20).astype(np.int)
    
    if Nx < 2 or Ny < 2 or Nz < 2:
        raise ValueError("Number of cells too small! Increase simulation box size.")

    # Inverse of the h matrix
    hinv = np.linalg.inv(h);
    cell, cellid = celllist(r, h, Nx, Ny, Nz)
    
    # Find the number of atoms
    nparticles = r.shape[0]
    
    # initialize Verlet list
    nn = np.zeros(nparticles, dtype=int)
    nindex = [[] for i in range(nparticles)]
    
    for i in range(nparticles):
        # position of atom i
        ri = r[i, :].reshape(1, 3)
        
        # find which cell (ix, iy, iz) that atom i belongs to
        ix, iy, iz = (cellid[i, 0], cellid[i, 1], cellid[i, 2])
        
        # go through all neighboring cells
        ixr = ix+1
        iyr = iy+1
        izr = iz+1
        if Nx < 3:
            ixr = ix
        if Ny < 3:
            iyr = iy
        if Nz < 3:
            izr = iz
        
        for nx in range(ix-1, ixr+1):
            for ny in range(iy-1, iyr+1):
                for nz in range(iz-1, izr+1):
                    # apply periodic boundary condition on cell id nnx, nny, nnz
                    nnx, nny, nnz = (nx%Nx, ny%Ny, nz%Nz)

                    # extract atom id in this cell
                    ind = cell[nnx][nny][nnz]
                    nc = len(ind)

                    # go through all the atoms in the neighboring cells
                    for k in range(nc):
                        j = ind[k]
                        # update nn[i] and nindex[i]
                        if i == j:
                            continue
                        else:
                            rj = r[j, :].reshape(1, 3)

                            # obtain the distance between atom i and atom j
                            drij = pbc(rj - ri, h, hinv)

                            if np.linalg.norm(drij) < rv:
                                nn[i] += 1
                                nindex[i].append(j)

    return nn, nindex

def convert_nindex_to_array(nn, nindex):
    '''convert the indexes of neighbors to an array

            Parameters
            ----------
            nn : int, dimension (natoms)
                List of number of nearest neighbors per atom
            nindex : int, dimension (natoms)
                List of index of neighbors

            Returns
            -------
            index_array : float, dimension (natoms, nn.max())
                nindex converted to an array

    '''
    nparticles = nn.shape[0]
    index_array = np.ones([nparticles, nn.max().astype(int)], dtype=int)*(-1)
    for i in range(nparticles):
        index_array[i,:nn[i].astype(int)] = nindex[i][:nn[i].astype(int)]
    return index_array

def pbc1d(r,h):
    '''calculate distance vector between i and j

            Considering periodic boundary conditions (PBC)

            Parameters
            ----------
            r : float, dimension (1, 3)
                distance vectors of atom pairs (Angstrom)
            h : float, dimension (3, 3)
                Periodic box size h = (c1|c2|c3)

            Returns
            -------
            r_ : float, dimension (npairs, 3)
                modified distance vectors of atom pairs considering PBC (Angstrom)

    '''
    s=r/h
    s=s-np.round(s)
    return s*h

def config_repeater(pos_a,h,size = 2):
    '''Repeat the configuration in case g(r) cutoff is too big for the current box

            Parameters
            ----------
            pos_a : float, dimension (npairs, 3)
                distance vectors of atom pairs (Angstrom)
            h : float, dimension (3, 3)
                Periodic box size h = (c1|c2|c3)
            size : int
                Number of times to repeat the vox in each direction

            Returns
            -------
            new_pos : float, dimension (npairs*size**3, 3)
                modified positions of atom pairs after box repetition
            new_h : float, dimension (3, 3)
                modified Periodic box size h = (c1|c2|c3)

    '''
    pos = pos_a.copy()
    boxl = np.mean(np.diag(h))
    new_h = h*size
    new_pos = np.zeros((pos.shape[0]*size**3,pos.shape[1]))
    atoms = pos.shape[0]
    ct=0
    for i in range(size):
        for j in range(size):
            for k in range(size):
                new_pos[ct*atoms:(ct+1)*atoms,:]=pos[:,:]+boxl*np.repeat(np.array([[i,j,k]]),atoms,axis=0)
                ct+=1
    return new_pos, new_h

def unwrap_trajectories(u_pos, pos, h):
    atoms = u_pos.shape[0]
    step = u_pos.shape[2]
    off_vec = np.zeros((atoms, 3))
    for i in range(step - 1):
        pos_new = pos[(i + 1) * atoms:(i + 2) * atoms].copy()
        if i == 0:
            pos_old = pos[(i) * atoms:(i + 1) * atoms].copy()
        new_off = pbc_msd(pos_new, pos_old, h)
        off_vec += new_off
        pos_old = pos_new
        u_pos[:, :, i + 1] = pos_new + off_vec
    return u_pos.copy()


