# --------------------------------------------------------
# load_util.py
# by Shaswat Mohanty, shaswatm@stanford.edu
#
# Objectives
# Data manipulation from LAMMPS dump/input files
#
# Cite: (https://doi.org/10.1088/1361-651X/ac860c)
# --------------------------------------------------------
import os
import numpy as np
import re


######################## io functions ########################
def load_atom_data(filename, skip_to_np, skip_to_box, skip_to_pos, verbose=True, style='full', h_full=True):
    '''Load LAMMPS .data type configuration files

        Parameters
        ----------
        filename : string
            Filename string, should be of '.data' type
        verbose : bool
            If True, print data information

        Returns
        -------
        pos : ndarray
            Real coordinates of atoms
        h : ndarray
            Simulation box size (c1|c2|c3)

    '''
    if not os.path.exists(filename):
        raise TypeError('File {} not exist!'.format(filename))

    if verbose:
        print('Reading LAMMPS data file', filename)

    nparticles = np.genfromtxt(filename, skip_header=skip_to_np, dtype=np.int, max_rows=1).item(0)
    box = np.genfromtxt(filename, skip_header=skip_to_box, dtype=np.double, max_rows=3)[:, :2]
    if h_full:
        h = np.diag(box[:, 1] - box[:, 0])
    else:
        h = box
    rawdata = np.genfromtxt(filename, skip_header=skip_to_pos, dtype=np.double, max_rows=nparticles)
    rawdata = rawdata[np.argsort(rawdata[:, 0])]
    pos = np.zeros([nparticles, 3])
    if style == 'full':
        pos[rawdata[:, 0].astype(int) - 1] = rawdata[:, 4:7]
    elif style == 'bond':
        pos[rawdata[:, 0].astype(int) - 1] = rawdata[:, 3:6]

    if verbose:
        print('Nparticles = %d' % (nparticles))
        print('Simulation Cell')
        print(h)
        print('Atomic positions: ')
        print(pos)
    return pos, h

def load_atom_data_binary(filename, skip_to_np, skip_to_box, skip_to_pos, verbose=True, style='full', family=None,
                          h_full=True, fill=0, atom_ids=False):
    '''Load LAMMPS .data type configuration files

        Parameters
        ----------
        filename : string
            Filename string, should be of '.data' type
        verbose : bool
            If True, print data information

        Returns
        -------
        pos : ndarray
            Real coordinates of atoms
        h : ndarray
            Simulation box size (c1|c2|c3)

    '''
    if not os.path.exists(filename):
        raise TypeError('File {} not exist!'.format(filename))

    if verbose:
        print('Reading LAMMPS data file', filename)

    nparticles = np.genfromtxt(filename, skip_header=skip_to_np, dtype=np.int, max_rows=1).item(0)
    box = np.genfromtxt(filename, skip_header=skip_to_box, dtype=np.double, max_rows=3)[:, :2]
    if h_full:
        h = np.diag(box[:, 1] - box[:, 0])
    else:
        h = box
    rawdata = np.genfromtxt(filename, skip_header=skip_to_pos, dtype=np.double, max_rows=nparticles)
    rawdata = rawdata[np.argsort(rawdata[:, 0])]
    o_ct = 0
    if family == 'polymer':
        o_ct = 1 + fill
    if style == 'sphere':
        a_col = 1
    else:
        a_col = 2
    ind_a = np.where(abs(rawdata[:, a_col] - (1 + o_ct)) < 1e-3)
    ind_b = np.where(abs(rawdata[:, a_col] - (2 + o_ct)) < 1e-3)
    atoms_a = rawdata[ind_a, 0].shape[1]
    atoms_b = rawdata[ind_b, 0].shape[1]
    pos_a = np.zeros((atoms_a, 3))
    pos_b = np.zeros((atoms_b, 3))
    if style == 'full':
        pos_a[:, :] = rawdata[ind_a, 4:7]
        pos_b[:, :] = rawdata[ind_b, 4:7]
    elif style == 'bond':
        pos_a[:, :] = rawdata[ind_a, 3:6]
        pos_b[:, :] = rawdata[ind_b, 3:6]
    if atom_ids:
        ind_tot = []
        ind_tot.append(ind_b)
        ind_tot.append(ind_a)
        id_no = rawdata[ind_tot, 0]
    if verbose:
        print('Nparticles = %d' % (nparticles))
        print('Simulation Cell')
        print(h)
        print('Atomic positions: ')
        print(pos)
    if atom_ids:
        return pos_a, pos_b, h, atoms_a, atoms_b, id_no
    else:
        return pos_a, pos_b, h, atoms_a, atoms_b

def load_dumpfile_atom_data(filename, total_steps, dump_frequency, verbose=True, at_type=1, h_full=True,
                            add='position'):
    '''Load LAMMPS .data type configuration files
        stored using "dump id all custom dump_filename id type x y z"
        Parameters
        ----------
        filename : string
            Filename string, should be of '.data' type
        verbose : bool
            If True, print data information

        Returns
        -------
        pos : ndarray
            Real coordinates of atoms
        h : ndarray
            Simulation box size (c1|c2|c3)

    '''
    if not os.path.exists(filename):
        raise TypeError('File {} not exist!'.format(filename))

    if verbose:
        print('Reading LAMMPS dumpfile file', filename)

    skip_to_np = 3
    fid = open(filename, 'r')
    lines = fid.readlines()
    npar = np.array(re.findall("[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", lines[skip_to_np].strip()), dtype=int)
    nparticles = npar[0]
    ndum = int(total_steps / dump_frequency)
    if h_full:
        h = np.zeros([ndum, 3])
    else:
        h = np.zeros([3, 2, ndum])
    pos = np.zeros([ndum * nparticles, 3])
    if at_type == 1:
        ct = 1
    elif at_type == 2:
        ct = 0
    for i in range(ndum):
        skip_to_pos = (i + ct) * nparticles + (i + ct + 1) * 9
        skip_to_box = (i + ct) * nparticles + (i + ct + 1) * 9 - 4
        if h_full:
            h[i, :] = get_h_from_lines(lines, skip_to_box)
        else:
            h[:, :, i] = get_h_from_lines(lines, skip_to_box, h_full=False)
        rawdata = get_pos_from_lines(lines, skip_to_pos, nparticles, add=add)
        rawdata = rawdata[np.argsort(rawdata[:, 0])]
        pos[i * nparticles:(i + 1) * nparticles] = rawdata[:, 2:5]
        if h_full:
            pos[i * nparticles:(i + 1) * nparticles] = np.dot(pos[i * nparticles:(i + 1) * nparticles],
                                                              np.diag(h[i, :]))
        else:
            hin = h[:, 1, i] - h[:, 0, i]
            pos[i * nparticles:(i + 1) * nparticles] = np.dot(pos[i * nparticles:(i + 1) * nparticles], np.diag(hin))
    if verbose:
        print('Nparticles = %d' % (nparticles))
        print('Simulation Cell')
        print(h)
        print('Atomic positions: ')
        print(pos)
    #    pos=np.dot(pos,h)
    return pos, h, nparticles

def load_dumpfile_atom_data_fast(filename, total_steps, dump_frequency, verbose=True, h_full=True, at_type=1,
                                 add='position'):
    '''Load LAMMPS .data type configuration files
        stored using "dump id all custom dump_filename id type x y z"
        Parameters
        ----------
        filename : string
            Filename string, should be of '.data' type
        verbose : bool
            If True, print data information

        Returns
        -------
        pos : ndarray
            Real coordinates of atoms
        h : ndarray
            Simulation box size (c1|c2|c3)

    '''
    if not os.path.exists(filename):
        raise TypeError('File {} not exist!'.format(filename))

    if verbose:
        print('Reading LAMMPS dumpfile file', filename)

    skip_to_np = 3
    fid = open(filename, 'r')
    lines = fid.readlines()
    npar = np.array(re.findall("[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", lines[skip_to_np].strip()), dtype=int)
    nparticles = npar[0]
    ndum = int(total_steps / dump_frequency)
    if h_full:
        h = np.zeros([ndum, 3])
    else:
        h = np.zeros([3, 2, ndum])
    pos = np.zeros([int(ndum * nparticles), 3])
    if at_type == 1:
        ct = 1
    elif at_type == 2:
        ct = 0
    for i in range(ndum):
        skip_to_pos = (i + ct) * nparticles + (i + ct + 1) * 9
        skip_to_box = (i + ct) * nparticles + (i + ct + 1) * 9 - 4
        if h_full:
            h[i, :] = get_h_from_lines(lines, skip_to_box)
        else:
            h[:, :, i] = get_h_from_lines(lines, skip_to_box, h_full=False)
        rawdata = get_pos_from_lines(lines, skip_to_pos, nparticles, add=add)
        rawdata = rawdata[np.argsort(rawdata[:, 0])]
        pos[i * nparticles:(i + 1) * nparticles] = rawdata[:, 2:5]
        if h_full:
            pos[i * nparticles:(i + 1) * nparticles] = np.dot(pos[i * nparticles:(i + 1) * nparticles],
                                                              np.diag(h[i, :]))
        else:
            hin = h[:, 1, i] - h[:, 0, i]
            pos[i * nparticles:(i + 1) * nparticles] = np.dot(pos[i * nparticles:(i + 1) * nparticles], np.diag(hin))

    if verbose:
        print('Nparticles = %d' % (nparticles))
        print('Simulation Cell')
        print(h)
        print('Atomic positions: ')
        print(pos)
    #    pos=np.dot(pos,h)
    return pos, h, nparticles

def load_dumpfile_velocity(filename, total_steps, dump_frequency, verbose=True, at_type=1):
    '''Load LAMMPS .data type configuration files
        stored using "dump id all custom dump_filename id type x y z vx vy vz"
        Parameters
        ----------
        filename : string
            Filename string, should be of '.data' type
        verbose : bool
            If True, print data information

        Returns
        -------
        pos : ndarray
            Real coordinates of atoms
        h : ndarray
            Simulation box size (c1|c2|c3)

    '''
    if not os.path.exists(filename):
        raise TypeError('File {} not exist!'.format(filename))

    if verbose:
        print('Reading LAMMPS dumpfile file', filename)

    skip_to_np = 3
    fid = open(filename, 'r')
    lines = fid.readlines()
    npar = np.array(re.findall("[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", lines[skip_to_np].strip()), dtype=int)
    nparticles = npar[0]
    ndum = int(total_steps / dump_frequency)
    h = np.zeros([ndum, 3])
    pos = np.zeros([int(ndum * nparticles), 3])
    if at_type == 1:
        ct = 1
    elif at_type == 2:
        ct = 0
    for i in range(ndum):
        skip_to_pos = (i + ct) * nparticles + (i + ct + 1) * 9
        skip_to_box = (i + ct) * nparticles + (i + ct + 1) * 9 - 4
        h[i, :] = get_h_from_lines(lines, skip_to_box)
        rawdata = get_pos_from_lines(lines, skip_to_pos, nparticles, add='velocity')
        rawdata = rawdata[np.argsort(rawdata[:, 0])]
        pos[i * nparticles:(i + 1) * nparticles] = rawdata[:, 5:]
    #        pos[i*nparticles:(i+1)*nparticles] = np.dot(pos[i*nparticles:(i+1)*nparticles],np.diag(h[i,:]))
    if verbose:
        print('Nparticles = %d' % (nparticles))
        print('Simulation Cell')
        print(h)
        print('Atomic positions: ')
        print(pos)
    #    pos=np.dot(pos,h)
    return pos, h, nparticles

def load_dumpfile_atom_data_binary(filename, total_steps, dump_frequency, verbose=True, at_type=1, family=None, fill=0):
    '''Load LAMMPS .data type configuration files
        stored using "dump id all custom dump_filename id type x y z"
        Parameters
        ----------
        filename : string
            Filename string, should be of '.data' type
        verbose : bool
            If True, print data information

        Returns
        -------
        pos : ndarray
            Real coordinates of atoms
        h : ndarray
            Simulation box size (c1|c2|c3)

    '''
    if not os.path.exists(filename):
        raise TypeError('File {} not exist!'.format(filename))

    if verbose:
        print('Reading LAMMPS dumpfile file', filename)

    skip_to_np = 3
    nparticles = np.genfromtxt(filename, skip_header=skip_to_np, dtype=np.int, max_rows=1).item(0)
    ndum = int(total_steps / dump_frequency)
    h = np.zeros([ndum, 3])
    pos = np.zeros([ndum * nparticles, 3])
    pos_n = np.zeros([ndum * nparticles, 5])
    o_ct = 0
    if family == 'polymer':
        o_ct = 1 + fill

    if at_type == 1:
        ct = 1
    elif at_type == 2:
        ct = 0
    for i in range(ndum):
        skip_to_pos = (i + ct) * nparticles + (i + ct + 1) * 9
        skip_to_box = (i + ct) * nparticles + (i + ct + 1) * 9 - 4
        box = np.genfromtxt(filename, skip_header=skip_to_box, dtype=np.double, max_rows=3)[:, :2]
        h[i, :] = np.transpose(box[:, 1] - box[:, 0])
        rawdata = np.genfromtxt(filename, skip_header=skip_to_pos, dtype=np.double, max_rows=nparticles)
        rawdata = rawdata[np.argsort(rawdata[:, 0])]
        pos[i * nparticles:(i + 1) * nparticles] = rawdata[:, 2:5]
        pos_n[i * nparticles:(i + 1) * nparticles] = rawdata[:, :5]
        pos[i * nparticles:(i + 1) * nparticles] = np.dot(pos[i * nparticles:(i + 1) * nparticles], np.diag(h[i, :]))
    ind_a = np.where(abs(pos_n[:, 1] - (o_ct + 1)) < 1e-3)
    ind_b = np.where(abs(pos_n[:, 1] - (o_ct + 2)) < 1e-3)
    atoms_a = int(pos[ind_a, 0].shape[1] / ndum)
    atoms_b = int(pos[ind_b, 0].shape[1] / ndum)
    pos_a = pos[ind_a, :]
    pos_b = pos[ind_b, :]
    pos_a = pos_a.reshape(-1, 3)
    pos_b = pos_b.reshape(-1, 3)
    if verbose:
        print('Nparticles = %d' % (nparticles))
        print('Simulation Cell')
        print(h)
        print('Atomic positions: ')
        print(pos)
    #    pos=np.dot(pos,h)
    return pos_a, pos_b, h, atoms_a, atoms_b

def load_dumpfile_atom_data_binary_fast(filename, total_steps, dump_frequency, h_full=True, verbose=True, at_type=1,
                                        add='position', family=None, fill=0):
    '''Load LAMMPS .data type configuration files
        stored using "dump id all custom dump_filename id type x y z"
        Parameters
        ----------
        filename : string
            Filename string, should be of '.data' type
        verbose : bool
            If True, print data information

        Returns
        -------
        pos : ndarray
            Real coordinates of atoms
        h : ndarray
            Simulation box size (c1|c2|c3)

    '''
    if not os.path.exists(filename):
        raise TypeError('File {} not exist!'.format(filename))

    if verbose:
        print('Reading LAMMPS dumpfile file', filename)

    skip_to_np = 3
    fid = open(filename, 'r')
    lines = fid.readlines()
    npar = np.array(re.findall("[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", lines[skip_to_np].strip()), dtype=int)
    nparticles = npar[0]
    ndum = int(total_steps / dump_frequency)
    if h_full:
        h = np.zeros([ndum, 3])
    else:
        h = np.zeros([3, 2, ndum])
    pos = np.zeros([ndum * nparticles, 3])
    pos_n = np.zeros([ndum * nparticles, 5])
    o_ct = 0
    if family == 'polymer':
        o_ct = 1 + fill

    if at_type == 1:
        ct = 1
    elif at_type == 2:
        ct = 0
    for i in range(ndum):
        skip_to_pos = (i + ct) * nparticles + (i + ct + 1) * 9
        skip_to_box = (i + ct) * nparticles + (i + ct + 1) * 9 - 4
        if h_full:
            h[i, :] = get_h_from_lines(lines, skip_to_box)
        else:
            h[:, :, i] = get_h_from_lines(lines, skip_to_box, h_full=False)
        rawdata = get_pos_from_lines(lines, skip_to_pos, nparticles, add=add)
        rawdata = rawdata[np.argsort(rawdata[:, 0])]
        pos[i * nparticles:(i + 1) * nparticles, :] = rawdata[:, 2:5]
        pos_n[i * nparticles:(i + 1) * nparticles, :] = rawdata[:, :5]
        if h_full:
            pos[i * nparticles:(i + 1) * nparticles, :] = np.dot(pos[i * nparticles:(i + 1) * nparticles, :],
                                                                 np.diag(h[i, :]))
        else:
            hin = h[:, 1, i] - h[:, 0, i]
            pos[i * nparticles:(i + 1) * nparticles, :] = np.dot(pos[i * nparticles:(i + 1) * nparticles, :],
                                                                 np.diag(hin))
    ind_a = np.where(abs(pos_n[:, 1] - (1 + o_ct)) < 1e-3)
    ind_b = np.where(abs(pos_n[:, 1] - (2 + o_ct)) < 1e-3)
    atoms_a = int(pos[ind_a, 0].shape[1] / ndum)
    atoms_b = int(pos[ind_b, 0].shape[1] / ndum)
    pos_a = pos[ind_a, :]
    pos_b = pos[ind_b, :]
    pos_a = pos_a.reshape(-1, 3)
    pos_b = pos_b.reshape(-1, 3)
    if verbose:
        print('Nparticles = %d' % (nparticles))
        print('Simulation Cell')
        print(h)
        print('Atomic positions: ')
        print(pos)
    #    pos=np.dot(pos,h)
    return pos_a, pos_b, h, atoms_a, atoms_b

def get_h_from_lines(lines, start, h_full=True):
    '''Get box dimensions from lines read from dumpfile

                Parameters
                ----------
                lines : List
                    List of all line
                h_full : boolean
                    True if all box lengths are required as a vector; False if hi and lo values of box sides are needed
                start : int
                    Line number to start from

                Returns
                -------
                h : float, dimension (1, 3) if True else (3,2)
                    Periodic box size h = (c1|c2|c3)

    '''
    if h_full:
        h = np.zeros([1, 3])
        for i in range(3):
            a = np.array(re.findall("[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", lines[start + i].strip()),
                         dtype=float)
            h[0, i] = a[1] - a[0]
    else:
        h = np.zeros([3, 2])
        for i in range(3):
            a = np.array(re.findall("[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", lines[start + i].strip()),
                         dtype=float)
            h[i, :] = np.array([a[0], a[1]])
    return h

def get_pos_from_lines(lines, start, atoms, add='position'):
    '''Get box dimensions from lines read from dumpfile

                    Parameters
                    ----------
                    lines : List
                        List of all line
                    atoms : int
                        Number of atoms per frame
                    start : int
                        Line number to start from
                    add: str,
                        If dumpfile has just positions or positions and velocities

                    Returns
                    -------
                    pos : float, dimension (atoms, 3)
                        Position of all atoms

    '''
    if add == 'position':
        raw = np.zeros([atoms, 5])
    else:
        raw = np.zeros([atoms, 8])
    for i in range(atoms):
        raw[i, :] = np.array(re.findall("[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", lines[start + i].strip()),
                             dtype=float)

    return raw

def write_lammps_dump(posit, save_file = 'sample', h_in = None, total_atoms = 200):
    '''Write a lammps dumpfile from a position array

                    Parameters
                    ----------
                    posit : float, dimension (natoms, 3, steps)
                        Atomic positions over all frames
                    save_file : str,
                        dump_filename to save to
                    h_in : float
                        Assumin cubic box -- just one side of the box


    '''
    if h_in is None:
        boxl = 17.0
    else:
        boxl = h_in
    if len(posit.shape) == 3:
        total_steps = posit.shape[2]
        N_atoms = posit.shape[0]
        for i in range(total_steps):
            s = posit[:,:,i] / boxl
            s = s - 0.5
            s = s - np.round(s)
            posit[:,:,i] = (s + 0.5)
        fid = open('dump.%s'%save_file,'w')
        for i in range(total_steps):
            fid.write('ITEM: TIMESTEP\n')
            fid.write('%d\n'%(i))
            fid.write('ITEM: NUMBER OF ATOMS\n')
            fid.write('%d\n'%(N_atoms))
            fid.write('ITEM: BOX BOUNDS pp pp pp\n')
            fid.write('0 %.8f\n'%boxl)
            fid.write('0 %.8f\n'%boxl)
            fid.write('0 %.8f\n'%boxl)
            fid.write('ITEM: ATOMS id type xs ys zs\n')
            for j in range(N_atoms):
                fid.write('%d 1 %.5f %.5f %.5f\n'%(j+1,posit[j,0,i],posit[j,1,i],posit[j,2,i]))
        fid.close()
    else:
        N_atoms = total_atoms
        total_steps = posit.shape[0]//N_atoms
        s = posit / boxl
        s = s - 0.5
        s = s - np.round(s)
        posit = (s + 0.5)
        fid = open('dump.%s'%save_file,'w')
        for i in range(total_steps):
            fid.write('ITEM: TIMESTEP\n')
            fid.write('%d\n'%(i))
            fid.write('ITEM: NUMBER OF ATOMS\n')
            fid.write('%d\n'%(N_atoms))
            fid.write('ITEM: BOX BOUNDS pp pp pp\n')
            fid.write('0 %.8f\n'%boxl)
            fid.write('0 %.8f\n'%boxl)
            fid.write('0 %.8f\n'%boxl)
            fid.write('ITEM: ATOMS id type xs ys zs\n')
            for j in range(N_atoms):
                fid.write('%d 1 %.5f %.5f %.5f\n'%(j+1,posit[i*N_atoms+j,0],posit[i*N_atoms+j,1],posit[i*N_atoms+j,2]))
        fid.close()
