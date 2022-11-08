# --------------------------------------------------------
# post_util.py
# by Shaswat Mohanty, shaswatm@stanford.edu
#
# Objectives
# Contains classes for all analysis by inheriting the other utility libraries
# test_util: Comparing first principle calculations with density field calculations
# XPCS_Suite: Wrapper around intensity_util.py for XPCS and XSVS analysis
# test_analysis: Class for developer testing -- not intended for users
# post_analysis: Wrapper for obtaining results from the MSMSE paper (https://doi.org/10.1088/1361-651X/ac860c)
#
# Cite: (https://doi.org/10.1088/1361-651X/ac860c)
# --------------------------------------------------------
import numpy as np

from intensity_util import *
from copy import deepcopy

class bcolors:
    '''
    Colors for test case indication
    '''
    RED = '\033[31m'
    GRN = '\033[32m'
    YEL = '\033[33m'
    BLU = '\033[34m'
    MAG = '\033[35m'
    CYN = '\033[36m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    UNDERLINE = '\033[4m'

class test_util():
    def __init__(self,box_length = 17.44024100780622, rc = 4.0, filename = '../datafiles/dump.XPCS',smear = True, position = 'test', N_grid = 400, mini_grid = 40, sigma_grid = 100, x_off = 0, y_off = 14, z_off = -11, fourier_smear = 0.25, offset = 0.0, comment = 'Default case', offset_vector = np.array([[0,0,1]]), default_s_q =0.669314655271, q_val = 6.28, q_tol = 0.015, total_steps = 40, timestep = 0.005, I_Q = False, S_Q = False, ISF = False, smeared_integral = False, direct_point = False, atoms_add = None, frames =40 ):
        if (int(I_Q) + int(S_Q) + int(ISF)) != 1:
            raise TypeError('Choose one and only one computation item')
        if (int(smeared_integral) + int(direct_point)) != 1:
            raise TypeError('Choose between direct computation and smeared integral')
        self.box_length = box_length
        self.smear = smear
        self.position = position
        self.N_grid = N_grid
        self.mini_grid = mini_grid
        self.sigma_grid = sigma_grid
        self.density_cutoff = int(np.floor(5.0*self.N_grid/self.sigma_grid))
        self.fourier_smear = fourier_smear
        if rc < 4/fourier_smear:
            self.rc = 4.0/self.fourier_smear
        else:
            self.rc = rc
        self.rv = 1.02*self.rc
        self.skin = self.rv - self.rc
        self.repeat_count = int(np.ceil(2*self.rv/self.box_length))
        self.x_global = self.N_grid//2 + x_off
        self.y_global = self.N_grid//2 + y_off
        self.z_global = self.N_grid//2 + z_off
        self.x0 = self.mini_grid + x_off
        self.y0 = self.mini_grid + y_off
        self.z0 = self.mini_grid + z_off
        self.tol = 1e-3
        self.offset = offset
        self.comment = comment
        self.offset_vector = offset_vector
        self.default_s_q = default_s_q
        self.q_val = q_val
        self.q_tol = q_tol
        self.filename = filename
        self.total_steps = total_steps
        self.timestep = timestep
        self.S_Q = S_Q
        self.I_Q = I_Q
        self.ISF = ISF
        self.smeared_integral = smeared_integral
        self.direct_point = direct_point
        self.atoms_add = atoms_add
        self.frames = frames

    def generate_configuration(self):
        if self.position == 'test':
            N_atoms = 4
            pos = self.offset*self.N_grid*np.repeat(np.array([[0,0,1]]),N_atoms,axis = 0) \
                + self.box_length/2 \
                + np.array([[1,0,0],[0,1,0],[0,0,1],[1/3-np.sqrt(10)/3,1/3-np.sqrt(10)/3,1/3-np.sqrt(10)/3]])
            hin = np.array([[0.0,self.box_length],[0.0,self.box_length],[0.0,self.box_length]])
        elif self.position == 'single':
            N_atoms = 1
            pos = self.box_length*np.array([[2.0,1.0,0.6]])
            hin = np.array([[0.0,self.box_length],[0.0,self.box_length],[0.0,self.box_length]])
        elif self.position == 'file':
            pos, hins, N_atoms = load_dumpfile_atom_data(self.filename, 1, 1, verbose=False, h_full=False)
            hin = hins[:,:,0]
            self.box_length=np.mean(hin[:,1]-hin[:,0])
        elif self.position == 'time':
            if self.total_steps < 100:
                pos, hins, N_atoms = load_dumpfile_atom_data(self.filename, self.total_steps, 1, verbose=False, h_full=False)
            else:
                pos, hins, N_atoms = load_dumpfile_atom_data_fast(self.filename, self.total_steps, 1, verbose=False, h_full=False)
            hin = hins[:,:,0]
            self.box_length=np.mean(hin[:,1]-hin[:,0])
        if np.linalg.norm(np.diff(hin,axis=1)-self.box_length) > 1e-8:
            raise TypeError('The simulation box needs to be cubic')
        s = pos/self.box_length
        s = s - 0.5
        s = s-np.round(s)
        pos = self.box_length*(s+0.5)
        if self.position == 'time':
            if self.atoms_add is None:
                atoms_add = N_atoms
            else:
                atoms_add = self.atoms_add
            pos_input = np.zeros((atoms_add+2,3,self.total_steps))
            for i in range(self.total_steps):
                temp_pos = pos[i*atoms_add:(i+1)*atoms_add,:]
                pos_input[:2,:,i] = hin.T
                pos_input[2:,:,i] = temp_pos
            self.pos_input = pos_input
        self.pos = pos
        self.hin = hin
        self.N_atoms = N_atoms
    def generate_q_grid(self):
        x_grid = np.linspace(self.N_grid//2-self.mini_grid,self.N_grid//2+self.mini_grid,2*self.mini_grid+1,dtype=np.int)
        x_grid_global = np.linspace(0,self.N_grid-1,self.N_grid)*self.box_length/self.N_grid
        [X0,Y0,Z0] = np.meshgrid(x_grid_global,x_grid_global,x_grid_global)
        mini_grid_index = np.ix_(x_grid,x_grid,x_grid)
        q_grid = 2*np.pi*np.linspace(-(self.N_grid//2),self.N_grid//2-1,self.N_grid,dtype=np.int)/self.box_length
        [Qx, Qy, Qz] = np.meshgrid(q_grid,q_grid,q_grid)
        Q_line = np.zeros((self.N_grid**3,3))
        Q_line[:,0] = Qx.reshape(-1,1).flatten()
        Q_line[:,1] = Qy.reshape(-1,1).flatten()
        Q_line[:,2] = Qz.reshape(-1,1).flatten()
        q_grid_magnitude = np.linalg.norm(Q_line,axis=1).reshape(Qx.shape)
        q_probe = q_grid_magnitude[mini_grid_index]
        probe_index = np.where(abs(q_probe.reshape(-1,1)-self.q_val)<self.q_tol)[0]
        q_values = len(x_grid)
        sigma_weights = self.fourier_smear/(2*np.pi)
        smear_width = int(np.ceil(6*sigma_weights*self.box_length))
        [X_grid,Y_grid,Z_grid]=np.meshgrid(x_grid,x_grid,x_grid)
        if self.position == 'time':
            point_grid_x = x_grid[self.x0-smear_width:self.x0+smear_width+1]-(self.N_grid//2-self.mini_grid)
            point_grid_y = x_grid[self.y0-smear_width:self.y0+smear_width+1]-(self.N_grid//2-self.mini_grid)
            point_grid_z = x_grid[self.z0-smear_width:self.z0+smear_width+1]-(self.N_grid//2-self.mini_grid)
            dummy_grid = np.zeros(X_grid.shape)
            dummy_ind =np.ix_(point_grid_y,point_grid_x,point_grid_z)
            dummy_grid[dummy_ind] = 1.0
            point_grid = np.where(dummy_grid.reshape(-1,1)==1.0)[0]
            self.point_grid = point_grid
        scale_pos = np.zeros((3,3))
        h = self.hin
        scale_pos[:2,:] = h.T
        scale_pos[2,:] = self.box_length*np.array([0.0,0.0,0.0])
        smear_adjust = s_q_from_pos_smear(scale_pos,self.N_grid, self.sigma_grid,
                             cs = self.density_cutoff, ms = self.mini_grid, dump=True, structure_factor=True, correction = True)
        qx = 2*np.pi*(X_grid[self.y0,self.x0,self.z0]-self.N_grid//2)/self.box_length
        qy = 2*np.pi*(Y_grid[self.y0,self.x0,self.z0]-self.N_grid//2)/self.box_length
        qz = 2*np.pi*(Z_grid[self.y0,self.x0,self.z0]-self.N_grid//2)/self.box_length
        qz_array = 2*np.pi*(Z_grid[self.y0,self.x0,:]-self.N_grid//2)/self.box_length
        q3_point=np.array([[qx,qy,qz]])
        q3_array = np.zeros((q_values,3))
        q3_array[:,0] = q3_point[0,0]*np.ones(q_values)
        q3_array[:,1] = q3_point[0,1]*np.ones(q_values)
        q3_array[:,2] = qz_array
        q_magnitude_array = np.linalg.norm(q3_array,axis=1).flatten()  #Scaling q magnitude along the line in q-space
        q_magnitude = np.linalg.norm(q3_point,axis=1).flatten()  # Scaling q magnitude at chosen value
        xind = np.linspace(self.mini_grid-smear_width,self.mini_grid+smear_width,2*smear_width+1,dtype=np.int)
        yind = np.linspace(self.mini_grid-smear_width,self.mini_grid+smear_width,2*smear_width+1,dtype=np.int)
        zind = np.linspace(self.mini_grid-smear_width,self.mini_grid+smear_width,2*smear_width+1,dtype=np.int)
        [x_weight_grid,y_weight_grid,z_weight_grid]=np.meshgrid(x_grid[xind],x_grid[yind],x_grid[zind])
        grid_center = np.array([X_grid[self.mini_grid,self.mini_grid,self.mini_grid],Y_grid[self.mini_grid,self.mini_grid,self.mini_grid],Z_grid[self.mini_grid,self.mini_grid,self.mini_grid]])
        weights = norm3d(grid_center/self.box_length,x_weight_grid/self.box_length,y_weight_grid/self.box_length,z_weight_grid/self.box_length,sigma_weights,self.N_grid//2,self.N_grid//2,self.N_grid//2)
        weights = np.reshape(weights,(-1,1))
        self.q_values = q_values
        self.qz_array = qz_array
        self.q3_point = q3_point
        self.q3_array = q3_array
        self.q_magnitude = q_magnitude
        self.q_magnitude_array = q_magnitude_array
        self.weights = weights
        self.smear_width = smear_width
        self.probe_index = probe_index
        self.q_grid_magnitude = q_grid_magnitude
        self.rescale_factor = smear_adjust
    def real_space_s_q_pairwise(self):
        h = np.diag(self.box_length*np.ones(3))
        if self.position != 'time':
            pos_ext, h_ext = config_repeater(self.pos,h,size=self.repeat_count)
            hinv=np.linalg.inv(h_ext)
            nn, index = verletlist(pos_ext, h_ext, self.rv, atoms = self.N_atoms)
            pos_ref = pos_ext
            r_array = get_r_array(pos_ext, h_ext, self.rc, nnlist=index, atoms_add = self.N_atoms)
            r_sq=np.linalg.norm(r_array,axis=1)
            rho = self.N_atoms/np.linalg.det(h)
            if self.fourier_smear < 1/4.0:
                set_num_threads(8)
            s_pairwise = s_q3_from_pos_par(self.q3_array, r_array, self.rc, rho, self.N_atoms,  smear = self.smear, ddq = self.fourier_smear , r_sq=r_sq)
            if self.S_Q:
                self.s_pairwise = s_pairwise.real
            elif self.I_Q:
                self.s_pairwise = s_pairwise.real*form_factor_analytical(self.q_magnitude_array)**2
        else:
            s_pairwise = np.zeros(self.total_steps)
            for i in range(self.total_steps):
                pos_time = self.pos[i*self.N_atoms:(i+1)*self.N_atoms]
                pos_ext, h_ext = config_repeater(pos_time,h,size=self.repeat_count)
                hinv=np.linalg.inv(h_ext)
                if i==0:
                    nn, index = verletlist(pos_ext, h_ext, self.rv, atoms = self.N_atoms)
                    pos_ref = pos_ext
                else:
                    if np.max(np.linalg.norm(pbc(pos_ext-pos_ref,h_ext,hinv),axis=1))>self.skin/2:
                        nn, index = verletlist(pos_ext, h_ext, self.rv, atoms = self.N_atoms)
                        pos_ref = pos_ext
                r_array = get_r_array(pos_ext, h_ext, self.rc, nnlist=index, atoms_add = self.N_atoms)
                r_sq=np.linalg.norm(r_array,axis=1)
                rho = self.N_atoms/np.linalg.det(h)
                set_num_threads(8)
                s_pairwise[i] = s_q3_from_pos_par(self.q3_point, r_array, self.rc, rho, self.N_atoms,  smear = self.smear, ddq = self.fourier_smear , r_sq=r_sq)
            if self.S_Q:
                self.s_pairwise = s_pairwise.real
            elif self.I_Q:
                self.s_pairwise = s_pairwise.real*form_factor_analytical(self.q_magnitude)**2

    def real_space_s_q_position(self):
        s_position= np.zeros(self.q_values,dtype = np.complex)
        for j in range(self.q_values):
            q3 = self.q3_array[j,:]
            q = np.linalg.norm(q3)
            adj_ISF = np.exp(-1j*np.dot(self.pos,q3.T)).sum()
            s_position[j]=adj_ISF*np.conj(adj_ISF)/self.N_atoms

        if self.S_Q:
            self.s_position = s_position.real
        elif self.I_Q:
            self.s_position = s_position.real*form_factor_analytical(self.q_magnitude_array)**2

    def fourier_space(self):
        s_smear = np.zeros(self.q_values,dtype = np.complex)
        h_pos = np.zeros((self.N_atoms+2,3))
        h = self.hin
        h_pos[:2,:] = h.T
        h_pos[2:,:] = self.pos
        grid_s_q = self.rescale_factor*s_q_from_pos_smear(h_pos,self.N_grid, self.sigma_grid,
                             cs = self.density_cutoff, ms = self.mini_grid, dump=True, structure_factor=True, q_magnitude = self.q_grid_magnitude)
        s_fourier = (grid_s_q[self.y_global,
                              self.x_global,
                              self.N_grid//2-self.mini_grid:self.N_grid//2+self.mini_grid+1] \
                              ).astype(np.complex)
        if self.S_Q:
            self.s_fourier = s_fourier.real
        elif self.I_Q:
            self.s_fourier = s_fourier.real*form_factor_analytical(self.q_magnitude_array)**2
        for i in range(self.q_values):
            s_q_integration_grid = np.zeros((2*self.smear_width+1)**3)
            xind = np.linspace(self.x_global-self.smear_width,self.x_global+self.smear_width,2*self.smear_width+1,dtype=np.int)
            yind = np.linspace(self.y_global-self.smear_width,self.y_global+self.smear_width,2*self.smear_width+1,dtype=np.int)
            zind = np.linspace(self.N_grid//2-self.mini_grid+i-self.smear_width,self.N_grid//2-self.mini_grid+i+self.smear_width,2*self.smear_width+1,dtype=np.int)
            fill_ind  = np.ix_(yind,xind,zind)
            s_q_integration_grid = np.reshape(grid_s_q[fill_ind],(-1,1))
            integrated_array = np.dot(np.diag(self.weights.flatten()),s_q_integration_grid)*self.box_length**-3
            s_smear[i] = np.sum(integrated_array,axis=0)
        if self.S_Q:
            self.s_fourier_smear = s_smear.real
        elif self.I_Q:
            self.s_fourier_smear = s_smear.real*form_factor_analytical(self.q_magnitude_array)**2
    def fourier_space_time(self):
        pool=mp.Pool(mp.cpu_count())
        if self.smeared_integral:
            compute_partial = partial(s_q_from_pos_smear,N = self.N_grid, wdt = self.sigma_grid,
                             cs = self.density_cutoff, ms = self.mini_grid, dump=True,structure_factor=True, correction_grid = self.rescale_factor, q_magnitude = self.q_grid_magnitude, ind_need = self.point_grid)
        elif self.direct_point:
            compute_partial = partial(s_q_from_pos_smear,N = self.N_grid, wdt = self.sigma_grid,
                         cs = self.density_cutoff, ms = self.mini_grid, dump=True, ISF=self.ISF, intensity =self.I_Q, structure_factor=self.S_Q, correction_grid = self.rescale_factor, q_magnitude = self.q_grid_magnitude, ind_need = self.point_grid)
        s_time = pool.map(compute_partial,[self.pos_input[:,:,r] for r in range(self.total_steps)])
        s_time = np.asarray(s_time)
        s_time = s_time.T.real
        if self.smeared_integral:
            integrated_array = (self.I_Q*(form_factor_analytical(self.q_magnitude)**2-1.0)+1.0)*np.dot(np.diag(self.weights.flatten()),s_time)*self.box_length**-3
            s_time_array = np.sum(integrated_array,axis=0)
            self.s_fourier_time = s_time_array
        elif self.direct_point:
            self.s_fourier_time = s_time

    def test_consistency(self, plot_results = False, fig_id = 1):
        self.generate_configuration()
        self.generate_q_grid()
        self.real_space_s_q_pairwise()
        self.real_space_s_q_position()
        self.fourier_space()
        print()
        print('*'*50)
        print('Test condition: %s'%self.comment)
        print('*'*50)
        print()
        print('The s(q) values obtained:\n'
              '1) From the real space method (position): %.12f\n'
              '2) From the real space method (pairwise): %.12f\n'
              '3) From the fourier space method:         %.12f\n'
              '4) From the fourier space method (smear): %.12f'
              %(self.s_position[self.z0],self.s_pairwise[self.z0],self.s_fourier[self.z0],self.s_fourier_smear[self.z0]))
        if self.position == 'test':
            error_pos_four = abs((self.default_s_q-self.s_fourier[self.z0])/self.default_s_q)
            error_pair_four = abs((self.default_s_q-self.s_position[self.z0])/self.default_s_q)
        else:
            error_pos_four = abs((self.s_position-self.s_fourier)/self.s_position)
            error_pair_four = abs((self.s_pairwise-self.s_fourier_smear)/self.s_pairwise)
        cleared_count = int(error_pos_four.mean()  < self.tol) \
                      + int(error_pair_four.mean() < self.tol)
        if cleared_count == 2:
            print('TEST: '+bcolors.GRN+'PASSED'+bcolors.RESET )
            test_success = True
        else:
            print('TEST: '+bcolors.RED+'FAILED'+bcolors.RESET )
            test_success = False
        if plot_results:
            plt.figure(fig_id)
            plt.plot(self.qz_array, self.s_position,     label = 'Real space - position')
            plt.plot(self.qz_array, self.s_pairwise, ':',label = 'Real space - pairwise')
            plt.plot(self.qz_array, self.s_fourier,  'o',label = 'Fourier space')
            plt.plot(self.qz_array, self.s_fourier_smear,  '*',label = 'Fourier space - smeared')
            plt.legend(loc='best')
        return test_success
    def compute_results(self):
        self.generate_configuration()
        self.generate_q_grid()
        self.real_space_s_q_pairwise()
        self.real_space_s_q_position()
        self.fourier_space()
    def compute_and_save(self):
        self.generate_configuration()
        self.generate_q_grid()
        if self.position == 'file':
            self.fourier_space()
            self.real_space_s_q_position()
            self.real_space_s_q_pairwise()
            np.savetxt('gg_cor_%1.2f.txt'%(self.fourier_smear),self.s_pairwise)
            np.savetxt('g_cor_%1.2f.txt'%(self.fourier_smear),self.s_fourier_smear)
            np.savetxt('xx_cor_%1.2f.txt'%(self.fourier_smear),self.s_position)
            np.savetxt('x_cor_%1.2f.txt'%(self.fourier_smear),self.s_fourier)
        elif self.position =='time':
            self.fourier_space_time()
            np.savetxt('s_cor_%1.2f.txt'%(self.fourier_smear),self.s_fourier_time)
            self.real_space_s_q_pairwise()
            np.savetxt('ss_cor_%1.2f.txt'%(self.fourier_smear),self.s_pairwise)
    def time_auto_correlation(self):
        if self.position != 'time':
            raise TypeError('Position type error: Valid only for time')
        pool=mp.Pool(mp.cpu_count())
        correl = partial(auto_corr,frames=self.frames)
        items = self.s_fourier_time.shape[0]
        g2 = pool.map(correl, [self.s_fourier_time[i,:] for i in range(items)])
        g2 = np.asarray(g2)
        beta=g2[:,0]-1.0
        g2_exp_fit = np.dot(np.diag(1.0/(beta)),(g2-1.0))
        g2_exp_fit_mean = np.mean(g2_exp_fit,axis=0)
        g2_mean = np.mean(g2,axis=0)
        self.g2 = g2
        self.g2_exp_fit = g2_exp_fit
        self.g2_mean = g2_mean
        self.g2_exp_fit_mean = g2_exp_fit_mean
    def compute_and_save_correlation(self):
        self.generate_configuration()
        self.generate_q_grid()
        self.fourier_space_time()
        self.time_auto_correlation()
        np.savez('ts_cor_%.2f_%.2f.npz'%(self.q_val,self.atoms_add),b = self.g2)
        np.savez('fit_cor_%.2f_%.2f.npz'%(self.q_val,self.atoms_add),b = self.g2_exp_fit)
        np.savetxt('ts_cor_%.2f_%.2f.txt'%(self.q_val,self.atoms_add), self.g2_mean)
        np.savetxt('fit_cor_%.2f_%.2f.txt'%(self.q_val,self.atoms_add), self.g2_exp_fit_mean)

class XPCS_Suite():
    def __init__(self, filename = 'dump.ljtest', q_val = 6.28, frames = 33,
                 pool = None, atoms_add = None, sigma_grid = 400, ISF = False,
                 N_grid = 400, mini_grid = 40, density_cutoff = 5, q_tol = 0.05,
                 intensity =  True, total_steps = 5000, timestep = 0.05,
                 system = 'liquid', atom_type = None, dir_name = ''):
        self.filename = filename
        self.q_val = q_val
        self.frames = frames
        self.pool = pool
        self.atoms_add = atoms_add
        self.sigma_grid = sigma_grid
        self.N_grid = N_grid
        self.mini_grid = mini_grid
        self.density_cutoff = density_cutoff
        self.q_tol = q_tol
        self.intensity = intensity
        self.total_steps = total_steps
        self.ISF = ISF
        self.timestep = timestep
        self.atom_type = atom_type
        self.system = system
        self.dir_name = dir_name
        self.times = np.linspace(0, self.timestep * (self.frames - 1), self.frames)
        if system == 'liquid':
            self.nondim_t = 2.156e-12
            self.nondim_d = 3.405e-10
        elif system == 'DPN':
            if self.atom_type is None:
                raise TypeError("atom_type must be assigned out of ligand or metal: atom_type = \'metal\'")
            self.nondim_t = 2.3e-10
            self.nondim_d = 15e-10
        self.times_ps = self.times/self.nondim_t*1e12
        self.times_fs = self.times / self.nondim_t * 1e15

    def load_and_process_trajectory(self):
        if self.system == 'liquid':
            if self.total_steps < 100:
                self.pos, self.hins, self.N_atoms = load_dumpfile_atom_data(self.filename, self.total_steps, 1, verbose=False, h_full=False)
            else:
                self.pos, self.hins, self.N_atoms = load_dumpfile_atom_data_fast(self.filename, self.total_steps, 1, verbose=False, h_full=False)
        elif self.system == 'DPN':
            self.pos_a, self.pos_b, self.hins, self.atoms_a, self.atoms_b = load_dumpfile_atom_data_binary_fast(self.filename, self.total_steps,
                                                                                      1, verbose=False,
                                                                                      family='polymer', h_full=False)
            if self.atom_type == 'metal':
                self.pos = self.pos_a.copy()
                self.N_atoms = self.atoms_a
            elif self.atom_type == 'ligand':
                self.pos = self.pos_b.copy()
                self.N_atoms = self.atoms_b
        self.hin = self.hins[:, :, 0]
        self.box_length = np.mean(self.hin[:, 1] - self.hin[:, 0])
        s = self.pos / self.box_length
        s = s - 0.5
        s = s - np.round(s)
        self.pos = self.box_length * (s + 0.5)
        self.h = self.hin
        if self.atoms_add is None:
            self.atoms_add = self.N_atoms
        else:
            self.atoms_add = self.atoms_add
        self.pos_input = np.zeros((self.atoms_add + 2, 3, self.total_steps))
        for i in range(self.total_steps):
            temp_pos = self.pos[i * self.N_atoms:i * self.N_atoms + self.atoms_add, :]
            self.pos_input[:2, :, i] = self.hin.T
            self.pos_input[2:, :, i] = temp_pos

    def prepare_intensity_correction(self):
        self.scale_pos = np.zeros((3, 3))
        self.scale_pos[:2, :] = self.h.T
        self.scale_pos[2, :] = self.box_length * np.array([0.0, 0.0, 0.0])
        if self.intensity:
            self.rescale_factor = s_q_from_pos_smear(self.scale_pos, N=self.N_grid, wdt=self.sigma_grid,
                                            cs=self.density_cutoff, ms=self.mini_grid, dump=True, structure_factor=True,
                                            correction=True)
        if self.ISF:
            self.rescale_factor_ISF = s_q_from_pos_smear(self.scale_pos, N=self.N_grid, wdt=self.sigma_grid,
                                                cs=self.density_cutoff, ms=self.mini_grid, dump=True, ISF=True, correction=True)

    def prepare_fourier_grid(self):
        x_grid = np.linspace(self.N_grid // 2 - self.mini_grid, self.N_grid // 2 + self.mini_grid, 2 * self.mini_grid + 1, dtype=np.int)
        mini_grid_index = np.ix_(x_grid, x_grid, x_grid)
        q_grid = 2 * np.pi * np.linspace(-(self.N_grid // 2), self.N_grid // 2 - 1, self.N_grid, dtype=np.int) / self.box_length
        [Qx, Qy, Qz] = np.meshgrid(q_grid, q_grid, q_grid)
        Q_line = np.zeros((self.N_grid ** 3, 3))
        Q_line[:, 0] = Qx.reshape(-1, 1).flatten()
        Q_line[:, 1] = Qy.reshape(-1, 1).flatten()
        Q_line[:, 2] = Qz.reshape(-1, 1).flatten()
        self.q_grid_magnitude = np.linalg.norm(Q_line, axis=1).reshape(Qx.shape)
        self.q_probe = self.q_grid_magnitude[mini_grid_index]
        if self.intensity:
            self.correct_grid = self.rescale_factor[mini_grid_index].reshape(-1, 1)
        if self.ISF:
            self.correct_grid_ISF = self.rescale_factor_ISF[mini_grid_index].reshape(-1, 1)
        self.probe_index = np.where(abs(self.q_probe.reshape(-1, 1) - self.q_val) < self.q_tol)[0]
        if len(self.probe_index) == 0:
            raise TypeError('No points selected')

    def solve_intensity(self):
        if self.intensity:
            compute_partial = partial(s_q_from_pos_smear, N=self.N_grid, wdt=self.sigma_grid,
                                      cs=self.density_cutoff, ms=self.mini_grid, dump=True, structure_factor=True,
                                      q_magnitude=self.q_grid_magnitude, ind_need=self.probe_index)
            self.s_time = self.pool.map(compute_partial, [self.pos_input[:, :, r] for r in range(self.total_steps)])
            self.s_time = np.asarray(self.s_time)
            self.s_time = self.s_time.T.real
            self.s_time = np.dot(np.diag(self.correct_grid[self.probe_index].flatten()), self.s_time)
        if self.ISF:
            compute_partial_ISF = partial(s_q_from_pos_smear, N=self.N_grid, wdt=self.sigma_grid,
                                          cs=self.density_cutoff, ms=self.mini_grid, dump=True, ISF=True,
                                          q_magnitude=self.q_grid_magnitude, ind_need=self.probe_index)
            self.s_time_ISF = self.pool.map(compute_partial_ISF, [self.pos_input[:, :, r] for r in range(self.total_steps)])
            self.s_time_ISF = np.asarray(self.s_time_ISF)
            self.s_time_ISF = self.s_time_ISF.T.real
            self.s_time_ISF = np.dot(np.diag(self.correct_grid_ISF[self.probe_index].flatten()), self.s_time_ISF)

    def solve_correlations(self):
        if self.intensity:
            self.items = self.s_time.shape[0]
            correl = partial(auto_corr, frames=self.frames)
            self.g2 = self.pool.map(correl, [self.s_time[i, :] for i in range(self.items)])
            self.g2 = np.asarray(self.g2)
            self.beta = self.g2[:, 0] - 1.0
            self.g2_exp_fit = np.dot(np.diag(1.0 / (self.beta)), (self.g2 - 1.0))
            self.g2_exp_fit_mean = np.mean(self.g2_exp_fit, axis=0)
            self.g2_mean = np.mean(self.g2, axis=0)
        if self.ISF:
            self.items = self.s_time_ISF.shape[0]
            correl_ISF = partial(ISF_corr, frames=self.frames)
            self.f_qt = self.pool.map(correl_ISF, [self.s_time_ISF[i, :] for i in range(self.items)])
            self.f_qt = np.asarray(self.f_qt)
            self.F_qt = np.mean(self.f_qt, axis=0) ** 2

    def save_correlations(self):
        if self.intensity:
            np.savez(self.dir_name+'ts_cor_%.2f_%d_%s.npz' % (self.q_val, self.atoms_add, self.ext), b=self.g2.real)
            np.savez(self.dir_name+'s_cor_%.2f_%d_%s.npz' % (self.q_val, self.atoms_add, self.ext), b=self.s_time.real)
            np.savez(self.dir_name+'fit_cor_%.2f_%d_%s.npz' % (self.q_val, self.atoms_add, self.ext), b=self.g2_exp_fit.real)
            np.savetxt(self.dir_name+'ts_cor_%.2f_%d_%s.txt' % (self.q_val, self.atoms_add, self.ext), self.g2_mean.real)
            np.savetxt(self.dir_name+'Auto_corr_%.2f_%d_%s.txt' % (self.q_val, self.atoms_add, self.ext), self.g2_exp_fit_mean.real)
        if self.ISF:
            np.savetxt(self.dir_name+'ISF_corr_%.2f_%d_%s.txt' % (self.q_val, self.atoms_add, self.ext), self.F_qt)

    def contrast_computation(self):
        tots = [1, 10, 100, 500, 1000, 2000, 5000]
        tots_ps = np.array(tots)*self.timestep/self.nondim_t*1e12
        tots_fs = np.array(tots) * self.timestep / self.nondim_t * 1e15
        st_array = []
        for i,tot in enumerate(tots):
            s_t = self.s_time[:,:tot].mean(1)
            st_array.append([tots_ps[i],optical_contrast(s_t)])

        self.contrast_data = np.array(st_array)

    def test_class(self):
        g2_func = np.load('ts_cor_%.2f_%d_1000.npz' % (self.q_val, self.atoms_add))['b']
        if np.allclose(self.g2,g2_func):
            print("Class carries out the same computation")
        else:
            print("Class fails the same computation")

    def test_class_workflow(self):
        self.load_and_process_trajectory()
        self.prepare_intensity_correction()
        self.prepare_fourier_grid()
        self.solve_intensity()
        self.solve_correlations()
        self.test_class()

    def get_gamma(self):
        popt, _ = curve_fit(objective_lin,self.times[3:19], np.log(self.g2_exp_fit_mean[3:19]))
        if self.q_val == 6.28:
            print(self.times[3:19], np.log(self.g2_exp_fit_mean[3:19]))
        a, b  = popt
        self.gamma_ns = -0.5*a*1e-9/self.nondim_t

    def store_gamma_workflow(self):
        self.prepare_intensity_correction()
        self.prepare_fourier_grid()
        self.solve_intensity()
        self.solve_correlations()
        self.get_gamma()

    def get_intensity_workflow(self):
        self.prepare_intensity_correction()
        self.prepare_fourier_grid()
        self.solve_intensity()

class test_analysis():
    def __init__(self, dump_filename = 'dump_name', filename = 'MSD',
                 save_dir = 'new_data', steps = 5000, atoms_add = 45):
        self.dump_filename  = dump_filename
        self.filename       = filename
        self.save_dir       = save_dir
        self.steps          = steps
        self.skip_to_np     = 2
        self.skip_to_box    = 5
        self.skip_to_pos    = 19
        self.r0             = 0.5  # minimum r for g(r) histogram
        self.rc             = 4.0;  # cut-off radius of g(r) calculation
        self.bins           = np.arange(self.r0, self.rc + 0.01, 0.01)
        self.q_array        = np.arange(1.0, 15.01, 0.01)

        self.atoms_add      = atoms_add
        self.sigma_grid     = 400
        self.N_grid         = 400
        self.mini_grid      = 40
        self.density_cutoff = 5
        self.q_tol          = 0.05
        self.q_val          = 6.28
        self.I_Q            = True
        self.frames         = 33
        self.ext            = '1000'

    def init_parallel(self):
        self.pool = mp.Pool(mp.cpu_count())

    def read_data(self):
        self.pos, self.h = load_atom_data(self.filename, self.skip_to_np, self.skip_to_box, self.skip_to_pos, verbose=False)
        self.rho = self.pos.shape[0] / np.linalg.det(self.h)

    def read_dump(self):
        self.pos, self.h, self.atoms = load_dumpfile_atom_data(self.dump_filename, self.steps, 1, verbose=False)

    def read_dump_fast(self):
        self.pos, self.hins, self.N_atoms = load_dumpfile_atom_data_fast(self.dump_filename, self.steps, 1, verbose=False, h_full=False)
        self.hin = self.hins[:, :, 0]
        if self.atoms_add is None:
            self.atoms_add = self.N_atoms

    def pre_process_corr(self):
        self.box_length = np.mean(self.hin[:, 1] - self.hin[:, 0])
        s = self.pos / self.box_length
        s = s - 0.5
        s = s - np.round(s)
        self.pos = self.box_length * (s + 0.5)
        self.h = self.hin
        self.pos_input = np.zeros((self.atoms_add + 2, 3, self.steps))
        for i in range(self.steps):
            temp_pos = self.pos[i * self.N_atoms:i * self.N_atoms + self.atoms_add, :]
            self.pos_input[:2, :, i] = self.hin.T
            self.pos_input[2:, :, i] = temp_pos

    def generate_q_grid(self):
        Ex = 5000  # energy of X-ray in eV
        lamx_in_A = 12398.0 / Ex  # wave length of X-ray in angstrom https://www.kmlabs.com/en/wavelength-to-photon-energy-calculator
        lj_len_in_A = 10  # lj length scale in angstrom
        lamx_in_lj = lamx_in_A / lj_len_in_A  # wave length of X-ray in lj length scale
        k0 = 2 * np.pi / lamx_in_lj  # wave vector of incident X-ray in lj unit
        ky_relative = np.arange(-0.5, 0.52, 0.02)
        kz_relative = np.arange(-0.5, 0.52, 0.02)
        self.q3_array = get_q3(k0, ky_relative, kz_relative)

    def g_r_compute(self):
        self.read_data()
        nn, index = verletlist(self.pos, self.h, self.rc)
        r_array, g_array = g_r_verlet(self.pos, self.bins, self.rc, self.h, nnlist=index)
        g_array = f_weight_array(g_array, r_array, self.rc)
        s_array = s_q_from_g_r(self.q_array, r_array, g_array, self.r0, self.rho)
        s_array_1 = s_q_from_pos(self.q_array, self.pos, self.h, self.rc, self.rho, nnlist=index)
        np.savez(self.save_dir+'/correlation_frame.npz', gr=g_array, sq1=s_array, sq2=s_array_1)

    def g_r_compute_time(self):
        self.read_dump()
        sx_array = np.zeros((self.steps, self.bins.shape[0] - 1))
        sy_array = np.zeros((self.steps, self.bins.shape[0] - 1))
        for i in range(self.steps):
            sx_array[i, :], sy_array[i, :] = g_r_verlet(self.pos[i * self.atoms:(i + 1) * self.atoms], self.bins, self.rc, np.diag(self.h[i, :]))

        np.savez(self.save_dir+'/all_gr.npz', rr=sx_array, gr=sy_array)

    def diffraction_grid(self):
        self.read_data()
        self.generate_q_grid()
        nn, index = verletlist(self.pos, self.h, self.rc)
        r_array = get_r_array(self.pos, self.h, self.rc, nnlist=index)
        q3_line = np.stack([np.zeros_like(self.q_array), self.q_array, np.zeros_like(self.q_array)], axis=-1)
        s_line = s_q3_from_pos_par(q3_line, r_array, self.rc, self.rho, self.pos.shape[0])
        q3_flat = self.q3_array.reshape([-1, 3])
        s_flat = s_q3_from_pos_par(q3_flat, r_array, self.rc, self.rho, self.pos.shape[0])
        s_array = s_flat.real.reshape(self.q3_array.shape[:-1])
        np.savez(self.save_dir+'/single_frame.npz', frames=s_array)

    def diffraction_grid_time(self):
        self.read_dump()
        self.generate_q_grid()
        q3_flat = self.q3_array.reshape([-1, 3])
        s_arrays = np.zeros((self.steps, self.q3_array.shape[0], self.q3_array.shape[1]))
        for i in range(self.steps):
            nn, index = verletlist(self.pos[i * self.atoms:(i + 1) * self.atoms], np.diag(self.h[i, :]), self.rc)
            rho = self.atoms / np.linalg.det(np.diag(self.h[i, :]))
            r_array = get_r_array(self.pos[i * self.atoms:(i + 1) * self.atoms], np.diag(self.h[i, :]), self.rc, nnlist=index)
            s_flat = s_q3_from_pos_par(q3_flat, r_array, self.rc, rho, self.atoms)
            s_arrays[i, :, :] = s_flat.real.reshape(self.q3_array.shape[:-1])  # For I(q)

        np.savez(self.save_dir+'/multiple_frames.npz', frames=s_arrays)

    def correlation_computer(self):
        self.read_dump_fast()
        self.pre_process_corr()
        pool = mp.Pool(mp.cpu_count())
        scale_pos = np.zeros((3, 3))
        scale_pos[:2, :] = self.h.T
        scale_pos[2, :] = self.box_length * np.array([0.0, 0.0, 0.0])
        rescale_factor = s_q_from_pos_smear(scale_pos, N=self.N_grid, wdt=self.sigma_grid,
                                            cs=self.density_cutoff, ms=self.mini_grid, dump=True, structure_factor=True,
                                            correction=True)
        rescale_factor_ISF = s_q_from_pos_smear(scale_pos, N=self.N_grid, wdt=self.sigma_grid,
                                                cs=self.density_cutoff, ms=self.mini_grid, dump=True, ISF=True, correction=True)
        x_grid = np.linspace(self.N_grid // 2 - self.mini_grid, self.N_grid // 2 + self.mini_grid, 2 * self.mini_grid + 1, dtype=np.int)
        mini_grid_index = np.ix_(x_grid, x_grid, x_grid)
        q_grid = 2 * np.pi * np.linspace(-(self.N_grid // 2), self.N_grid // 2 - 1, self.N_grid, dtype=np.int) / self.box_length
        [Qx, Qy, Qz] = np.meshgrid(q_grid, q_grid, q_grid)
        Q_line = np.zeros((self.N_grid ** 3, 3))
        Q_line[:, 0] = Qx.reshape(-1, 1).flatten()
        Q_line[:, 1] = Qy.reshape(-1, 1).flatten()
        Q_line[:, 2] = Qz.reshape(-1, 1).flatten()
        q_grid_magnitude = np.linalg.norm(Q_line, axis=1).reshape(Qx.shape)
        q_probe = q_grid_magnitude[mini_grid_index]
        correct_grid = rescale_factor[mini_grid_index].reshape(-1, 1)
        correct_grid_ISF = rescale_factor_ISF[mini_grid_index].reshape(-1, 1)
        probe_index = np.where(abs(q_probe.reshape(-1, 1) - self.q_val) < self.q_tol)[0]
        if len(probe_index) == 0:
            raise TypeError('No points selected')
        compute_partial = partial(s_q_from_pos_smear, N=self.N_grid, wdt=self.sigma_grid,
                                  cs=self.density_cutoff, ms=self.mini_grid, dump=True, structure_factor=True,
                                  q_magnitude=q_grid_magnitude, ind_need=probe_index)
        s_time = pool.map(compute_partial, [self.pos_input[:, :, r] for r in range(self.steps)])
        s_time = np.asarray(s_time)
        s_time = s_time.T.real
        s_time = np.dot(np.diag(correct_grid[probe_index].flatten()), s_time)
        compute_partial_ISF = partial(s_q_from_pos_smear, N=self.N_grid, wdt=self.sigma_grid,
                                      cs=self.density_cutoff, ms=self.mini_grid, dump=True, ISF=True,
                                      q_magnitude=q_grid_magnitude, ind_need=probe_index)
        s_time_ISF = pool.map(compute_partial_ISF, [self.pos_input[:, :, r] for r in range(self.steps)])
        s_time_ISF = np.asarray(s_time_ISF)
        s_time_ISF = s_time_ISF.T.real
        s_time_ISF = np.dot(np.diag(correct_grid_ISF[probe_index].flatten()), s_time_ISF)

        correl = partial(auto_corr, frames=self.frames)
        correl_ISF = partial(ISF_corr, frames=self.frames)
        items = s_time.shape[0]
        g2 = pool.map(correl, [s_time[i, :] for i in range(items)])
        g2 = np.asarray(g2)
        f_qt = pool.map(correl_ISF, [s_time_ISF[i, :] for i in range(items)])
        f_qt = np.asarray(f_qt)
        F_qt = np.mean(f_qt.real, axis=0) ** 2

        beta = g2[:, 0] - 1.0
        g2_exp_fit = np.dot(np.diag(1.0 / (beta)), (g2 - 1.0))
        g2_exp_fit_mean = np.mean(g2_exp_fit.real, axis=0)
        g2_mean = np.mean(g2, axis=0)
        np.savez(self.save_dir+'/ts_cor_%.2f_%d_%s.npz' % (self.q_val, self.atoms_add, self.ext), b=g2)
        np.savez(self.save_dir+'/s_cor_%.2f_%d_%s.npz' % (self.q_val, self.atoms_add, self.ext), b=s_time)
        np.savez(self.save_dir+'/ISF_cor_%.2f_%d_%s.npz' % (self.q_val, self.atoms_add, self.ext), b=s_time_ISF)
        np.savez(self.save_dir+'/fit_cor_%.2f_%d_%s.npz' % (self.q_val, self.atoms_add, self.ext), b=g2_exp_fit)
        np.savetxt(self.save_dir+'/ts_cor_%.2f_%d_%s.txt' % (self.q_val, self.atoms_add, self.ext), g2_mean)
        np.savetxt(self.save_dir+'/Auto_corr_%.2f_%d_%s.txt' % (self.q_val, self.atoms_add, self.ext), g2_exp_fit_mean)
        np.savetxt(self.save_dir+'/ISF_corr_%.2f_%d_%s.txt' % (self.q_val, self.atoms_add, self.ext), F_qt)

    def contrast_computer(self):
        pool = mp.Pool(mp.cpu_count())
        s_file = self.save_dir+'/s_cor_6.28_45_1000.npz'
        ISF_file = self.save_dir+'/ISF_cor_6.28_45_1000.npz'
        s_cor = np.load(s_file)['b']
        ISF_cor = np.load(ISF_file)['b']
        frames = 33
        correl = partial(auto_corr, frames=frames)
        correl_ISF = partial(ISF_corr, frames=frames)
        items = s_cor.shape[0]
        tots = [100, 500, 1000, 2000, 5000]
        fid = open(self.save_dir+'/contrast_g2_F.txt', 'w')
        for tot in tots:
            g2 = np.zeros((items, frames))
            f_qt = np.zeros((items, frames))
            time = 5e-2 * 2.156 * tot
            s_time = s_cor[:, :tot]
            s_time_ISF = ISF_cor[:, :tot]
            g2 = pool.map(correl, [s_time[i, :] for i in range(items)])
            g2 = np.asarray(g2)
            beta = g2[:, 0] - 1.0
            g2_exp_fit = np.dot(np.diag(1.0 / (beta)), (g2 - 1.0))
            g2_exp_fit_mean = np.mean(g2_exp_fit.real, axis=0)
            f_qt = pool.map(correl_ISF, [s_time_ISF[i, :] for i in range(items)])
            f_qt = np.asarray(f_qt)
            F_qt = np.mean(f_qt.real, axis=0) ** 2
            fid.write('%.4f \t %.4f \t %.4f\n' % (time, g2_exp_fit_mean[10], F_qt[10]))
        fid.close()

    def test_validity(self):
        d1 = 'old_data/'
        d2 = 'new_data/'
        txts = ['ISF_corr_6.28_45_1000.txt',
                'Auto_corr_6.28_45_1000.txt',
                'contrast_g2_F.txt']
        npzs = ['fit_cor_6.28_45_1000.npz',
              'ISF_cor_6.28_45_1000.npz',
              's_cor_6.28_45_1000.npz',
              'ts_cor_6.28_45_1000.npz',
               'all_gr.npz', 'multiple_frames.npz',
               'single_frame.npz', 'correlation_frame.npz']
        labels = [['b'],['b'],['b'],['b'],['rr','gr'],['frames'],['frames'],['gr','sq1','sq2']]
        for i,txt in enumerate(txts):
            c1 = np.loadtxt(d1 + txt)
            c2 = np.loadtxt(d2 + txt)
            if np.allclose(c1, c2):
                print("Test for text file %d passed"%(i + 1))
            else:
                print("Test for text file %d failed" % (i + 1))

        for i,npz in enumerate(npzs):
            for j,label in enumerate(labels[i]):
                c1 = np.load(d1 + npz)[label]
                c2 = np.load(d2 + npz)[label]
                if np.allclose(c1, c2):
                    print("Test for npz file %d and label %d passed" % (i + 1, j + 1))
                else:
                    print("Test for npz file %d and label %d failed" % (i + 1, j + 1))

class post_analysis():
    def __init__(self, timestep = 0.01078, dump_filename = 'dump_name', save_data = False,
                 save_dir = '../runs', plot_results = True, poolsize = None,
                 steps = 5000, freq = 1, atoms_add = 45, q_vals = None):
        self.timestep      = timestep
        self.dump_filename = dump_filename
        self.save_dir      = save_dir
        self.steps         = steps
        self.freq          = freq
        self.ins           = self.steps // self.freq
        self.t             = np.linspace(0,self.timestep*self.freq*(self.ins-1),self.ins)
        self.nd            = 3.405
        self.save_data     = save_data
        self.plot_results  = plot_results
        self.r0            = 0.25  # minimum r for g(r) histogram
        self.rc            = 4.0;  # cut-off radius of g(r) calculation
        self.bins          = np.arange(self.r0, self.rc + 0.01, 0.01)
        self.N             = 400
        self.wdt           = 400
        self.cs            = 5
        self.ms            = 40
        self.atoms_add     = atoms_add
        self.poolsize      = poolsize
        self.q_vals        = q_vals

    def init_parallel(self):
        if self.poolsize is None:
            self.pool = mp.Pool(mp.cpu_count())
        else:
            self.pool = mp.Pool(self.poolsize)

    def load_LAMMPS(self):
        self.md_pos, self.hins, self.N_atoms = load_dumpfile_atom_data_fast(self.dump_filename, self.steps, 1, verbose=False, h_full=False)
        self.hin = self.hins[:, :, 0]
        self.h = np.diag(self.hin[:, 1] - self.hin[:, 0])

    def cal_rdf_and_sq(self):
        self.load_LAMMPS()
        self.init_parallel()
        scale_pos = np.zeros((3, 3))
        r_array = 0.5 * (self.bins[:-1] + self.bins[1:])
        compute_MD = partial(gr_MD, bins=self.bins, rc=self.rc, h=self.h)
        g_arr_MD = self.pool.map(compute_MD,
                            [self.md_pos[i * self.freq * self.N_atoms:(i * self.freq + 1) * self.N_atoms, :].copy() for i in range(self.ins)])
        g_arr_MD = np.array(g_arr_MD)
        gr_md = g_arr_MD.mean(0)
        scale_pos[1, :] = self.hin[:, 1] - self.hin[:, 0]
        correct_MD = s_q_from_pos_smear(scale_pos, N=self.N, wdt=self.wdt, cs=self.cs, ms=self.ms, dump=True, structure_factor=True,
                                        correction=True)
        if self.save_data:
            np.savetxt(self.save_dir+'/MD_GR.txt', np.stack([r_array, gr_md], axis=1))
        if self.plot_results:
            plt.xlim(0, 4)
            plt.ylim(0, 3.5)
            plt.plot(r_array, gr_md, 'b:', label='g(r)')
            plt.xlabel(r'r')
            plt.ylabel(r'g(r)')
            plt.legend()
            plt.savefig(self.save_dir+'/MD_g_r.png')
            plt.clf()
        an_md_pos = np.zeros((self.N_atoms + 2, 3, self.steps))
        ################## S(q) ####################
        for i in range(self.steps):
            temp_pos = self.md_pos[i * self.N_atoms:(i + 1) * self.N_atoms, :]
            an_md_pos[:2, :, i] = self.hin.T
            an_md_pos[2:, :, i] = temp_pos

        q_md, _ = s_q_from_pos_smear_array(an_md_pos[:, :, 0], h=None, N=self.N, wdt=self.wdt, cs=self.cs, ms=self.ms, dump=True,
                                           correction_grid=correct_MD)

        compute_sq_MD = partial(local_sq, h=None, N=self.N, wdt=self.wdt, cs=self.cs, ms=self.ms, dump=True, correction_grid=correct_MD)
        sq_array_MD = self.pool.map(compute_sq_MD, [an_md_pos[:, :, self.freq * i] for i in range(self.ins)])
        sq_array_MD = np.array(sq_array_MD)
        sq_md = sq_array_MD.mean(0)
        if self.save_data:
            np.savetxt(self.save_dir+'/MD_SQ.txt', np.stack([q_md[1:], sq_md[1:]], axis=1))
        if self.plot_results:
            plt.ylim(0, 3.5)
            plt.xlim(0, 3.5)
            plt.plot(q_md[1:], sq_md[1:], 'b:', label='s(q)')
            plt.xlabel(r'q')
            plt.ylabel(r's(q)')
            plt.legend()
            plt.savefig(self.save_dir+'/MD_s_q.png')
            plt.clf()

    def cal_MSD(self):
        self.load_LAMMPS()
        msd_md = np.zeros(self.ins)
        u_pos_md = np.zeros((self.N_atoms, 3, self.steps))
        u_pos_md[:, :, 0] = self.md_pos[:self.N_atoms, :].copy()
        u_pos_md = unwrap_trajectories(u_pos_md.copy(), self.md_pos, self.h)
        for i in range(1, self.ins):
            diff_arr_md = u_pos_md[:, :, 0] - u_pos_md[:, :, self.freq * i]
            msd_md[i] = np.mean(np.linalg.norm(diff_arr_md, axis=1) ** 2)

        slope_md, _ = curve_fit(objective, self.t, msd_md)
        conv_un = self.nd ** 2 * 1e4
        if self.save_data:
            np.savetxt(self.save_dir+'/MSD_values.txt', np.stack([self.t, msd_md], axis=1))
        if self.plot_results:
            plt.plot(self.t, msd_md, 'r', label='MD')
            plt.plot(self.t, objective(self.t, slope_md), 'r--', label=r'D$_{MD}$=%.2f $\mu$m$^2$/s' % (slope_md * conv_un / 6))
            plt.legend()
            plt.xlabel(r'Time (ps)')
            plt.ylabel(r'MSD (LJ)')
            plt.savefig(self.save_dir+'/MSD_plots.png')
            plt.clf()

    def cal_XPCS(self):
        self.init_parallel()
        testing = XPCS_Suite(filename=self.dump_filename, atoms_add=self.atoms_add)
        testing.ext = 'MD'
        testing.load_and_process_trajectory()
        testing.nondim_d = self.nd * 1e-10
        testing.dir_name = self.save_dir +'/'
        store = []
        for q in self.q_vals:
            temp = deepcopy(testing)
            temp.pool = self.pool
            temp.q_val = q
            temp.store_gamma_workflow()
            if self.save_data:
                temp.save_correlations()
            if self.plot_results:
                plt.plot(temp.times_ps, temp.g2_exp_fit_mean, lw =2)
                plt.xlabel(r'$t$ (ps)')
                plt.ylabel(r'$(g_2(q,t)-1)/\beta(q)$')
                plt.savefig(self.save_dir + '/g2_decay_%.2f.png'%q)
                plt.clf()
            store.append([temp.gamma_ns, q ** 2, q])

        store = np.array(store)
        if self.save_data:
            np.savetxt(self.save_dir+'/Gamma_MD.txt', store)
        if len(self.q_vals)>1:
            popt, _ = curve_fit(objective_lin, store[:, 1], store[:, 0])
            a, b = popt
            diff = a * temp.nondim_d ** 2 * 1e21
            if self.plot_results:
                plt.plot(store[:,1]/self.nd**2,store[:,0],lw=3)
                plt.xlabel(r'$q^2$ (Angstrom$^{-2}$)')
                plt.ylabel(r'$\Gamma(q^2)$ (ns$^{-1}$)')
                plt.savefig(self.save_dir + '/Gamma_plots.png')
                plt.clf()
            print(r'The diffusivity is %.2f um.m/s' % diff)

    def cal_XSVS(self):
        self.init_parallel()
        testing = XPCS_Suite(filename=self.dump_filename, atoms_add=self.atoms_add)
        testing.ext = 'MD'
        testing.load_and_process_trajectory()
        testing.nondim_d = self.nd * 1e-10
        testing.dir_name = self.save_dir +'/'
        for q in self.q_vals:
            temp = deepcopy(testing)
            temp.pool = self.pool
            temp.q_val = q
            temp.get_intensity_workflow()
            temp.contrast_computation()
            if self.save_data:
                np.savetxt(self.save_dir+'/contrast_at_%.2f.txt'%q, temp.contrast_data)
            if self.plot_results:
                plt.plot(temp.contrast_data[:,0], temp.contrast_data[:,1], lw =2)
                plt.xlabel(r'$t$ (ps)')
                plt.ylabel(r'$\beta(q)$')
                plt.savefig(self.save_dir + '/contrast_at_%.2f.png'%q)
                plt.clf()

    def check_consistency(self):
        self.ref_dir = '../reference'
        ref_files = ['/Auto_corr_1.57_45_MD.txt',
                     '/Gamma_MD.txt',
                     '/MD_GR.txt',
                     '/MSD_values.txt']
        for ref_file in ref_files:
            base_file = self.save_dir + ref_file
            reference_file = self.ref_dir + ref_file
            if os.path.exists(base_file):
                base_val = np.loadtxt(base_file)
                ref_val = np.loadtxt(reference_file)
                if np.allclose(base_val,ref_val):
                    print('TEST: ' + bcolors.GRN + 'PASSED' + bcolors.RESET)
                else:
                    print('TEST: ' + bcolors.RED + 'FAILED' + bcolors.RESET)



