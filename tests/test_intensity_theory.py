# --------------------------------------------------------
# test_intensity_theory.py
# by Shaswat Mohanty, shaswatm@stanford.edu
#
# Objectives
# Used to test the equivalence between first principle calculations and the proposed FFT based method
#
# Usage
# python3 test_intensity_theory.py
#
# Cite: (https://doi.org/10.1088/1361-651X/ac860c)
# --------------------------------------------------------
import sys, os, time
import matplotlib;  matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import argparse
sys.path.append(os.path.realpath('../python'))
from post_util import *


'''
Main program starts here
'''
def do_tests(case_list, plot_results=False):
	for case in case_list:
		if case == 1:
			test_default = test_util(smeared_integral = True,S_Q = True, comment = '1. Default Case')
			test_default.test_consistency(plot_results = plot_results, fig_id = 1)
		elif case == 2:
			test_offset = test_util(smeared_integral = True,S_Q = True, offset = 0.5, comment = '2. Atoms displaced in z-direction towards the Periodic boundary')
			test_offset.test_consistency(plot_results = plot_results, fig_id = 2)
		elif case == 3:
			test_sharp_density = test_util(smeared_integral = True,S_Q = True, sigma_grid = 400, comment = '3. Default Case with sharper density smear')
			test_sharp_density.test_consistency(plot_results = plot_results, fig_id = 3)
		elif case == 4:
			test_config = test_util(smeared_integral = True,S_Q = True, N_grid = 400, sigma_grid = 100, fourier_smear = 0.25, rc = 4.0, position = 'file', comment = '4. Atomic configuration read from the dumpfile')
			test_config.test_consistency(plot_results = plot_results, fig_id = 4)
		elif case == 5:
			test_config_smear = test_util(smeared_integral = True,S_Q = True, position = 'file', rc = 8.0, fourier_smear = 0.5, 
		             comment = '5. Atomic configuration read from the dumpfile with q-space smear')
			test_config_smear.test_consistency(plot_results = plot_results, fig_id = 5)
		elif case == 6:
			test_config_smear = test_util(smeared_integral = True,S_Q = True,position = 'file', rc = 30.0, fourier_smear = 1.0/6.0, 
		             comment = '6. Atomic configuration read from the dumpfile with q-space smear and sigma_hat = 6.0, rc = 20.0')
			test_config_smear.test_consistency(plot_results = plot_results, fig_id = 6)
		elif case == 7:
			test_default = test_util(smeared_integral = True,S_Q=True, sigma_grid = 50, position = 'single', rc = 20, comment = '7. Default Case - Single atom at origin')
			test_default.test_consistency(plot_results = plot_results, fig_id = 7)
		elif case == 8:
			test_single = test_util(sigma_grid =400, smeared_integral = True,fourier_smear = 0.25,I_Q = True, position ='file', comment = '8. Compute the intensity at a given snapshot along the line using both methods and saving into a text file')
			test_single.test_consistency(plot_results = plot_results, fig_id = 8)
		else:
			print(bcolors.RED+'Unknown test case'+bcolors.RESET+" ("+key+")")

	if plot_results:
		plt.show()

if __name__ == "__main__":
	parser = argparse.ArgumentParser("Testing consistency between different methods for calculating diffraction intensity")
	parser.add_argument('case_list', metavar='k', type=int, nargs='*', default=[1,2,3,4,5], help='cases to test')
	parser.add_argument('-p', '--plot_results', choices=('True','False'), default='False', help='turn on or off plotting')
	args = parser.parse_args()
	do_tests(args.case_list, plot_results=(args.plot_results=='True'))
