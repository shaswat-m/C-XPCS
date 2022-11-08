# --------------------------------------------------------
# test_MSMSE.py
# by Shaswat Mohanty, shaswatm@stanford.edu
#
# Objectives
# Carry out basic scattering theory and structure based analysis (g(r), s(q), XPCS or XSVS)
#
# Usage (Check README.md for other applications)
# python3 test_MSMSE.py --msd --steps 50 ; (calculates MSD)
#
# Cite: (https://doi.org/10.1088/1361-651X/ac860c)
# --------------------------------------------------------
import os, sys
import argparse
sys.path.append(os.path.realpath('../python'))
from post_util import *

def main():
    parser = argparse.ArgumentParser(description="Computing test function analysis")
    parser.add_argument("--save_data", default=False, action="store_true", help="Flag to save data")
    parser.add_argument("--plot_results", default=False, action="store_true", help="Flag to plot figures")
    parser.add_argument("--rdf_and_sq", default=False, action="store_true", help="s(q) from data file")
    parser.add_argument("--msd", default=False, action="store_true", help="s(q) from dump file")
    parser.add_argument("--xpcs", default=False, action="store_true", help="g2(q) from dump file")
    parser.add_argument("--xsvs", default=False, action="store_true", help="beta(q) from dump file")
    parser.add_argument("--check", default=False, action="store_true", help="Check if installation has worked")
    parser.add_argument("--q_vals", nargs="+", type=float, default=[1.57, 3.14, 4.44, 5.43, 6.28], help="wave-vector magnitudes over which XPCS/XSVS analysis is to be carried")
    parser.add_argument("--dump_filename", type=str, default='../datafiles/dump.ljtest', help="dump file")
    parser.add_argument("--save_dir", type=str, default='../runs', help="data file")
    parser.add_argument("--atoms_add", type=int, default=45, help="atoms analyzed in XPCS")
    parser.add_argument("--steps", type=int, default=50, help="total steps in dump file to be read")
    parser.add_argument("--freq", type=int, default=1, help="freqeucny of calculating MSD, g(r) or s(q)")
    parser.add_argument("--poolsize", type=int, default=None, help="freqeucny of calculating MSD, g(r) or s(q)")
    parser.add_argument("--timestep", type=float, default=0.01078, help="simulation timestep in ps")
    args = parser.parse_args()
    util = post_analysis(dump_filename = args.dump_filename, save_data = args.save_data,
                         save_dir = args.save_dir, steps = args.steps, q_vals = args.q_vals,
                         atoms_add = args.atoms_add, plot_results = args.plot_results,
                         freq = args.freq, timestep = args.timestep, poolsize = args.poolsize)
    if args.rdf_and_sq:
        util.cal_rdf_and_sq()
    if args.msd:
        util.cal_MSD()
    if args.xpcs:
        util.cal_XPCS()
    if args.xsvs:
        util.cal_XSVS()
    if args.check:
        util.check_consistency()

if __name__ == '__main__':
    main()