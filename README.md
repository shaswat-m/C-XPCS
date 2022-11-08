## C-XPCS: A library for computational XPCS and XSVS
Using computational analysis to compute the XSVS and XPCS experimental data by analyzing Molecular Dynamics trajectories. 

## Cite
If you use any particular segment of the code for your XPCS analysis, then please cite the following paper:
* Computational Approaches to Model X-ray Photon Correlation Spectroscopy from Molecular Dynamics (https://doi.org/10.1088/1361-651X/ac860c)

## Installation
Use the following lines to get all the dependencies setup

```
git clone git@gitlab.com:micronano_public/c-xpcs.git ; 
cd c-xpcs ;
workdir=$(pwd) ;
python3 -m pip install -r py_requirements.txt ;
```

Get the 5000 frame LAMMPS dumpfile in case you want to test the XPCS analysis.
```
cd $workdir/datafiles ;
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=13WDenyx3IYEwZhFj6qwDmEueY5oiHVDG' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=13WDenyx3IYEwZhFj6qwDmEueY5oiHVDG" -O dump.XPCS && rm -rf /tmp/cookies.txt ;
```
 
## Tests XPCS theory
Use the following script to run the tests corresponding to the equivalence between the real-space (direct) method [Algorithm 1] and the FFT based method [Algorithm 2]. Change directory to the test directory. This should run the default set of 5 tests.
```commandline
cd $workdir/tests ;
python3 test_intensity_theory.py ;
```

## Key XPCS Functionalities
Note: If you are using a local system or a login node of  cluster to check then use `--poolside 1` or `$n_cpu`. By default the multiprocessing pool will be initialized to use all CPUs. You can run the scripts without the `--poolsize` flag on a cluster to use the default.  

Obtain the the `g(r)` and `s(q)` for a single snapshot and averaged over 50 frames using:
```commandline
cd $workdir/tests ;
python3 test_MSMSE.py --rdf_and_sq --steps 1 --plot_results --poolsize 1 ;
python3 test_MSMSE.py --rdf_and_sq --steps 50 --plot_results --poolsize 1 --save_data ;
```
Obtain the mean-squared displacement using:
```commandline
python3 test_MSMSE.py --msd --plot_results --timestep 21.56 --save_data ;
```

Consider using the following scripts on a cluster without the `--poolsize` flag to use all CPUs available on the compute node.

The XPCS analysis can be carried out using:
```commandline 
python3 test_MSMSE.py --xpcs --steps 5000 --dump_filename '../datafiles/dump.XPCS' --plot_results --save_data --poolsize 1 ;
```
The XSVS analysis can be carried out using (computation of the optical contrast for different X-ray pulse widths):
```commandline 
python3 test_MSMSE.py --xsvs --steps 5000 --dump_filename '../datafiles/dump.XPCS' --plot_results --save_data --poolsize 1 ;
```

If you run the above scripts just as they are then you should have text files generated in the `$workdir/runs` directory which you can compare against the reference results in `$workdir/reference` to ensure the installation is done correctly. This can be done by running the following script:
```commandline
python3 test_MSMSE.py --check
```
## Publications 
* Computational Approaches to Model X-ray Photon Correlation Spectroscopy from Molecular Dynamics (https://doi.org/10.48550/arXiv.2204.13241)
* Computational Approaches to Model X-ray Photon Correlation Spectroscopy from Molecular Dynamics (https://doi.org/10.1088/1361-651X/ac860c)

## Conference presentations
* Computational X-ray Photon Correlation Spectroscopy from Molecular Dynamics Trajectories (MRS Spring 2022, Hawaii - Poster)

## Support and Development

For any support regarding the implementation of the source code, contact the developers at: 
* Shaswat Mohanty (shaswatm@stanford.edu)
* Wei Cai (caiwei@stanford.edu)


## Contributing
The development is actively ongoing and the sole contributors are from Shaswat Mohanty and Wei Cai.  Request or suggestions for implementing additional functionalities to the library can be made to the developers directly.

## Project status
Development is currently ongoing and is intended to be the dissertation project of Shaswat Mohanty.
