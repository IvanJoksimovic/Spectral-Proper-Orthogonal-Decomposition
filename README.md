# Spectral-Proper-Orthogonal-Decomposition
Spectral Proper Orthogonal Decomposition (SPOD) represents a technique for extracting dominant coherent structures from large data-sets, whereas each coherent-structure (so called mode) is characterised by unique oscillating frequency. Interested reader is reffered to the works of Town et Al. (2017) and Schmidt and Colonius (2020) for a more detailed discussion of alghoritm. Data-set may contain either measured, or numerical data. 

SPOD.py is an open source python library, created by Ivan Joksimovic of the Technical University of Darmstadt,  for distributed Spectral Proper Orthogonal Decomposition of snapshot matrix (mxn), representing the spatio-temporal sample of an observed system. System may be corrupted by either white- or pink-noise.

Observed system contains m Degrees of Freedom and the observation is realized in n equidistant time points. As a result, div(n,2) modes appear, each mode corresponding with a single oscilating frequency. 

Code is executed in following steps:

1) Analysis of source directory, determination of spectral properties (sample frequency, resolution etc), prompting the user for further calculation
2) Distributed extraction of each spapshot and allocation of zero-padded snapshot-matrix 
3) Row-vise Welch analysis of snapshot-matrix and allocation of results in spectral-density matrix Q
4) For each frequency of Q, SVD decomposition is being performed, whereas the left singular vectors U (or a single singular vector if 1 Welch block is used) represent modes.
5) First 20 most-energetic modes are stored in results directory

# Use
It is assumed that the system-snapshots are distributed in a classical *OpenFOAM* fashion: 

A single *source directory* contains multiple sub-directories (*time-directories*), which indicate time-instants of each snapshot. Each *time-directory* contains one or more discrete fields (scalar-, vector-, tensor- etc) which are recorded at speciffic time instant and stored in a column-like fashion. Time increments must be equal. 


Following arguments need to be provided
- Path to the system directory
- Name of the snapshot-file, containing snapshot-vector for each time-step
- Name of the method used for reading and extraction of files (input function)
- Path to directory where the results will be stored (program will automatically create it)

Optionally, following arguments may be added
- Time instant from which the input should started (if not provided, input will begin at first time-instant in the source directory)
- Time instant at which the input should stop (if not provided, all time-instants in a source directory will be read)
- Number of overlapping frequency blocks for Welch analysis (default is 1)

Full list of arguments can be obtained by typing python3 SPOD.py -h or python3 SPOD.py --help. All input functions are stored as methods of DATA_INPUT_FUNCTIONS class and must return a single numpy.array. User may add additional input functions at own discression. Most time-consuming phase represents a data input, which is therefore performed in a distributed fashion using the concurrent.futures module. All awailable proecessors are used. User may chacnge max number of processors at own discression. 


# Examples 

Two examples are provided:

A Von-Karman vortex street behing a 2D cylinder is realized in OpenFOAM. Snapshots of velocity,pressure and vorticity are stored with a 1000Hz rate (every 0.001 s). 

First run the code by typing:

cd VonKarmanVortexStreet
decomposePar 
mpirun -np 6 pimpleFoam -parallel
cd ..

Optionally change numer of processors. Wait until the simulation finishes. Run-time post-processing creates the source directory: postProcessing/planeSample, containing time-directories. Each time-directory contains following fields: *p_plane1.raw* , *U_plane1.raw* and *vorticity_plane1.raw*. In this example, we will use pressure to calculate SPOD modes. 

In order to execute program, type for the execution with ex. 16 processes: 

mpirun -np 16 python3 -m mpi4py SPOD.py -d ./VonKarmanVortexStreet/postProcessing/planeSample/ -f vorticity_plane1.raw -i readScalar -r VonKarman_SPOD_Results -t0 3 -n 2

This will: 
- read data from the source directory ./VonKarmanVortexStreet/postProcessing/planeSample/
- read the field named p_plane1.raw
- use the function readScalar to extract data 
- create and then write all results to VonKarman_Results directory
- read every snapshot from 3 seconds of simulation time and onwards (initial transient takes some time to be blown out of the domain)
- create two overlapping blocks of which will be subjected to the svd analysis, frequency by frequency.

Results are stored in a matrix form (*.npy),readable with numpy.load. 

Folloving fields will be written-out in the results directory:
- XYZ_Coordinates.npy, contains Mx3 matrix, where M is the number of points. Each row contains x,y and z coordinate of the point. 
- MeanField.npy (values of the mean field), contains Mxk matrix, where k is 1 for scalar field, 3 for vector field
- Frequencies.py, contains the matrix Nx1, where N is the number of frequencies between 0 and Nyquist frequency
- EigenValues.npy, contains NxNbl matrix, where Nbl is the number of overlapping blocks. Each row corresponds to the eigenvalues of the mode at the N-th frequency

Additionally, program will automatically write the leading 30 SPOD modes, with largest eigenvalues. They are stored in the format:
Mode_{number}_f_{frequency}.npy, where number is the sorted position of the mode, and frequency is the frequency at which the Mode oscillates.






