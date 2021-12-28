# Spectral-Proper-Orthogonal-Decomposition
Specral Proper Orthogonal Decomposition (SPOD) represents a technique for extracting dominant spatial coherent structures from large data-sets, whereas each coherent-structure (so called mode) is characterised by unique oscillating frequency. Interested reader is reffered to the works of Town et Al. (2017) and Schmidt and Colonius (2020) for a more detailed discussion of alghoritm. Data-set may contain either measured, or numerical data. 

SPOD.py is an open source python library, created by Ivan Joksimovic of the Technical University of Darmstadt,  for distributed Spectral Proper Orthogonal Decomposition of snapshot matrix (mxn), representing the spatio-temporal sample of an observed system. System may be corrupted by either white- or pink-noise.

Observed system contains m Degrees of Freedom and the observation is realized in n equidistant time points. As a result, div(n,2) modes appear, each mode corresponding with a single oscilating frequency. 

Code is executed in following steps:

1) Analysis of source directory, determination of spectral properties (sample frequency, resolution etc), prompting the user for further calculation
2) Distributed extraction of each spapshot and allocation of zero-padded snapshot-matrix 
3) Row-vise Welch analysis of snapshot-matrix and allocation of results in spectral-density matrix Q
4) For each frequency of Q, SVD decomposition is being performed, whereas the left singular vectors U 
5) First 20 most-energetic modes are stored



# Use
It is assumed that the system-snapshots are distributed in a classical *OpenFOAM* fashion: 

A single *source directory* contains multiple sub-directories (*time-directories*), which indicate time-instants of each snapshot. Each *time-directory* contains one or more discrete fields (scalar-, vector-, tensor- etc) which are recorded at speciffic time instant and stored in a column-like fashion. Time increments must be equal. 


Following arguments need to be provided
- Path to the system directory
- Name of the snapshot-file, containing snapshot-vector for each time-step
- Name of the method used for extraction of files
- Path to directory where the results will be stored (program will automatically create it)

Optionally, following arguments may be added
- Time instant from which the input should started (if not provided, input will begin at first time-instant in the source directory)
- Time instant at which the input should stop (if not provided, all time-instants in a source directory will be read)
- Number of overlapping frequency blocks for Welch analysis (default is 1)

Full list of arguments can be obtained by typing python3 SPOD.py -h or python3 SPOD.py --help


# Examples 

Two examples are provided:

cd VonKarmanVortexStreet
decomposePar 
mpirun -np 6 pimpleFoam -parallel

Optionally change numer of processors. Wait until the simulation finishes. Run-time post-processing creates the source directory: postProcessing/planeSample, containing time-directories. Each time-directory contains following fields: *p_plane1.raw* , *U_plane1.raw* and *vorticity_plane1.raw*. In this example, we will use vorticity (its Z-component) to calculate SPOD modes. 
In order to execute program, type: 

python3 SPOD.py -d ./VonKarmanVortexStreet/postProcessing/planeSample/ -f vorticity_plane1.raw -i readOpenFOAMRawFormatVector_ComponentZ -r VonKarman_Results -t0 3

This will: 
- read data from the source directory ./VonKarmanVortexStreet/postProcessing/planeSample/
- read the field named vorticity_plane1.raw
- use the function readOpenFOAMRawFormatVector_ComponentZ to extract data 
- create and write all results to VonKarman_Results
- read every snapshot from 3 seconds and onwards (initial transient takes some time to be blown out of the domain)

As a result, VonKarman_Results contains 20 files with column-wise corresponding modes. Name of each file also contains its frequency. Also, ordered list named: FrequencyBySingluarValues contains all frequencies and their corresponding singular values. 





