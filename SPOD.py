#!/usr/bin/python3
import os
import time
from concurrent.futures import ProcessPoolExecutor as Executor
import numpy as np
from numpy.fft import fft,fftfreq 
import matplotlib.pyplot as plt
import sys
import shutil
import argparse
import math
from tqdm import tqdm
from scipy.signal import hann
from dask.array.linalg import svd


from dask import delayed,compute

from dask.distributed import Client, progress
import dask.array as da
DATA_INPUT_METHOD = "foo"


class DATA_INPUT_FUNCTIONS:
    # Most simple method, file contains only one column
    def readSingleColumn(path):
        data = np.genfromtxt(path,delimiter=None)
        #print(path)
        return data
    # Usually openFoam raw output method
    def readOpenFOAMRawFormatScalar(path):
        data = np.genfromtxt(path,delimiter=None,skip_header=2)
        #print(path)
        return data[:,-1]
    # Usually openFoam raw output method
    def readOpenFOAMRawFormatVector_ComponentZ(path):
        data = np.genfromtxt(path,delimiter=None,skip_header=2)
        #print(path)
        return data[:,-1]
    # Usually openFoam raw output method
    def readOpenFOAMRawFormatVector_ComponentY(path):
        data = np.genfromtxt(path,delimiter=None,skip_header=2)
        #print(path)
        return data[:,-2]
    # Usually openFoam raw output method
    def readOpenFOAMRawFormatVector_ComponentX(path):
        data = np.genfromtxt(path,delimiter=None,skip_header=2)
        #print(path)
        return data[:,-3]

    # Usually openFoam raw output method
    def readFirstColumn(path):
        data = np.genfromtxt(path,delimiter=None,skip_header=2)
        data = data[:,0]
        return data.flatten('F')

    def readSecondColumn(path):
        data = np.genfromtxt(path,delimiter=None,skip_header=2)
        return data.flatten('F')
 
    # Usually openFoam raw output method
    def readThirdColumn(path):
        data = np.genfromtxt(path,delimiter=None,skip_header=2)
        return data.flatten('F')

    # Usually openFoam raw output method
    def readAllThreeVectorComponents(path):
        data = np.genfromtxt(path,delimiter=None,skip_header=2)
        data = data[ind,-3:]
        return data.flatten('F')

    # Usually openFoam raw output method
    def readXZVectorComponents(path):
        data = np.genfromtxt(path,delimiter=None,skip_header=2)
        data = data[:,-3:]
        dataXZ = np.zeros((data.shape[0],2))
        dataXZ[:,0] = data[:,0]
        dataXZ[:,1] = data[:,2]

        #print(path)
        return dataXZ.flatten('F')
    
    def readXYVectorComponents(path):
        data = np.genfromtxt(path,delimiter=None,skip_header=2)
        data = data[:,-3:]
        dataXY = np.zeros((data.shape[0],2))
        dataXY[:,0] = data[:,0]
        dataXY[:,1] = data[:,1]
    
def dataInput(path):
    return getattr(DATA_INPUT_FUNCTIONS, DATA_INPUT_METHOD)(path)



def fftChunk(chunk):

    if(len(chunk.shape) == 2):

        #print("Two dimensional matrix,after transposing ",chunk.shape)
        chunk = np.expand_dims(chunk,axis = 2)
        chunk = np.swapaxes(chunk,1,2)
        chunk = np.swapaxes(chunk,0,1)

    M,N,B = chunk.shape
    #print("In fftChunk, chunk.shape = ",chunk.shape)
    yf = fft(chunk,axis = 1)
    return (1/N)*yf[0:N//2]

    
        
def main():
    
    # Create input arguments:
    
    ap = argparse.ArgumentParser()
    
    ap.add_argument("-d", "--sourceDirectory", required=True,help="String with a path to directory containing time directories")
    ap.add_argument("-f", "--sourceFileName", required=True,help="Exact name of the file that will be located in time directories")
    ap.add_argument("-i", "--inputMethod", required=True,help="Name of the method for data input")
    ap.add_argument("-r", "--resultsDirectory", required=True,help="Exact name of the results directory")
    
    ap.add_argument("-t0", "--timeStart", required=False,help="Time from which to start")
    ap.add_argument("-t1", "--timeFinish", required=False,help="Time with which to finish")
    ap.add_argument("-s", "--step", required=False,help="Sampling step")
    ap.add_argument("-n", "--NBLOCKS", required=False,help="Number of blocks to for Welch transformation")
    
    args = vars(ap.parse_args())
    
    # Parse input arguments
    
    
    directory        = str(args['sourceDirectory'])
    name             = str(args['sourceFileName'])
    
    resultsDirectory = str(args['resultsDirectory'])
    
    if(os.path.exists(resultsDirectory)):       
        try:
            os.rmdir(resultsDirectory)
            os.mkdir(resultsDirectory)
        except:
            print("Unable to remove {}, directory not empty".format(resultsDirectory))
            sys.exit()
    else:
	
        print("Creating directory: " + resultsDirectory)

        os.mkdir(resultsDirectory)
        
                         
    global DATA_INPUT_METHOD 
    DATA_INPUT_METHOD = str(args['inputMethod'])
    
    if(DATA_INPUT_METHOD not in dir(DATA_INPUT_FUNCTIONS)):
        print("ERROR: " + DATA_INPUT_METHOD + " is not among data input functions:")
        [print(d) for d in dir(DATA_INPUT_FUNCTIONS) if d.startswith('__') is False]
        print("Change name or modify DATA_INPUT_FUNCTIONS class at the start of the code")
        sys.exit()
                                            
    try:
        TSTART = float(args['timeStart'])
    except:
        TSTART = 0

    try:
        TEND = float(args['timeFinish'])
    except:
        TEND = 1e80

    try:
        N_BLOCKS = int(args['NBLOCKS'])
    except:
        N_BLOCKS = 1

    try:
        STEP = int(args['step'])
    except:
        STEP = 1

    #**********************************************************************************
    #**********************************************************************************
    #
    #    Read field
    #
    #**********************************************************************************
    #**********************************************************************************
        
    #timeFiles =  [float(t) for t in os.listdir(directory) if float(t) >= TSTART and float(t) <= TEND]
    timeFilesUnsorted =  set([t for t in os.listdir(directory) if float(t) >= TSTART and float(t) <= TEND])
		
    timeFilesStr = sorted(timeFilesUnsorted, key=lambda x: float(x))
    
    timeFilesStr = timeFilesStr[::STEP]

    timeFiles = [float(t) for t in timeFilesStr]
  
    if(TEND > timeFiles[-1]):
        TEND =  timeFiles[-1]
    
    N_FFT = round(math.floor(2*len(timeFiles)/(N_BLOCKS+1)))
    
    N = min(len(timeFiles),round(0.5*N_FFT*(N_BLOCKS+1)))

    timeFiles = timeFiles[0:N]
    timeFilesStr = timeFilesStr[0:N]
    
    TIME = timeFiles
    timePaths = [os.path.join(directory,str(t),name) for t in timeFilesStr]
    

    # At this point, prompt user 
    dts = np.diff(TIME)
    dt = np.mean(np.diff(TIME))
        
    freq = fftfreq(N_FFT,dt)
    fs = 1.0/dt
    
    print("SPECTRAL POD data:")
    print("------------------------------------------------------------------")

    print("   Start time                     = {} s".format(TIME[0]))
    print("   End time                       = {} s".format(TIME[-1]))
    print("   Number of samples              = {} ".format(N))
    print("   Number of blocks               = {} ".format(N_BLOCKS))
    print("   Number of points per block     = {} ".format(N_FFT))
    print("   Min delta t                    = {} s".format(min(dts)))
    print("   Max delta t                    = {} s".format(max(dts)))
    print("   Avg delta t                    = {} s".format(dt))
    print("   Sampling frequency             = {} Hz".format(fs))
    print("   Nyquist frequency              = {} Hz".format(fs/2.0))
    print("   Frequency resolution           = {} Hz".format(fs/N_FFT)) 
    print("   Input method                   = {}   ".format(DATA_INPUT_METHOD)) 
    print("   Results directory              = {}   ".format(resultsDirectory))
 
    print("------------------------------------------------------------------")
    
    answer = input("If satisfied with frequency resolution, continue y/n?  ")
    
    if( answer not in ["y","Y","yes","Yes","z","Z"]):
        print("OK, exiting calculation")
        sys.exit()
    
    start = time.perf_counter()

    print("Starting the dask client...")
    #client = Client(n_workers=10)
    
    client = Client(#processes=True, 
                    #threads_per_worker=10,
                    #n_workers=4, 
                    #memory_limit='2GB'
                    )

    client.restart()
    

    nProc = len(client.scheduler_info()['workers'])


    # ***********************************************************************************************
    #
    #     Reading the data    
    #
    # ***********************************************************************************************

    print("Done. Mapping the data-input to the client...")
    R = client.map(getattr(DATA_INPUT_FUNCTIONS,DATA_INPUT_METHOD),timePaths)
    print("Done. Reading {} data files with {} distributed workers ...".format(len(timePaths),nProc))
    progress(R)
    print('Done. Gathering the result...')
    R = client.gather(R)
    print('Done. Stacking the data into the snapshot matrix...')
    DATA_MATRIX = np.stack(R,axis = 1)

    m,n = DATA_MATRIX.shape
    print('Done. Mean padding the snapshot matrix with dimensions {} x {}...'.format(m,n))
    DATA_MATRIX -= DATA_MATRIX.mean(axis=1,keepdims = True)

    # ***********************************************************************************************
    #
    #     Calculate spectral-density matrix   
    #
    # ***********************************************************************************************

    print("Done. Preparing the graph for fft...")    
    
    w = np.hanning(N_FFT) # Hanning window
    W = np.stack([w for i in range(m)])

    SD = np.stack([W*DATA_MATRIX[:,i*N_FFT//2: (i+2)*N_FFT//2] for i in range(N_BLOCKS)],axis = 2)


    print("Done. Mapping the data-input to the client...")
    R = client.map(fftChunk,[SD[i,:,:] for i in range(m)])
    print("Done. Performing the fft on {} chunks {} with {} distributed workers ...".format(len(R),SD[0,:,:].shape,nProc))
    progress(R)
    print('Done. Gathering the result...')
    R = client.gather(R)
    print('Done. Stacking the data into the spectral-density matrix...')
    SD = np.vstack(R)
    del R

    # ***********************************************************************************************
    #
    #     Compute PSD    
    #
    # ***********************************************************************************************

    PSD = da.sum(da.absolute(SD),axis = (0,2))
    print("Computing PSD")
    PSD = PSD.persist()  # start computation in the background
    progress(PSD)      # watch progress
    PSD = PSD.compute()      # convert to final result when done if desired

    freq = freq[0:N_FFT//2]
    PSD = PSD[0:N_FFT//2]
    SD = SD[:,0:N_FFT//2,:]

    print("Saving PSD/frequency...")
    np.savetxt(os.path.join(resultsDirectory,"Frequencies"),freq)
    np.savetxt(os.path.join(resultsDirectory,"PSD"),PSD)

    # ***********************************************************************************************
    #
    #     Compute and save modes    
    #
    # ***********************************************************************************************   

    ind = np.argsort(-1*PSD)

    j = 1
    for i in ind[0:20]:
        U,Sig,Vh = svd(da.from_array(SD[:,i,:])) #,compute_svd=True)
        
        print("Calculating and saving spatial modes on a frequency {}".format(freq[i]))
        U = U.persist()
        progress(U)
        U = U.compute()
        np.save(os.path.join(resultsDirectory,"Mode_{}_f_{}".format(j,freq[i])),U)
        j+=1
    

        

    # Coordinates 
    print("     Saving XYZ coordinates")
    XYZ = np.genfromtxt(timePaths[0])[:,0:3]
    np.save(os.path.join(resultsDirectory,"XYZ_Coordinates"), XYZ)

    print("All done!")

    
if __name__ == "__main__":
    main()    

       
