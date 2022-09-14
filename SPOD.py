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
        #print(path)
        return data[:,0]
    # Usually openFoam raw output method
    def readSecondColumn(path):
        data = np.genfromtxt(path,delimiter=None,skip_header=2)
        #print(path)
        return data[:,1]
    # Usually openFoam raw output method
    def readThirdColumn(path):
        data = np.genfromtxt(path,delimiter=None,skip_header=2)
        #print(path)
        return data[:,2]
    # Usually openFoam raw output method
    def readAllThreeVectorComponents(path):
        data = np.genfromtxt(path,delimiter=None,skip_header=2)
        data = data[:,-3:]
        #print(path)
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



def FFT(row):
    # Creating a Hanning window


    w = np.matrix(hann(600))

    N = len(row)
    j = np.linspace(0,N-1,N)
    #w = 0.5 - 0.5*np.cos(2*np.pi*j/(N-1)) # Hamming window
    aw = 1.0 #- correction factor
    
    #yf = np.abs(fft(np.multiply(row,w)))
    yf = fft(row)
    yf[1:N//2] *=2 # Scaling everythig between 0 and Nyquist
    
    return (aw/N) * yf[0:N//2]
    
        
def main():
    
    # Create input arguments:
    
    ap = argparse.ArgumentParser()
    
    ap.add_argument("-d", "--sourceDirectory", required=True,help="String with a path to directory containing time directories")
    ap.add_argument("-f", "--sourceFileName", required=True,help="Exact name of the file that will be located in time directories")
    ap.add_argument("-i", "--inputMethod", required=True,help="Name of the method for data input")
    ap.add_argument("-r", "--resultsDirectory", required=True,help="Exact name of the results directory")
    
    ap.add_argument("-t0", "--timeStart", required=False,help="Time from which to start")
    ap.add_argument("-t1", "--timeFinish", required=False,help="Time with which to finish")
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
    freq = freq[0:len(freq)//2]
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
    
    R = []

    print("Reading data files...")
    with Executor() as executor:
        R = list(tqdm(executor.map(dataInput, timePaths), total=len(timePaths)))
                
    finish = time.perf_counter()
    print("===========================================================")
    print("Finished in: " + str(finish - start) + "s" )
        
    DATA_MATRIX = np.vstack(R).T   

    del R # Free memory

    meanData = DATA_MATRIX.mean(axis=1,keepdims = True) 

    print("Saving mean field...")
    np.save(os.path.join(resultsDirectory,"MeanField"),meanData)

    DATA_MATRIX -= meanData # Mean padded 
			       
    #**********************************************************************************
    #**********************************************************************************
    #
    #    Perform the windowed FFT, one block at the time
    #
    #**********************************************************************************
    #**********************************************************************************
    print('DATA_MATRIX.shape = ',DATA_MATRIX.shape)
    SD = [] # List containing spectral density matrices
    for i in range(0,N_BLOCKS):
        ind1 = i*N_FFT//2
        ind2 = (i+2)*N_FFT//2 
        dd = DATA_MATRIX[:,ind1:ind2]
        print('Chunk shape = ',dd.shape)
        CHUNK = list(DATA_MATRIX[:,ind1:ind2])
        print("Performing FFT on chunk {} of {}".format(i+1,N_BLOCKS))
        R = []

        with Executor() as executor:
        	R = list(tqdm(executor.map(FFT, CHUNK), total=len(CHUNK)))
        SD.append(np.vstack(R))

    SD = np.stack(SD,axis = 2) # SD matrix has the dimensions: dimension x frequency x block
        
    del DATA_MATRIX # Free memory 
                    
    #**********************************************************************************
    #**********************************************************************************
    #
    #    Sort modes  
    #
    #**********************************************************************************
    #**********************************************************************************

    PSD = np.sum(np.sum(np.abs(SD),axis = 2),axis = 0) # list containing the PSD per frequency

    totalPower = np.sum(PSD)

    print("Saving the frequencies..")
    np.savetxt(os.path.join(resultsDirectory,"frequencies"),freq.T)

    print("Saving the PSD..")
    np.savetxt(os.path.join(resultsDirectory,"PSD"),PSD.T)

    ind = np.argsort(-1*PSD)[0:20] # First 20 according to the captured spectral power will be 

    for i in ind[0:1]:
        print("Performing the SVD on the frequency: {}, PSD: {}".format(freq[i],PSD[i])/totalPower)
        # Perform SVD on a set of dimesnion x block, at frequency

        start = time.perf_counter()
        Q,R = qr(SD[:,i,:])
        Ur,Sig,Vh = svd(R,full_matrices=False)
        U = np.dot(Q,Ur)

        finish = time.perf_counter()
        print("Finished in: {} s, saving the mode ...".format(finish - start))
        np.savetxt(os.path.join(resultsDirectory,"./SpatialMode_f:{}".format(freq[i])),U)



    # Coordinates 
    print("     Saving XYZ coordinates")
    X = np.matrix(getattr(DATA_INPUT_FUNCTIONS,'readFirstColumn')(timePaths[0]))
    Y = np.matrix(getattr(DATA_INPUT_FUNCTIONS,'readSecondColumn')(timePaths[0]))
    Z = np.matrix(getattr(DATA_INPUT_FUNCTIONS,'readThirdColumn')(timePaths[0]))

    np.savetxt(os.path.join(resultsDirectory,"XYZ_Coordinates"), np.vstack((X,Y,Z)).T)   



    
if __name__ == "__main__":
    main()    

       
