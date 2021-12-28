#!/usr/bin/python3
import os
import time
#from concurrent.futures import ThreadPoolExecutor  as Executor
from concurrent.futures import ProcessPoolExecutor as Executor
import numpy as np
from numpy.fft import fft,fftfreq 
import matplotlib.pyplot as plt
import sys
import shutil
import argparse


DATA_INPUT_METHOD = "foo"


class DATA_INPUT_FUNCTIONS:
    # Most simple method, file contains only one column
    def readSingleColumn(path):
        data = np.genfromtxt(path,delimiter=None)
        print(path)
        return data
    # Usually openFoam raw output method 
    def readRawFormatLastColumn(path):
        data = np.genfromtxt(path,delimiter=None)
        print(path)
        return data[:,-1]
    # Usually openFoam raw output method 
    def readRawFormatSecondToLastColumn(path):
        data = np.genfromtxt(path,delimiter=None)
        print(path)
        return data[:,-2]
    # Usually openFoam raw output method 
    def readRawFormatThirdToLastColumn(path):
        data = np.genfromtxt(path,delimiter=None)
        print(path)
        return data[:,-3]
    
def dataInput(path):
    return getattr(DATA_INPUT_FUNCTIONS, DATA_INPUT_METHOD)(path)



def FFT(row):
    # Creating a Hanning window
    N = len(row)
    j = np.linspace(0,N-1,N)
    w = 0.5 - 0.5*np.cos(2*np.pi*j/(N-1)) # Hamming window
    aw = 2.0 #- correction factor
    
    yf = fft(np.multiply(row,w))
    
    return aw*2.0/N * np.abs(yf[0:N//2])
    
        
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
        except:
            print("Unable to remove {}, directory not empty".format(resultsDirectory))
            sys.exit()
    else:
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
        N_BLOCKS = float(args['timeFinish'])
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
    timeFilesUnsorted =  [t for t in os.listdir(directory) if float(t) >= TSTART and float(t) <= TEND]
    timeFilesStr = sorted(timeFilesUnsorted, key=lambda x: float(x)) 

    timeFiles = [float(t) for t in timeFilesStr]
    
    if(TEND > timeFiles[-1]):
        TEND =  timeFiles[-1]
    
    N_FFT = round(2*len(timeFiles)/(N_BLOCKS+1))
    
    N = min(len(timeFiles),round(0.5*N_FFT*(N_BLOCKS+1)))
    
    timeFiles = timeFiles[0:N]
    
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
    print("   Number of samples              = {} ".format(len(timePaths)))
    print("   Min delta t                    = {} s".format(max(dts)))
    print("   Max delta t                    = {} s".format(min(dts)))
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
    with Executor() as executor:
        for r in executor.map(dataInput,timePaths):
            R.append(r)
                
    finish = time.perf_counter()
    print("===========================================================")
    print("Finished in:" + str(finish - start))
        
    DATA_MATRIX = np.vstack(R).T   

    del R # Free memory
     
    DATA_MATRIX -= DATA_MATRIX.mean(axis=1, keepdims=True) # Mean padded
   
    #**********************************************************************************
    #**********************************************************************************
    #
    #    Perform Welch, one frequency at the time
    #
    #**********************************************************************************
    #**********************************************************************************
    
    SD_LIST = [] # List containing spectral density marices
    for i in range(0,N_BLOCKS):
        CHUNK = list(DATA_MATRIX[:,i*N_FFT//2:(i+2)*N_FFT//2])
        print("Performing FFT on chunk {} of {}".format(i+1,N_BLOCKS))
        R = []
        with Executor() as executor:
            for r in executor.map(FFT,CHUNK):
                R.append(r)
        SD_LIST.append(np.vstack(R))
        
    del DATA_MATRIX # Free memory 
                    
    #**********************************************************************************
    #**********************************************************************************
    #
    #    First calculate singular values and use them to sort modes 
    #
    #**********************************************************************************
    #**********************************************************************************
    S = [] # List containing sum of all eigenvalues for each mode
    f = []
    PHI = []
    for i in range(0,len(freq)):
        Q = []
        for SD in SD_LIST:
            Q.append(SD[:,i])
        Q = np.vstack(Q).T
        print("Calculating singular values for frequency {} : {} of {}".format(freq[i],i+1,len(freq)))
        Sigma = np.linalg.svd(Q,compute_uv=False) # Only compute singular values at this stage, we want to avoid caluclating SVD for every mode there is
        S.append(np.dot(Sigma,Sigma))
        f.append(freq[i])
       
    #**********************************************************************************
    #**********************************************************************************
    #
    #    Next, perform full SVD for the most energetic 20 modes 
    #
    #**********************************************************************************
    #**********************************************************************************    
    print("Modes by contained variance:")
    print("---------------------------------------------------------")
        
    indxS = np.argsort(S) # Sorted indexes of S, from largest to smallest
    indxS = indxS[::-1]
    iii = 1
    for ii in indxS[0:20]:
        Q = []
        for SD in SD_LIST:
            Q.append(SD[:,i])
        Q = np.vstack(Q).T
        print("		Calculating mode {} on frequency {}".format(iii,freq[ii]))
        [U,Sigma,Vh] = np.linalg.svd(Q)
        
        PHI = U[:,0] # Save only the most energetic mode at the specific frequency
        
        fName = "Mode_{}:Frequency_{}".format(iii,f[ii])
        np.savetxt(os.path.join(resultsDirectory,fName), PHI)
        print("		Saving the mode {}".format(iii))
        iii+=1

        
    plt.stem(f,S)  
    plt.xlim(0,20)
    plt.savefig(os.path.join(resultsDirectory,"SpectralEnergy.png"))
    print("All done!")





    
if __name__ == "__main__":
    main()    

       


