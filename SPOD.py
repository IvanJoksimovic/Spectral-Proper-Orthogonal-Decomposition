from mpi4py import MPI
import numpy as np
import argparse
import numpy as np
import os
import time
from pathlib import Path
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

from numpy.linalg import eigvalsh

from scipy.linalg import svd

from tqdm import tqdm


DATA_INPUT_METHOD = "foo"


global comm 
nprocs = 0 # Global variable

def getArgsFromCmd():
    pass

def splitEficiently(m):
        ave, res = divmod(m, nprocs)
        count = [ave + 1 if p < res else ave for p in range(nprocs)]
        count = np.array(count)
        return count

def distribute2DColumnChunksToRowChunks(A):
        RECV = []
        m,n = A.shape

        rowCcount = splitEficiently(m) # Matrix read by the first processor should have equal number of rows as every other
        rowCcount = comm.bcast(rowCcount, root=0) # Scatter the list of files for each processor to read

        cols = comm.gather(n,root=0)
        cols = comm.bcast(cols,root=0) # How many columns are there on each processor

        #if rank ==1:
        #    print(f"On processor 1, cols = {cols}")


        for i in range(0,nprocs):
                if rank == i:

                        sendbuf = A.flatten() # It will flatten A rowwise, row by row
                        count = rowCcount*n # Data size to be send to each processor

                        # displacement: the starting index of each sub-task
                        displ = [sum(count[:p]) for p in range(nprocs)]
                        displ = np.array(displ)
                else:
                        sendbuf = None
                        # initialize count on worker processes
                        count = np.zeros(nprocs, dtype=np.int64)
                        displ = None
        # broadcast count

                comm.Bcast(count, root=i)
                recvbuf = np.zeros(count[rank])
                comm.Scatterv([sendbuf, count, displ,MPI.DOUBLE], recvbuf, root=i)     
                RECV.append(recvbuf)

        return np.block([r.reshape(-1,nn) for r,nn in zip(RECV,cols)]) 

def gather2DDataToRoot(Qhati):

        m,n = Qhati.shape

        send_data = Qhati.flatten() # Will flatten the data row-wise, row by row
        send_count = np.array([len(send_data)], dtype=int)
        
        count = send_data.size
        recv_counts = comm.gather(count,root = 0)

        # Gather the data on the root process

        if rank == 0:
                recvbuf = np.zeros(sum(recv_counts))

                displ = [sum(recv_counts[:p]) for p in range(nprocs)]

                displ = np.array(displ)
        else:
            recvbuf = None
            displ = None


        comm.Gatherv(send_data, [recvbuf, recv_counts, displ, MPI.DOUBLE], root=0)

        if rank ==0:
                Qhati = recvbuf.reshape(-1,n)
        else:
                Qhati = None

        #comm.Gatherv([send_data, len(send_data), MPI.DOUBLE], [recv_data, recv_counts, None, MPI.INT], root=0)

        return np.array(Qhati)

def Info(string):
        if rank ==0:
                print(string)


class DATA_INPUT_FUNCTIONS:

    def readScalar(path,returnOnlyCoordinates = False):
        data = np.genfromtxt(path,delimiter=None,skip_header=0)
        #print(path)
        if not returnOnlyCoordinates:
            return data[:,-1]
        else:
            return data[:,0:3]

    def readAllThreeVectorComponents(path,returnOnlyCoordinates = False):
        data = np.genfromtxt(path,delimiter=None,skip_header=0)
        #print(path)
        if not returnOnlyCoordinates:
            data = data[:,3:6]
            return data.flatten('F')
        else:
            return data[:,0:3]


    
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
    return (1/N)*yf[:,0:N//2]

def fftChunkWindowed(chunk):

    M,N,B = chunk.shape

    W = np.hanning(N) # Hanning window
    W = np.stack([W for i in range(M)],axis = 0)
    W = np.stack([W for i in range(B)],axis = 2)

    #print("In fftChunk, chunk.shape = ",chunk.shape)
    yf = fft(W*chunk,axis = 1)
    return (1/N)*yf[:,0:N//2,:]


def main():


    global nprocs
    global comm
    global rank
    

    ## -------------------------------------------------------------------
    ## initialize MPI
    ## -------------------------------------------------------------------
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nprocs = size = comm.Get_size()
    except:
        comm = None
        rank = 0

        

    if(rank ==0):
        

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
                comm.Abort()
                sys.exit()
        else:
        
            print("Creating directory: " + resultsDirectory)
            os.makedirs(resultsDirectory,exist_ok=True)
            
                             
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
        #plt.hist(dts)
        #plt.show()
        #comm.Abort()
        dt = np.mean(np.diff(TIME))
            
        freq = fftfreq(N_FFT,dt)
        fs = 1.0/dt
        
        print("SPECTRAL POD data:")
        print("------------------------------------------------------------------")

        print("   Start time                     = {} s ".format(TIME[0]))
        print("   End time                       = {} s ".format(TIME[-1]))
        print("   Number of samples              = {}   ".format(N))
        print("   Number of blocks               = {}   ".format(N_BLOCKS))
        print("   Number of points per block     = {}   ".format(N_FFT))
        print("   Min delta t                    = {} s ".format(min(dts)))
        print("   Max delta t                    = {} s ".format(max(dts)))
        print("   Avg delta t                    = {} s ".format(dt))
        print("   Sampling frequency             = {} Hz".format(fs))
        print("   Nyquist frequency              = {} Hz".format(fs/2.0))
        print("   Frequency resolution           = {} Hz".format(fs/N_FFT)) 
        print("   Input method                   = {}   ".format(DATA_INPUT_METHOD)) 
        print("   Results directory              = {}   ".format(resultsDirectory))
        print("   Number of processes            = {}   ".format(nprocs))
     
        print("------------------------------------------------------------------")
    
        #answer = input("If satisfied with frequency resolution, continue y/n?  ")
        answer = "y"    
        if( answer not in ["y","Y","yes","Yes","z","Z"]):
            print("OK, exiting calculation")
            sys.exit()
        
        start = time.perf_counter()
        print(f'Readinng {len(timePaths)} files with {nprocs} processors')
        split_file_list = np.array_split(timePaths, size)
    else:
        split_file_list = None
        DATA_INPUT_METHOD = None
        N_FFT = None
        N_BLOCKS = None
        freq = None

    local_files = comm.scatter(split_file_list, root=0) # Scatter the list of files for each processor to read
    DATA_INPUT_METHOD = comm.bcast(DATA_INPUT_METHOD, root=0)
    freq = comm.bcast(freq, root=0)

    N_FFT = comm.bcast(N_FFT, root=0)
    N_BLOCKS = comm.bcast(N_BLOCKS, root=0)

    # ***********************************************************************************************
    #
    #     Reading the data, creating column-wise chunks   
    #
    # ***********************************************************************************************

    Q = []

    if rank == 0:
        Range = tqdm(local_files)
    else:
        Range = local_files

    #for timePath in tqdm(local_files):
    for timePath in Range:
        Q.append(getattr(DATA_INPUT_FUNCTIONS,DATA_INPUT_METHOD)(timePath))
    Q= np.stack(Q,axis = 1).astype(np.float64)

    #print(f"For processor {rank}, chunk size before distribution is {Q.shape}")
    Q = distribute2DColumnChunksToRowChunks(Q)

    #print(f"For processor {rank}, chunk size after distribution is {Q.shape}")

    chunkSizes = comm.gather(sys.getsizeof(Q),root=0)
    comm.barrier()
    if rank ==0 : print(f"Done. Total size of the data matrix: {sum(chunkSizes)/(1024*1024)} MB")

    # Calculate and subtract mean:
    # ******************************************
    if rank ==0 : print(f"Calculating, subtracting and saving mean...")
    Qmean = Q.mean(axis=1)
    Q -= Q.mean(axis=1,keepdims = True)

    Qmean = np.array( comm.gather(Qmean,root = 0) )

    if rank == 0 : 

        np.save(os.path.join(resultsDirectory,"MeanField"),Qmean)

        print("Done. Saving coordinates...")
        XYZ = getattr(DATA_INPUT_FUNCTIONS,DATA_INPUT_METHOD)(local_files[0],returnOnlyCoordinates = True)
        np.save(os.path.join(resultsDirectory,"XYZ_Coordinates"), XYZ)

    # Send Q back to processor 0

    # ***********************************************************************************************
    #
    #     Calculate spectral-density matrix   
    #
    # ***********************************************************************************************

    if rank ==0 : print("Done. Creating and re-chunking the windowed spectral-density..")

    Qhat = fftChunkWindowed(np.stack([Q[:,i*N_FFT//2: (i+2)*N_FFT//2] for i in range(N_BLOCKS)],axis = 2)) # This will reduce the number of dimensions, due to the Nyquist frequency
    del Q
    #print(f"On processor {rank}, shape of X is {X.shape}")

    # ***********************************************************************************************
    #
    #     Compute eigenvalues   
    #
    # ***********************************************************************************************      

    if rank ==0 : print("Done. Calculating eigenvalues...")

    m,n,nb = Qhat.shape
    
    Cp = [] # Covarince matrix for all frequencies for all processors
    
    # We will compute eigenvalues for each frequency
    for j in range(0,n):
    	QQ = Qhat[:,j,:].reshape(m,nb)
    	Cj = np.transpose(np.conj(QQ)).dot(QQ) # Calculate covariance matrix for the frequency j
    	Cp.append(Cj)
    Cp = np.stack(Cp,axis = 2)

    C = None
    C = comm.gather(Cp,root = 0)

    #if rank == 0: print(f'C = {C}')

    #if rank == 0: print(f'C.shape = {C.shape}')

    
    EigenValues = []

    if rank ==0 : 
        C = sum(C) # Covariance matrix with shape Nblk x Nblk x N_FFT//2
        n1,n2,nFreq = C.shape

        for j in range(0,N_FFT//2):
            eigs = np.array( sorted(eigvalsh(C[:,:,j].reshape(n1,n2)),reverse = True) )
            EigenValues.append(eigs)

        EigenValues = np.stack(EigenValues,axis = 0)

        np.save(os.path.join(resultsDirectory,"EigenValues"),EigenValues)
        print("Done.")

        print("Saving Frequencies...")
        np.save(os.path.join(resultsDirectory,"Frequencies"),freq[0:EigenValues.shape[0]])


        '''
        colors = ['r','b','g','k']
        plt.figure()
        for kk in range(0,n2):
            plt.plot(freq[0:EigenValues.shape[0]],EigenValues[:,kk],colors[kk])


        plt.xlim(0,100)
        plt.grid()
        plt.show()
        '''

        
        BiggestEigs = EigenValues[:,0]
        indx = np.argsort(-1*BiggestEigs) 

    else:
        indx = None

    indx  = comm.bcast(indx ,root = 0)
    #if rank ==1 : print(indx)

    # ***********************************************************************************************
    #
    #     Compute SPOD modes with largest eigenvalues  
    #
    # ***********************************************************************************************     

    maxModes = 40
    
    
    if rank == 0: print(f"Saving first {maxModes} with largest eigenvalues, change the variable maxModes when needed")

    for i in range(0,maxModes):

        ind = indx[i]

        Qhati  = Qhat[:,ind,:]

        # Since the passing the complex numbers through mpi.comm is not trivial, we well assemble them separately

        QhatiReal = gather2DDataToRoot(Qhati.real)
        QhatiImag = gather2DDataToRoot(Qhati.imag)


        if rank ==0:
            Qhati = np.array(QhatiReal+1j*QhatiImag)
            #print(Qhati)

            print("Calculating and saving spatial modes on a frequency {} Hz ...".format(freq[ind]))

            t1 = time.perf_counter()
            Phi, Sig,Vh = svd(Qhati, full_matrices=False)    
            #print(f"Total shape and size of the Modeare: {Phi.shape} and {sys.getsizeof(Phi)/(1024*1024)} MB")
            np.save(os.path.join(resultsDirectory,"Mode_{}_f_{}".format(i,freq[ind])),Phi)
            t2 = time.perf_counter()
            print(f'Done in {t2-t1} sec')
           
        
if __name__ == "__main__":
    main()   
