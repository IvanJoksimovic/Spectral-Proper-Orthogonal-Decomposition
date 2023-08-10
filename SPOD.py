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
global nprocs
global maxModes 

maxModes = 60

nprocs = 1 # Global variable

def calc_confidenceBounds(Nb,confidence = 0.99):
    from scipy.stats import chi2

    alpha = 1 -confidence

    population_below = 1-alpha/2
    L = chi2.ppf(population_below , df = 2*Nb)
    #print(f"The critical value with {df} degrees of freedom and {100*population_below}% below it is: {L:.3f}")

    population_below = alpha/2
    R = chi2.ppf(population_below , df = 2*Nb)
    #print(f"The critical value with {df} degrees of freedom and {100*population_below}% below it is: {R:.3f}")

    lb = 2*Nb/L 
    rb = 2*Nb/R

    return lb,rb

def getArgsFromCmd():
    pass

def splitEficiently(m):
        ave, res = divmod(m, nprocs)
        count = [ave + 1 if p < res else ave for p in range(nprocs)]
        count = np.array(count)
        return count

def splitListIntoChunks(L):
    m = len(L)
    ave, res = divmod(m, nprocs)
    count = [ave + 1 if p < res else ave for p in range(nprocs)]
    counts = np.array(count)

    parts = []
    s = 0
    for c in counts:
        parts.append(L[s:s+c])
        s+=c

    return parts,counts

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


def distribute3DRowChunksToColumnChunks(A):
    RECV = []

    m,n,k = A.shape

    colCount = splitEficiently(n)
    colCount = comm.bcast(colCount, root=0) # Scatter the list of files for each processor to read

    rows = comm.gather(m,root=0)
    rows = comm.bcast(rows,root=0) # How many rows are there on each processor (must be the same)

    for i in range(0,nprocs):

        if rank == i:
            sendbuf = A.transpose(1,0,2).flatten()
            count = colCount*k*rows[rank]

            #displacement: the starting index of each sub-task
            displ = [sum(count[:p]) for p in range(nprocs)]
            displ = np.array(displ)

            #print(f"On processor {rank}, sendbuf = {sendbuf}, count = {count}, displ = {displ}")   
        else:
            sendbuf = None
            count = np.zeros(nprocs, dtype=np.int64)
            displ = None       

        comm.Bcast(count, root=i)
        recvbuf = np.zeros(count[rank])

        comm.Scatterv([sendbuf, count, displ,MPI.DOUBLE], recvbuf, root=i)   

        #print(f"On processor {rank}, recvbuf = {recvbuf}, size of {len(recvbuf)}")   

        RECV.append(recvbuf.reshape(colCount[rank],-1,k).transpose(1,0,2))

    return np.vstack(RECV)
 

        




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
                f = open(logFile,'a')
                f.write(string)
                f.write('\n')
                f.close()



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

    def readVectorMagnitudeFromPowerFLOWCSV(path,returnOnlyCoordinates = False):
        data = np.genfromtxt(path,delimiter=',',skip_header=1,filling_values = 0.0)
        #print(path)
        if not returnOnlyCoordinates:
            data = data[:,3]
            #print(f"For time {path.split('/')[-2]}, shape of the data is {data.shape}")
            #print(f"Shape of the data is {data.shape}")
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
    global resultsDirectory

    global logFile
    

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
        nprocs = 1

        

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

        logFile = os.path.join(resultsDirectory,'log_SPOD')
            
                             
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
            
        freqs = fftfreq(N_FFT,dt)[0:N_FFT//2] # Only the positive side of the spectrum is considered

        freqs_split,freq_counts = splitListIntoChunks(freqs)

        #freqs_perProc = comm.scatter(parts)

        fs = 1.0/dt

        confidence = 0.95

        lb,ub = calc_confidenceBounds(N_BLOCKS)
        
        Info("SPECTRAL POD data:")
        Info("------------------------------------------------------------------")

        Info( "   Start time                                 = {} s ".format(TIME[0]))
        Info( "   End time                                   = {} s ".format(TIME[-1]))
        Info( "   Number of samples                          = {}   ".format(N))
        Info( "   Number of blocks                           = {}   ".format(N_BLOCKS))
        Info( "   Number of points per block                 = {}   ".format(N_FFT))
        Info( "   Min delta t                                = {} s ".format(min(dts)))
        Info( "   Max delta t                                = {} s ".format(max(dts)))
        Info( "   Avg delta t                                = {} s ".format(dt))
        Info( "   Sampling frequency                         = {} Hz".format(fs))
        Info( "   Nyquist frequency                          = {} Hz".format(fs/2.0))
        Info( "   Frequency resolution                       = {} Hz".format(fs/N_FFT)) 
        Info( "   Input method                               = {}   ".format(DATA_INPUT_METHOD)) 
        Info( "   Results directory                          = {}   ".format(resultsDirectory))
        Info( "   Number of processes                        = {}   ".format(nprocs))
        Info( "   Lower relative confidence bound ({}%)      = {}   ".format(int(100*confidence),lb))
        Info( "   Upper relative confidence bound ({}%)      = {}   ".format(int(100*confidence),ub))
  
        Info("------------------------------------------------------------------")
    
        #answer = input("If satisfied with frequency resolution, continue y/n?  ")
        answer = "y"    
        if( answer not in ["y","Y","yes","Yes","z","Z"]):
            print("OK, exiting calculation")
            sys.exit()
        
        start = time.perf_counter()
        Info(f'Readinng {len(timePaths)} files with {nprocs} processors')
        split_file_list = np.array_split(timePaths, size)
    else:
        split_file_list = None
        DATA_INPUT_METHOD = None
        N_FFT = None
        N_BLOCKS = None
        freqs = None
        freqs_split,freq_counts = None,None
        resultsDirectory = None


    resultsDirectory = comm.bcast(resultsDirectory,root = 0)


    

    local_files = comm.scatter(split_file_list, root=0) # Scatter the list of files for each processor to read
    DATA_INPUT_METHOD = comm.bcast(DATA_INPUT_METHOD, root=0)
    freqs = comm.bcast(freqs, root=0)

    N_FFT = comm.bcast(N_FFT, root=0)
    N_BLOCKS = comm.bcast(N_BLOCKS, root=0)

    freqs_perProc = comm.scatter(freqs_split,root = 0)

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

    Q = distribute2DColumnChunksToRowChunks(Q)

    chunkSizes = comm.gather(sys.getsizeof(Q),root=0)
    comm.barrier()
    if rank ==0 : 
        totalSizeInMB = sum(chunkSizes)/(1024*1024)
    else:
        totalSizeInMB = None

    Info(f"Done. Total size of the data matrix: {totalSizeInMB} MB")

    # Calculate and subtract mean, calculate variance:
    # ***********************************************
    Info(f"Calculating mean and variance, subtracting mean...")


    Qmean = Q.mean(axis=1).reshape((-1,1))
    Qvar = np.var(Q,axis = 1).reshape((-1,1))
    Q -= Q.mean(axis=1,keepdims = True)

    Qmean = np.array( comm.gather(Qmean,root = 0),dtype = object)
    Qvar = np.array( comm.gather(Qvar,root = 0),dtype = object )
    
    if rank == 0 : 
    
        Qmean = np.concatenate(Qmean)
        Qvar = np.concatenate(Qvar)

        Info("Done. Saving coordinates...")
        XYZ = getattr(DATA_INPUT_FUNCTIONS,DATA_INPUT_METHOD)(local_files[0],returnOnlyCoordinates = True)
        np.save(os.path.join(resultsDirectory,"XYZ_Coordinates"), XYZ)
    
        Info("Done. Saving Mean...")
        np.save(os.path.join(resultsDirectory,"MeanField"),Qmean)
        
        Info("Done. Saving Variance...")
        np.save(os.path.join(resultsDirectory,"VarianceField"),Qvar)


    # ***********************************************************************************************
    #
    #     Calculate spectral-density matrix   
    #
    # ***********************************************************************************************

    Info("Done. Creating and re-chunking the windowed spectral-density..")

    Qhat = fftChunkWindowed(np.stack([Q[:,i*N_FFT//2: (i+2)*N_FFT//2] for i in range(N_BLOCKS)],axis = 2)) # This will reduce the number of dimensions, due to the Nyquist frequency
    del Q

    Qhat_r = distribute3DRowChunksToColumnChunks(Qhat[:,0:N_FFT//2,:].real) # This will re-distribute real part of Qhat into colmn chunks of shape: 

    Qhat_i = distribute3DRowChunksToColumnChunks(Qhat[:,0:N_FFT//2,:].imag) # This will re-distribute real part of Qhat into colmn chunks of shape: 

    Qhat = Qhat_r + 1j*Qhat_i

    del Qhat_r,Qhat_i

    m,n,k = Qhat.shape


    # ***********************************************************************************************
    #
    #     Compute SPOD modes, then eigenvalues, and then, save largest ones
    #
    # ***********************************************************************************************  


    PHI = []
    EigenValuesPerProc = []


    Info("Done. Calculating SPOD modes")
    for j in range(0,len(freqs_perProc)): # Loop around all frequencies, whose modes are calculated on this processor
        Phi, Sig,Vh = svd(Qhat[:,j,:], full_matrices=False) 
        PHI.append(Phi)
        EigenValuesPerProc.append(np.transpose(np.array(Sig)**2))

    PHI = np.stack(PHI,axis = 2)
    EigenValuesPerProc = np.vstack(EigenValuesPerProc)

    Info(f"Finding and saving SPOD modes with {maxModes} largest eigenvalues")
    EigenValues = comm.gather(EigenValuesPerProc,root = 0) # Gather all eigenvalues to root

    if rank ==0:
        EigenValues = np.vstack(EigenValues) # EigenValues for all frequencies 

        Info("Savign EigenValues...")
        np.save(os.path.join(resultsDirectory,"EigenValues"),EigenValues)
        Info("Done.")

        Info("Saving all frequencies...")
        np.save(os.path.join(resultsDirectory,"Frequencies"),freqs)

        sorted_ind = np.argsort(np.sum(EigenValues,axis = 1))[::-1]

        sorted_ind = sorted_ind[0:maxModes] # Only first maxModes modes will be written

        freqs_to_write = freqs[sorted_ind]
    else:
        EigenValues = None 
        sorted_ind = None 
        freqs_to_write = None

    EigenValues = comm.bcast(EigenValues,root = 0)
    sorted_ind = comm.bcast(sorted_ind,root = 0)
    freqs_to_write = comm.bcast(freqs_to_write,root = 0)


    for ind,f in enumerate(freqs_to_write):
        if f in freqs_perProc:
            j = np.where(freqs_perProc==f)[0]
            U = PHI[:,:,j]
            m1,n1,k1 = U.shape
            U = U.reshape(m1,n1)
            Info(f"Processor {rank}: Saving mode {ind} on frequency {f}")
            np.save(os.path.join(resultsDirectory,"Mode_{}_f_{}".format(ind,f)),U)


    # ***********************************************************************************************
    #
    #     Compute similarity matrix for the first SPOD modes
    #
    # ***********************************************************************************************      

    Info("Computing similarity matrix for the first modes on all frequencies")

    PHI1 = PHI[:,0,:]
    # Re-distribute again row-wise
    PHI1_real = distribute2DColumnChunksToRowChunks(PHI1.real)  
    PHI1_imag = distribute2DColumnChunksToRowChunks(PHI1.imag)  

    PHI1 = PHI1_real + 1j*PHI1_imag


    S = abs(np.transpose(np.conj(PHI1)).dot(PHI1))

    SimilarityMatrix = comm.gather(S,root = 0)

    if rank ==0:
        SimilarityMatrix = sum(SimilarityMatrix)
        Info(f"Processor {rank}: Saving similarity matrix for the first SPOD modes on all frequencies")
        np.save(os.path.join(resultsDirectory,"SimilarityMatrix"),SimilarityMatrix)


    Info("All done.")
        
if __name__ == "__main__":
    main()   
