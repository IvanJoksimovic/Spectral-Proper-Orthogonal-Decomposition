from mpi4py import MPI
import argparse
import numpy as np
import os
import time
import sys
import shutil
import argparse
import math
from tqdm import tqdm
from scipy.fft import rfft,rfftfreq,fft,fftfreq
from scipy.linalg import svd,eig
from tqdm import tqdm
import matplotlib.pyplot as plt



global nprocs
global comm
global rank

global sampleDirectories 
global fieldName
global resultsDirectory

global DATA_INPUT_METHOD
global tStart
global tEnd
global nBlocks
global nMeasurements

global nFFT 
global windowingFunction
global binWidth

global maxModes 
global dt
global N_FFT
global availableWindowingFunctions

global dt_max
global dt_min
global dt_avg
global fs_avg
global dt 
global fs

global timePaths

global N_PER_BLOCK

maxModes = 60

global logFile




availableWindowingFunctions = {'Hamming':1.85,'Hanning':2.0,'Blackman':2.80}   

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

    #global rank 

    if rank ==0:

        print(string)
        #f = open(logFile,'a')
        #f.write(string)
        #f.write('\n')
        #f.close()



class DATA_INPUT_FUNCTIONS:

    def readScalar(path,returnOnlyCoordinates = False):
        data = np.genfromtxt(path,delimiter=None,skip_header=1)
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

    def readVector_DrivAer_FrontLeft_Tyre(path,returnOnlyCoordinates = False):
        data = np.genfromtxt(path,delimiter=None,skip_header=2)
        x = data[:,0]
        y = data[:,1]
        z = data[:,2]

        indXYZ = np.where( (x<2) & (x>-0.5) & (y<-0) & (y>-1.5) & (z<0.45) )
        indXYZ1 = np.where( (x-0.012033614)**2 + (z+0.002480046)**2   > 0.270966386**2 )   
        indXYZ = np.intersect1d(indXYZ,indXYZ1)

        if not returnOnlyCoordinates:
            data = data[indXYZ,3:6]
            return data.flatten('F')
        else:
            return data[indXYZ,0:3]

    def readVector_DrivAer_FrontLeft_Tyre_FullDomain(path,returnOnlyCoordinates = False):
        data = np.genfromtxt(path,delimiter=None,skip_header=2)
        x = data[:,0]
        y = data[:,1]
        z = data[:,2]

        indXYZ = np.where( (x<2) & (x>-0.5) & (y<-0) & (y>-1.5) & (z<0.45) )
        #indXYZ1 = np.where( (x-0.012033614)**2 + (z+0.002480046)**2   > 0.270966386**2 )   
        #indXYZ = np.intersect1d(indXYZ,indXYZ1)

        if not returnOnlyCoordinates:
            data = data[indXYZ,3:6]
            return data.flatten('F')
        else:
            return data[indXYZ,0:3]
    def readVectorMagnitude_DrivAer_FrontLeft_Tyre(path,returnOnlyCoordinates = False):
        data = np.genfromtxt(path,delimiter=None,skip_header=2)
        x = data[:,0]
        y = data[:,1]
        z = data[:,2]

        indXYZ = np.where( (x<2) & (x>-0.5) & (y<-0) & (y>-1.5) & (z<0.45) )
        indXYZ1 = np.where( (x-0.012033614)**2 + (z+0.002480046)**2   > 0.270966386**2 )   
        indXYZ = np.intersect1d(indXYZ,indXYZ1)

        if not returnOnlyCoordinates:
            Ux = data[indXYZ,3]     
            Uy = data[indXYZ,4] 
            Uz = data[indXYZ,5]
            Umag = np.sqrt(Ux**2 + Uy**2 + Uz**2)   

            return Umag.flatten('F')
        else:
            return data[indXYZ,0:3]

    def readVector_DrivAer_Mirror(path,returnOnlyCoordinates = False):
        data = np.genfromtxt(path,delimiter=None,skip_header=2)
        x = data[:,0]
        y = data[:,1]
        z = data[:,2]

        indXYZ = np.where( (x<2) & (x> 0.5) & (y<-0) & (y>-1.5) & (z>0.45) & (z<1.05) )
        indXYZ1 = np.where( (x-0.012033614)**2 + (z+0.002480046)**2   > 0.270966386**2 )   
        indXYZ = np.intersect1d(indXYZ,indXYZ1)

        if not returnOnlyCoordinates:
            data = data[indXYZ,3:6]
            return data.flatten('F')
        else:
            return data[indXYZ,0:3]

    def readVector_DrivAer_xPlane(path,returnOnlyCoordinates = False):
        data = np.genfromtxt(path,delimiter=None,skip_header=2)
        x = data[:,0]
        y = data[:,1]
        z = data[:,2]

        indXYZ = np.where((y > -1.2) & (y <1.2) & (z<1.0))
  

        if not returnOnlyCoordinates:
            data = data[indXYZ,3:6]
            return data.flatten('F')
        else:
            return data[indXYZ,0:3]



def dataInput(path):
    return getattr(DATA_INPUT_FUNCTIONS, DATA_INPUT_METHOD)(path)


def fftChunkWindowed(chunk):

    global dt 
    global nFFT
    global windowingFunction 


    M,N,B = chunk.shape

    amplitudeCorrectionFactor = availableWindowingFunctions[windowingFunction]

    #print(f"On processor {rank}, window = {window}, with the amplitudeCorrectionFactor = {amplitudeCorrectionFactor}")

    if windowingFunction == 'Hamming':
        W = np.hamming(N) # Hanning window

    elif windowingFunction == 'Hanning':
        W = np.hanning(N) # Hanning window

    elif windowingFunction == 'Blackman':
        W = np.blackman(N) # Hanning window

    # Scaling coefficient for the amplitude due to the rfft:
    if N%2 ==0:
        nScale = (N/2)+1
    else:
        nScale = (N+1)/2

    W = np.stack([W for i in range(M)],axis = 0)
    W = np.stack([W for i in range(B)],axis = 2)

    chunk_fft = rfft(W*chunk,axis = 1,n = nFFT*N)

    freqs = rfftfreq(nFFT*N,d = dt)

    chunk_amplitude = amplitudeCorrectionFactor*chunk_fft/nScale

    return freqs, chunk_amplitude

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False  

def is_numeric_dir(directory_path, item):

    item_path = os.path.join(directory_path, item)
    return os.path.isdir(item_path) and (item.isdigit() or is_float(item))

def list_numeric_directories(directory_path):

    numeric_dirs = [item for item in os.listdir(directory_path) if is_numeric_dir(directory_path, item)]
    return numeric_dirs

def parseInputArguments():

    global nprocs
    global comm
    global rank

    global sampleDirectories 
    global fieldName
    global resultsDirectory
    
    global DATA_INPUT_METHOD
    global tStart
    global tEnd
    global nBlocks
    global nMeasurements

    global nFFT 
    global windowingFunction
    global binWidth

    global dt_max
    global dt_min
    global dt_avg
    global fs_avg
    global dt 
    global fs

    Info("Parsing data ...")

    if(rank ==0):
        
        # Create input arguments:
        
        ap = argparse.ArgumentParser()
        
        ap.add_argument("-d", "--sampleDirectories",nargs="+", required=True,help="String with a path to directoris containing time directories")
        ap.add_argument("-f", "--fieldName", required=True,help="Exact name of the field that will be located in time directories")
        ap.add_argument("-i", "--inputMethod", required=True,help="Name of the method for data input")
        ap.add_argument("-r", "--resultsDirectory", required=True,help="Exact name of the results directory")
        
        ap.add_argument("-t0", "--timeStart", required=False,help="Time from which to start")
        ap.add_argument("-t1", "--timeFinish", required=False,help="Time with which to finish")
        ap.add_argument("-n", "--nBlocks", required=False,help="Number of blocks per sample for Welch transformation")
        ap.add_argument("-nfft", "--N_FFT", required=False,help="Multiplicator of the time-dimension for zero-padding")        
        ap.add_argument("-w", "--window", required=False,help="Windowing function, default = Hamming")        
        ap.add_argument("-b", "--binWidth", required=False,help="Width of the bin, in which the modes with maximum eigenvalues will be exported")   

        args = vars(ap.parse_args())
        
        # Parse input arguments

        sampleDirectories = args['sampleDirectories']

        fieldName             = str(args['fieldName'])
        
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
            print("SPOD ERROR: " + DATA_INPUT_METHOD + " is not among data input functions:")
            [print(d) for d in dir(DATA_INPUT_FUNCTIONS) if d.startswith('__') is False]
            print("SPOD ERROR: Change name or modify DATA_INPUT_FUNCTIONS class at the start of the code")
            comm.Abort()
                                                
        try:
            tStart = float(args['timeStart'])
        except:
            tStart = 0

        try:
            tEnd = float(args['timeFinish'])
        except:
            tEnd = 1e80

        try:
            nBlocks = int(args['nBlocks'])
        except:
            nBlocks= 1

        try:
            nFFT = int(args['N_FFT'])
        except:
            nFFT = 1

        try:
            windowingFunction = args['window']
        except:
            pass

        if windowingFunction == None: windowingFunction = 'Hanning'

        try:
            binWidth = float(args['binWidth'])
        except:
            binWidth = 5.

        #print(f"On processor {rank}, WINDOW = {WINDOW}")

        if windowingFunction not in list(availableWindowingFunctions.keys()):
            print(f"SPOD ERROR: {windowingFunction} is not in the list of available windowing functions: {list(availableWindowingFunctions.keys())}")
            comm.Abort()
        else:
            pass

        nMeasurements = len(sampleDirectories)

    Info("Done.")

def evaluateSourceData():

    global nprocs
    global comm
    global rank

    global sampleDirectories 
    global fieldName
    global resultsDirectory

    global DATA_INPUT_METHOD
    global tStart
    global tEnd
    global nBlocks
    global nMeasurements

    global nFFT 
    global windowingFunction
    global binWidth

    global dt_max
    global dt_min
    global dt_avg
    global fs_avg
    global dt 
    global fs

    global timePaths

    global N_PER_BLOCK

    Info("Evaluating source data ...")

    if rank ==0:

        timePaths = []

        dt_max = [] 
        dt_min = [] 
        dt_avg = []
        fs_avg = []
        length_timePaths = []

        for directory in sampleDirectories:


            time_directories = list_numeric_directories(directory)

            timeFilesUnsorted =  set([t for t in time_directories if (float(t) >= 3.097263122397397) ])

            #timeFilesUnsorted =  set([t for t in time_directories])
        
        
            timeFilesStr = sorted(timeFilesUnsorted, key=lambda x: float(x))
            
            timeFiles = [float(t) for t in timeFilesStr]
          
            if(tEnd > timeFiles[-1]):
                tEnd =  timeFiles[-1]
        
            N_PER_BLOCK = round(math.floor(2*len(timeFiles)/(nBlocks+1)))
            
            N = min(len(timeFiles),round(0.5*N_PER_BLOCK *(nBlocks+1)))

            timeFiles = timeFiles[0:N]
            timeFilesStr = timeFilesStr[0:N]
            
            TIME = timeFiles
            timePaths.append([os.path.join(directory,str(t),fieldName) for t in timeFilesStr])

            length_timePaths.append(N)

            # At this point, prompt user 
            dts = np.diff(TIME)
            dt_max.append(np.max(dts))
            dt_min.append(np.max(dts))
            dt_avg.append(np.mean(dts))
                
            fs_avg.append(1.0/dt_avg[-1])

        # Perform the checks to see if data-sets are of the same length and if the sampling frequency is the same 

        terminateSPOD = False

        if not all([l == length_timePaths[0] for l in length_timePaths[1:]]):
            Info(f"SPOD ERROR: time lists are not of the same length, lengths are {length_timePaths}")
            terminateSPOD = True
        elif not all([round(dt,6) == round(dt_avg[0],6) for dt in dt_avg ]):
            Info(f"SPOD ERROR: time steps in samples are different, they are {dt_avg}")     
            terminateSPOD = True

        if terminateSPOD:
            comm.Abort()

        dt = np.mean(dt_avg)
        fs = 1./dt

        if binWidth < fs/(nFFT*N_PER_BLOCK):
            binWidth = fs/(nFFT*N_PER_BLOCK)

        Info(f"Done. Time steps in {nMeasurements} samples are the same up to the 6th decimal place")

def main():


    global nprocs
    global comm
    global rank

    global sampleDirectories 
    global fieldName
    global resultsDirectory

    global DATA_INPUT_METHOD
    global tStart
    global tEnd
    global nBlocks
    global nMeasurements

    global nFFT 
    global windowingFunction
    global binWidth

    global maxModes 
    global dt
    global N_FFT
    global availableWindowingFunctions

    global dt_max
    global dt_min
    global dt_avg
    global fs_avg
    global dt 
    global fs

    global timePaths

    global logFile

    global N_PER_BLOCK

    

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

    parseInputArguments() # First, parse the input arguments

    evaluateSourceData() # Then, evaluate source data

    if rank ==0 :

        confidence = 0.95

        lb,ub = calc_confidenceBounds(nBlocks*nMeasurements,confidence)

        Info("SPECTRAL POD data:")
        Info("------------------------------------------------------------------")

        #Info( "   Start time                                 = {} s ".format(TIME[0]))
        #Info( "   End time                                   = {} s ".format(TIME[-1]))
        Info( "   Number of measurements                     = {}   ".format(nMeasurements))
        Info( "   Number of samples per measurement          = {}   ".format(len(timePaths[0])))
        Info( "   Number of blocks per measurement           = {}   ".format(nBlocks))
        Info( "   Number of points per block                 = {}   ".format(N_PER_BLOCK ))
        Info( "   Relative increase with zero-padding        = {}   ".format(nFFT)) 
        #Info( "   Min delta t per sample measurement         = {} s ".format(dt_min))
        #Info( "   Max delta t per sample measurement         = {} s ".format(dt_max))
        #Info( "   Averaged delta t per sample measurement    = {} s ".format(dt_avg))
        #Info( "   Sampling frequency per sample measurement  = {} s ".format(fs_avg))
        Info( "   Applied   delta t                          = {} s ".format(dt))
        Info( "   Applied sampling frequency                 = {} Hz".format(fs))
        Info( "   Nyquist frequency                          = {} Hz".format(fs/2.0))
        Info( "   Frequency resolution                       = {} Hz".format(fs/N_PER_BLOCK )) 
        Info( "   Frequency resolution with zero-padding     = {} Hz".format(fs/(nFFT*N_PER_BLOCK) )) 
        Info( "   Applied windowing function                 = {}   ".format(windowingFunction)) 
        Info( "   Input method                               = {}   ".format(DATA_INPUT_METHOD)) 
        Info( "   Results directory                          = {}   ".format(resultsDirectory))
        Info( "   Bin width for exporting the largest modes  = {}   ".format(binWidth))
        Info( "   Number of processes                        = {}   ".format(nprocs))
        Info( "   Lower relative confidence bound ({}%)      = {}   ".format(int(100*confidence),lb))
        Info( "   Upper relative confidence bound ({}%)      = {}   ".format(int(100*confidence),ub))
  
        Info("------------------------------------------------------------------")

        #answer = input("If satisfied with frequency resolution, continue y/n?  ")
        answer = "y"    
        if( answer not in ["y","Y","yes","Yes","z","Z"]):
            print("OK, exiting calculation")
            comm.Abort()

        start = time.perf_counter()

        split_file_list = [] 
        for paths in timePaths:
            split_file_list.append(np.array_split(paths, size))

    else:
        split_file_list = None
        nMeasurements = None
        DATA_INPUT_METHOD = None
        N_PER_BLOCK = None
        nFFT = None
        nBlocks = None
        windowingFunction = None
        binWidth = None 
        resultsDirectory = None
        dt = None
        

    # Broadcast variables 
 
    split_file_list = comm.bcast(split_file_list,root = 0)
    nMeasurements = comm.bcast(nMeasurements,root = 0)
    DATA_INPUT_METHOD = comm.bcast(DATA_INPUT_METHOD, root=0)

    N_PER_BLOCK  = comm.bcast(N_PER_BLOCK , root=0)
    nFFT  = comm.bcast(nFFT , root=0)
    nBlocks = comm.bcast(nBlocks, root=0)
    windowingFunction = comm.bcast(windowingFunction, root=0)
    binWidth = comm.bcast(binWidth,root = 0)
    resultsDirectory = comm.bcast(resultsDirectory,root = 0)

    dt = comm.bcast(dt,root = 0)
    #**********************************************************************************
    #**********************************************************************************
    #
    #    Read field
    #
    #**********************************************************************************
    #**********************************************************************************

    Q = []
    Qmean = []
    Qvar = []

    for measurement in range(0,nMeasurements):

        local_files = split_file_list[measurement][rank]
        #print(f"On processor {rank}, for measurement {measurement}, number of local files is {len(local_files)}")

        Info(f"Reading the measurement {measurement+1} from {nMeasurements}")

        if rank == 0:
            Range = tqdm(local_files)
        else:
            Range = local_files

        # ***********************************************************************************************
        #
        #     Reading the data, creating column-wise chunks   
        #
        # ***********************************************************************************************
        Qm = []
        for timePath in Range:
            Qm.append(getattr(DATA_INPUT_FUNCTIONS,DATA_INPUT_METHOD)(timePath))

        Qm = np.stack(Qm,axis = 1).astype(np.float64)

        Qm = distribute2DColumnChunksToRowChunks(Qm)

        Qmm = Qm.mean(axis=1).reshape((-1,1))

        #Qm = np.stack([Qm[:,i*N_PER_BLOCK //2: (i+2)*N_PER_BLOCK //2] for i in range(nBlocks)],axis = 2)

        Q.append(Qm)

    Info("Calculating mean and variance")

    m,n = Q[0].shape
    Qmean = np.zeros((m,1))
    Qvar = np.zeros((m,1))

    if rank == 0:
        Range = tqdm(range(0,m))
    else:
        Range = range(0,m)

    for i in Range:
        values_at_point_m = np.concatenate([list(q[i,:]) for q in Q])
        Qmean[i] = np.mean(values_at_point_m)
        Qvar[i] = np.var(values_at_point_m)        
        #if rank == 0: print(f"shape of values_at_point_m = {values_at_point_m.shape}")

    Info("Done. Re-chunking the data matrix")
    if rank == 0:
        Range = tqdm(range(0,len(Q)))
    else:
        Range = range(0,len(Q))
    for i in Range:
        Qm = Q[i]
        Q[i] = np.stack([Qm[:,i*N_PER_BLOCK //2: (i+2)*N_PER_BLOCK //2] for i in range(nBlocks)],axis = 2)



    # Check if the measurements can be concatenated
    terminateSPOD = False
    try:
        Q = np.concatenate(Q,-1)
    except:
        print(f"On procesor {rank}, unable to concatenate chunks, sizes are {[q.shape for q in Q]}")
        terminateSPOD = True
    if terminateSPOD:
        comm.barrier()
        comm.Abort()
    del Qm

    #print(f"On processor {rank},shape of Q = {Q.shape}")
    chunkSizes = comm.gather(sys.getsizeof(Q),root=0)
    comm.barrier()
    if rank ==0 : 
        totalSizeInMB = sum(chunkSizes)/(1024*1024)
    else:
        totalSizeInMB = None

    Info(f"Done. Total size of the data matrix: {totalSizeInMB} MB")
    #**********************************************************************************
    #**********************************************************************************
    #
    #    Calculate mean, variance and total power
    #
    #**********************************************************************************
    #**********************************************************************************

    Info(f"Calculating mean,variance and total power...")

    # Statistics:
    # ---------------------------------
   
    #Q -= Qmean #.mean(axis=(1,2),keepdims = True)

    m1,n1 = Qmean.shape
    Q = Q - Qmean.reshape(m1,n1,1)
    #print(f"On processor {rank},shape of Qmean = {Qmean.shape}")
    # Local kinetic energy in time :
    # ---------------------------------
    Klocal = Qvar #np.sum(Q*Q, axis = (0,2)).reshape((-1,1))
    #print(f"On processor {rank}, Klocal.shape = {Klocal.shape}")
    Ktotal = np.array(comm.gather(Klocal,root = 0),dtype = object)

    # Average power:
    # ---------------------------------

    Plocal = Qvar                      

    Ptotal = np.array(comm.gather(Plocal,root = 0),dtype = object)

    Qmean = np.array(comm.gather(Qmean,root = 0),dtype = object)

    Qvar = np.array(comm.gather(Qvar,root = 0),dtype = object)    
    if rank == 0:

        # Local kinetic energy in time:

        Ptotal  = np.concatenate(Ptotal )
        mm = len(Ptotal)
        Ptotal= np.sum(Ptotal)
        Ptotal_VolumeAveraged = Ptotal/mm

        Info(f"Total power                  = {Ptotal}")
        Info(f"Total power, volume averaged = {Ptotal_VolumeAveraged}")
    
        Qmean = np.concatenate(Qmean).reshape((-1,1))
        Qvar = np.concatenate(Qvar).reshape((-1,1))

        mm,nn = Qmean.shape

        #print(f"Qmean.shape = {Qmean.shape}")

        Info("Done. Saving coordinates...")
        XYZ = getattr(DATA_INPUT_FUNCTIONS,DATA_INPUT_METHOD)(local_files[0],returnOnlyCoordinates = True)
        np.save(os.path.join(resultsDirectory,"XYZ_Coordinates"), XYZ)
    
        Info("Done. Saving Mean...")
        np.save(os.path.join(resultsDirectory,"MeanField"),Qmean)
        
        Info("Done. Saving Variance...")
        np.save(os.path.join(resultsDirectory,"VarianceField"),Qvar)

    #**********************************************************************************
    #**********************************************************************************
    #
    #     Calculate spectral-density matrix   
    #
    #**********************************************************************************
    #**********************************************************************************

    Info("Done. Creating and re-chunking the windowed spectral-density..")

    freqs,Qhat = fftChunkWindowed(Q) # This will reduce the number of dimensions, due to the Nyquist frequency
    del Q

    Qhat_r = distribute3DRowChunksToColumnChunks(Qhat.real) # This will re-distribute real part of Qhat into colmn chunks of shape: 

    Qhat_i = distribute3DRowChunksToColumnChunks(Qhat.imag) # This will re-distribute real part of Qhat into colmn chunks of shape: 

    Qhat = Qhat_r + 1j*Qhat_i

    

    del Qhat_r,Qhat_i

    m,n,k = Qhat.shape


    # ***********************************************************************************************
    #
    #     Compute SPOD modes, then eigenvalues, and then, save largest ones
    #
    # ***********************************************************************************************  

    freqs_split,freq_counts = splitListIntoChunks(freqs)

    freqs_perProc = comm.scatter(freqs_split,root = 0)

    PHI = []
    EigenValuesPerProc = []
    CoeffsPerProc = []
    FirstModeEigenValuesPerProc = []

    Info("Done. Calculating SPOD modes")
    if rank == 0:
        Range = tqdm(range(0,len(freqs_perProc)))
    else:
        Range = range(0,len(freqs_perProc))

    for j in Range: # Loop around all frequencies, whose modes are calculated on this processor
        QQ = Qhat[:,j,:]

        mm,kk = QQ.shape
        # First way, using svd decopmosition

        Phi, Sig,Vh = svd(Qhat[:,j,:], full_matrices=False) 

        eigvals = (Sig**2)

        Sig = np.matrix(np.diag(Sig))

        coeffs = np.array(np.dot(Sig,Vh)) # Coefficients to multiply modes
        ''' 
        # Second way, using method of snapshots
        C = np.dot(np.conjugate(QQ).T,QQ)

        eigvals, Psi = eig(C)

        ind = np.argsort(np.abs(eigvals))[::-1]

        eigvals = eigvals[ind]
        Psi = Psi[:,ind]

        Phi = np.dot(QQ,Psi)

        coeffs = np.dot(np.conjugate(Phi).T,QQ)
        '''

        
        PHI.append(Phi)

        EigenValuesPerProc.append(np.transpose(np.array(eigvals)))

        CoeffsPerProc.append(coeffs)

        FirstModeEigenValuesPerProc.append(eigvals[0])


    PHI = np.stack(PHI,axis = 2)
    EigenValuesPerProc = np.vstack(EigenValuesPerProc)
    CoeffsPerProc = np.stack(CoeffsPerProc,axis = 2)
    FirstModeEigenValuesPerProc = np.array(FirstModeEigenValuesPerProc)

    

    comm.barrier()
    Info(f"Done. Calculating spectrum bins, with bin size of approx {binWidth}")

    num_bins = int(np.ceil((freqs_perProc.max() - freqs_perProc.min()) / binWidth))
    bin_edges = np.linspace(freqs_perProc.min(), freqs_perProc.max(), num_bins + 1)
    bin_indices = np.digitize(freqs_perProc, bin_edges)

    max_eig_per_bin = np.zeros(num_bins)
    max_eig_indices = np.zeros(num_bins,dtype = int)

    for i in range(1, num_bins):
        bin_mask = (bin_indices == i)
        if np.any(bin_mask):
            max_index = np.argmax(FirstModeEigenValuesPerProc[bin_mask])
            max_eig_indices[i] = np.where(bin_mask)[0][max_index]


    Info("Saving modes from the binned spectrum...")
    for max_eig_index in tqdm(max_eig_indices):
        f = freqs_perProc[max_eig_index]
        U = PHI[:,:,max_eig_index]
        m1,n1 = U.shape
        U = U.reshape(m1,n1)
        #print(f"Processor {rank}: Saving mode with the spape {m1,n1} on frequency {f}")
        np.save(os.path.join(resultsDirectory,"Mode_f_{}".format(f)),U)

    comm.barrier()
    Info("Done")

    #Info(f"Finding and saving SPOD modes with {maxModes} largest eigenvalues")
    EigenValues = comm.gather(EigenValuesPerProc,root = 0) # Gather all eigenvalues to root

    Coeffs = comm.gather(CoeffsPerProc,root = 0) # Gather all coeffs to root

    if rank ==0:

        EigenValues = np.vstack(EigenValues)/(nBlocks*nMeasurements) # EigenValues for all frequencies 

        Coeffs = np.concatenate(Coeffs,axis = 2)

        Info("Saving EigenValues...")
        np.save(os.path.join(resultsDirectory,"EigenValues"),EigenValues)
        Info("Done.")

        Info("Saving Coefficients...")
        np.save(os.path.join(resultsDirectory,"Coefficients"),Coeffs)
        Info("Done.")

        #print(f"Ptotal_VolumeAveraged = {Ptotal_VolumeAveraged}")

        PSD = 0.5*np.abs(EigenValues/Ptotal)
        
        Info("Saving PSD per frequency...")
        np.save(os.path.join(resultsDirectory,"PSD"),PSD)
        Info("Done.")

        Info("Saving all frequencies...")
        np.save(os.path.join(resultsDirectory,"Frequencies"),freqs)

        #sorted_ind = np.argsort(np.sum(EigenValues,axis = 1))[::-1]

        #sorted_ind = sorted_ind[0:maxModes] # Only first maxModes modes will be written

        #freqs_to_write = freqs[sorted_ind]
    else:
        EigenValues = None 
        sorted_ind = None 
        #freqs_to_write = None

    #EigenValues = comm.bcast(EigenValues,root = 0)
    #sorted_ind = comm.bcast(sorted_ind,root = 0)
    #freqs_to_write = comm.bcast(freqs_to_write,root = 0)



    '''
    for ind,f in enumerate(freqs_to_write):
        if f in freqs_perProc:
            j = np.where(freqs_perProc==f)[0]
            U = PHI[:,:,j]
            m1,n1,k1 = U.shape
            U = U.reshape(m1,n1)
            print(f"Processor {rank}: Saving mode {ind} on frequency {f}")
            np.save(os.path.join(resultsDirectory,"Mode_{}_f_{}".format(ind,f)),U)
    '''

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
