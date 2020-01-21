import re
import numpy as np
import h5py
import os.path
import scipy.io
import scipy.signal
import matplotlib.pylab as plt
import matplotlib.mlab as mlab
from scipy import linalg as LA



def load_stateidx(ppath, name) :
    """ load the sleep state file of recording ppath/name
    @RETURN:
    M     -      sequency of sleep states
    """   
    file = ppath + '/' + name + '/'+ 'remidx_' + name + '.txt'
    
    f = open(file, 'r')    
    lines = f.readlines()
    f.close()
    
    n = 0
    for l in lines :
        if re.match('\d', l):
            n = n+1
            
    M = np.zeros(n)
    
    i = 0
    for l in lines :
        
        if re.search('^\s+$', l) :
            continue
        if re.search('\s*#', l) :
            continue
        
        if re.match('\d+\s+\d+', l) :
            a = re.split('\s+', l)
            M[i] = float(a[0])
            i = i+1   
            
    return M




def state_durations(M, dt, thr=0, istate=1):
    """
    @RETURN: list of $istate durations
    """
    seq = np.nonzero(M==istate)
    seq = seq[0]
    seq = get_sequences(seq)

    durations = []
    for s in seq :
        dur = (s[-1]-s[0]+1)*dt
        if dur >= thr :
            durations.append(dur)

    return durations
    


def state_intervals(M, dt, istate=1) :
    """state_intervals(M, dt, istate=1)...
    calculate all intervals between istate-sequences
    @RETURN: list of inter-$istate intervals
    """
    
    idx = np.nonzero(M==istate)[0]
    seq = get_sequences(idx)
    
    intvals = []
    for i in range(1,len(seq)) :
        dur = (seq[i][0] - seq[i-1][-1] - 1)*dt
        intvals.append(dur)
        
    return intvals
    

        
       
def calculate_spectrogram(sig, SR, win_length, ns=1, pnorm=0) :

    Pxx,freqs,t = mlab.specgram(sig, int(win_length), Fs=SR, noverlap=np.round(win_length/2.0))
    if ns > 1 :
        Pxx = scipy.signal.convolve2d(Pxx, np.ones((ns,ns))/(ns*ns), mode='same')

    if pnorm == 1 :
        nPxx = np.array([np.linalg.norm(Pxx[i,:]) for i in range(Pxx.shape[0])])
        Pxx = ( Pxx.T / nPxx ).T

    return Pxx, t, freqs



def load_eeg(ppath, name) :
    file = os.path.join(ppath, name, 'EEG.mat')
    #print file
    mat = scipy.io.loadmat(file)['EEG'];
  
    return np.reshape(mat*1.0, mat.shape[0])
        
        
        

        
def get_sequences(idx, ibreak=1) :  
    """
    get_sequences(idx, ibreak=1)
    idx     -    np.vector of indices
    @RETURN:
    seq     -    list of np.vector
    """
    diff = idx[1:] - idx[0:-1]
    breaks = np.nonzero(diff>ibreak)[0]
    breaks = np.append(breaks, len(idx)-1)
    
    seq = []    
    iold = 0
    for i in breaks :
        r = range(iold, i+1)
        seq.append(idx[r])
        iold = i+1
        
    return seq

         
        
def laser_start_end(laser, SR=1525.88, intval=5) :
    """laser_start_end(ppath, name) ...
    print start and end index of laser stimulation periods\
    returns the tuple (istart, iend), both indices are inclusive,\
    i.e. part of the sequence
    
    laser    -    laser, vector of 0s and 1s
    intval   -    min time interval [s] between two laser sequences
    """
    idx = np.nonzero(laser > 0.1)[0]
    if len(idx) == 0 :
        return (None, None)
    
    idx2 = np.nonzero(np.diff(idx)*(1./SR) > intval)[0]
    istart = np.hstack([idx[0], idx[idx2+1]])
    iend   = np.hstack([idx[idx2], idx[-1]])    
    
    return (istart, iend)
    

def downsample_vec(x, nbin) :
    '''
    y = downsample_vec(x, nbin)
    downsample the vector x by replacing nbin consecutive \
    bin by their mean \
    @RETURN: the downsampled vector 
    '''
    n_down = np.floor(len(x) / nbin)
    x = x[0:n_down*nbin]
    x_down = np.zeros((n_down,))

    # 0 1 2 | 3 4 5 | 6 7 8 
    for i in range(nbin) :
        idx = range(i, int(n_down*nbin), int(nbin))
        x_down += x[idx]

    return x_down / nbin


def downsample_matrix(X, nbin) :
    """
    y = downsample_matrix(X, nbin)
    downsample the matrix X by replacing nbin consecutive \
    rows by their mean \
    @RETURN: the downsampled matrix 
    """
    n_down = int(np.floor(X.shape[0] / nbin))
    X = X[0:n_down*nbin,:]
    X_down = np.zeros((n_down,X.shape[1]))

    # 0 1 2 | 3 4 5 | 6 7 8 
    for i in range(nbin) :
        idx = range(i, int(n_down*nbin), int(nbin))
        X_down += X[idx,:]

    return X_down / nbin
    

    

def firing_rate(ppath, name, grp, un, nbin) :
    """
    y = firing_rate(ppath, name, grp, un, nbin) 
    """
    file = os.path.join(ppath, name, 'Spk' + str(grp) + '.mat')
    #print file
    l = scipy.io.loadmat(file)
    train = l['S']['train'][0][un-1].toarray()[:,0]
    spikesd = downsample_vec(train, nbin)
    return spikesd

    
def power_spectrum(x, length, SR) :
    """
    f,y = power_spectrum(x, length, SR)
    @OUTPUT:
    f   -   frequencies
    y   -   power
    """    
    if len(x) <= length:
        x2 = np.zeros((length,))
        gap = length - len(x)
        d = np.round(gap/2.0)
        x2[d:d+len(x)] = x
        x = x2        
    
    f, y = scipy.signal.welch(x, SR, nperseg=length, nfft=length)
    return f,y 
    

def smooth_data(x, sig) :
    """
    y = smooth_data(x, sig)
    smooth data vector @x with gaussian kernal
    with standard deviation $sig
    """
    sig = float(sig)
    if sig == 0. :
        return x
        
    # gaussian:
    gauss = lambda (x, sig) : (1/(sig*np.sqrt(2.*np.pi)))*np.exp(-(x*x)/(2.*sig*sig))

    p = 1000000000
    L = 10.
    while (p > p) :
        L = L+10
        p = gauss((L, sig))

    F = map(lambda (x) : gauss((x, sig)), np.arange(-L, L+1.))
    F = F / np.sum(F)
    
    return scipy.signal.fftconvolve(x, F, 'same')



def spectrogram(x, SR, winlen) :
    Pxx, freqs, bins, im = mlab.specgram(x,NFFT=winlen,Fs=SR)
   

def nrem2rem_transition(Pxx, freqs, npca=2, pnorm=0) :

    # delta
    d1 = np.mean(Pxx[(freqs>0.) & (freqs<=5)], axis=0)
    # theta / delta, before freqs < 12
    n1 = np.mean(Pxx[(freqs>6) & (freqs<12)], axis=0)
    th = n1 / d1
    # spindle band, before freqs>9
    s1 = np.mean(Pxx[(freqs>12) & (freqs<20)], axis=0)
    # EMG
    m1 = np.mean(Pxx[(freqs>100) & (freqs<700)], axis=0)

    MX = np.zeros((len(d1), 4))
    MX[:,0] = d1
    MX[:,1] = th
    MX[:,2] = s1
    MX[:,3] = m1

    if pnorm == 1 :
        # normalize matrix
        norm = np.array([np.linalg.norm(MX[:,i]) for i in range(MX.shape[1])])
        MX = MX/norm

    pMX = pca(MX.copy(), npca)[0]

    return pMX


def nrem2rem_transition_wholeeeg(Pxx, npca=2, pnorm=1) :

    if pnorm == 1:
        tmp = Pxx.T / Pxx.mean(axis=1)
    else :
        tmp = Pxx.copy().T

    pMX = pca(tmp, npca)[0]
    return pMX
   
   
def pca(data,dims=2) :
    """
    @data is a 2D matrix.
    each row is an observation (a data point).
    each column is a variable (sub data point)
    """
    m, n = data.shape
    # mean center the data, i.e. calculate the average "row" and subtract it
    # from all other rows
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric, 
    # the performance gain is substantial
    evals, evecs = LA.eigh(R)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims]
    print evecs
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors

    
    return np.dot(evecs.T, data.T).T, evals, evecs



def least_squares(x, y, n) :
    A = np.zeros((len(x), n+1))
    for i in range(n+1) :
        A[:,n-i] = np.power(x, i)
    
    p = LA.lstsq(A, y)[0]

    r2=-1
    # #calculate r2 coefficient
    # S_tot = np.var(y - np.mean(y))
    # f = np.zeros( (len(x),1) )
    # for i in range(n+1) :
    #     f = f + np.power(x, n-i) * p[i]
    
    # S_res = np.var(y - f)
    # print S_res, S_tot
    # r2 = 1 - S_res / S_tot

    return p, r2

    
def bootstrap_resample(x, y, nboot=10000):
    """
    test if x and y are different
    """
    import numpy.random as rand
    b = np.zeros((nboot,))
    for i in range(nboot):
        xb = x[rand.randint(len(x), size=(len(x,)))]
        yb = y[rand.randint(len(y), size=(len(y,)))]
        
        b[i] = np.mean(xb) - np.mean(yb) 
        
    return len(np.where(b>0)[0]) / (nboot*1.0)
               
    


    
if __name__ == "__main__":
    ppath = '/Users/tortugar/Documents/Berkeley/Data/RawData'
    ppath = '/media/Backup1/Data/RawData'
    #ppath = '/home/franz/Windows/Data/RawData'
    #name = 'G23_022114n6'
    #name = '701_051013n2'
    #name = 'G34_051614n1'
    name = 'G18_111213n3'
    #name = 'G22_030114n1'
    #name = 'G6_052413n2'
    #name = 'T1_041914n1'
    #name = 'G35_042614n1'
    #name = 'G90_022815n1'


    SR = 1525.88
    M = load_stateidx(ppath, name)
    eeg = load_eeg(ppath, name)
    
    
    

    # calculate spectrogram
    #ns = 5;
    #win_length = int(round(SR)*5)
    #Pxx,freqs,t = mlab.specgram(eeg, win_length, Fs=round(SR), noverlap=win_length/2)
    #Pxx = scipy.signal.convolve2d(Pxx, np.ones((ns,ns))/(ns*ns), mode='same')
    #nPxx = np.array([np.linalg.norm(Pxx[i,:]) for i in range(Pxx.shape[0])])
    #normPxx = ( Pxx.T / nPxx ).T     
    Pxx, t, freqs = calculate_spectrogram(eeg, SR, round(SR)*5, ns=5)


    #ns = 5
    #Pxx, tmp, t, freqs = load_spectrogram(ppath, name)
    #Pxx = scipy.signal.convolve2d(Pxx, np.ones((ns,ns))/(ns*ns), mode='same')

    # calculate ratio 1
    #n1 = np.mean(Pxx[(freqs>10.5) & (freqs<20)], axis=0)
    #d1 = np.mean(Pxx[(freqs>0.5) & (freqs<=5)], axis=0)
    #r1 = d1
    
    #n2 = np.mean(Pxx[(freqs>0.5) & (freqs<4.5)], axis=0)
    #d2 = np.mean(Pxx[(freqs>6) & (freqs<10)], axis=0)
    #r2 = d2 / n2

    # delta
    d1 = np.mean(Pxx[(freqs>0.) & (freqs<=5)], axis=0)

    # theta / delta
    n1 = np.mean(Pxx[(freqs>6) & (freqs<12)], axis=0)
    th = n1 / d1

    # spindle band, before (freqs>9)
    s1 = np.mean(Pxx[(freqs>12) & (freqs<20)], axis=0)

    # EMG
    m1 = np.mean(Pxx[(freqs>200) & (freqs<700)], axis=0)

    MX = np.zeros((len(d1), 4))
    MX[:,0] = d1
    MX[:,1] = th
    MX[:,2] = s1
    MX[:,3] = m1

    # normalize matrix
    norm = np.array([np.linalg.norm(MX[:,i]) for i in range(MX.shape[1])])
    #MX = MX/norm

    pMX = pca(MX.copy(), 3)[0]
    
    r1 = pMX[:,0]
    r2 = pMX[:,1]
    
    plt.figure()
    plt.hold('on')
    M = M[1:len(t)+1]
    idx = np.where(M==2)
    #plt.plot(r2[idx], r1[idx], 'r.')
    idx = np.where(M==3)
    plt.plot(r2[idx], r1[idx], 'bo')
    idx = np.where(M==1)
    plt.plot(r2[idx], r1[idx], 'go')    
    plt.show()


    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # #ax = Axes3D(fig)
    # idx = np.where(M==1)
    # ax.scatter(pMX[idx,0], pMX[idx,1], pMX[idx,2], c='g')
    # idx = np.where(M==2)
    # ax.scatter(pMX[idx,0], pMX[idx,1], pMX[idx,2], c='r')
    # idx = np.where(M==3)
    # ax.scatter(pMX[idx,0], pMX[idx,1], pMX[idx,2], c='b')

    # plt.show()

    

    plt.figure()
    pMX = pca(MX, 1)[0]
    p = pMX[(M==1) | (M==3)]
    plt.hist(p, 100)
    plt.show()

    plt.figure()
    plt.pcolormesh(t, freqs[1:150], Pxx[1:150][:], vmin=0, vmax=5000)
    plt.ylim((0,20))
    plt.axis('off')
    plt.axis('tight')
    plt.show()
    
    
