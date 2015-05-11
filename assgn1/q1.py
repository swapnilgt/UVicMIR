import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wv
import random
from scipy.fftpack import fft
import pickle
import timeit as ti


def sampleBased(N = 100, S = 100, A = 1.0, phi = 10):
    jump = np.pi * 2 / N
    phiAngle = np.pi * 2 * phi / N
    result = np.zeros(S)
    for i in range(S):
        result[i] = A * np.cos(i * jump + phiAngle)
    #plt.plot(result)
    return result



'''
Creates a mix of sinusoids for 0.01 seconds
Default f = 440 Hz
        fs = 44100 Hz
        t = 2.0 sec
'''
def mixSinusoids(f = 440, randPhase = False, writeFile = False, fs = 44100, t = 2.0):
    ampF = 1.0
    amp2F = 0.5
    amp3F = 0.33

    numSamples = int(fs * t)
    
    phase = 0

    sampF = fs / f
    samp2F = fs / (2 * f)
    samp3F = fs / (3 * f)

    if randPhase == True:
        phase = random.randrange(0, sampF) + 1
        print 'The random phase chosen for f is:' + str(phase)

    sigF = sampleBased(sampF, numSamples, ampF, phase)
    
    if randPhase == True:
        phase = random.randrange(0, samp2F) + 1
        print 'The random phase chosen for 2f is:' + str(phase)
    
    sig2F = sampleBased(samp2F, numSamples, amp2F, phase)
    
    if randPhase == True:
        phase = random.randrange(0, samp3F) + 1
        print 'The random phase chosen for 3f is:' + str(phase)
    
    sig3F = sampleBased(samp3F, numSamples, amp3F, phase)

    sumSig = sigF + sig2F + sig3F
    sumSig = sumSig/ np.max(sumSig)

    if writeFile:
        fileName = 'temp.wav'
        wv.write(fileName, fs, sumSig)

    return [sumSig, sigF, sig3F]

def innerProduct(f = 440, fs = 44100):

    sumSig = mixSinusoids(f = f , randPhase = True, fs = fs)[0]

    sampF = fs / f
    samp2F = fs / (2 * f)
    samp3F = fs / (3 * f)

    corrF = np.zeros(sampF)
    for i in range(sampF):
        tempSignal = sampleBased(sampF, 44100 * 2, 1.0, i)
        corrF[i] = np.dot(tempSignal, sumSig)

    print np.pi * np.argmax(corrF) / sampF
    
    plt.plot(corrF)
    
    corr2F = np.zeros(samp2F)
    for i in range(samp2F):
        tempSignal = sampleBased(samp2F, 44100 * 2, 1.0, i)
        corr2F[i] = np.dot(tempSignal, sumSig)
    
    print np.pi * np.argmax(corr2F) / samp2F

    plt.plot(corr2F)
    
    corr3F = np.zeros(samp3F)
    for i in range(samp3F):
        tempSignal = sampleBased(samp3F, 44100 * 2, 1.0, i)
        corr3F[i] = np.dot(tempSignal, sumSig)

    print np.pi * np.argmax(corr3F) / samp3F

    plt.plot(corr3F)

    plt.show()

'''
Default fs = 44100
'''
def DFT1(x = None, fs = 44100, N = 1024, H = 512):

    if x == None:
        x = mixSinusoids(randPhase = True, fs = fs, t = 0.02)


    #Converting to numpy array
    x = np.array(x)

    #Initializing the DFT Matrix
    numHops = (x.size - N) / H

    #Check if all the samples are covered
    extraSamples = (x.size - N) - (numHops * H)

    print 'The incomplete frame is of size..' + str(extraSamples)

    if extraSamples != 0:
        print 'The N and H do not completely cover the signal.. Adding dummy samples'
        sampAdd = H - extraSamples
        print 'Number of dummy samples to add..' + str(sampAdd)
        
        #Add sampAdd to the end of x
        x = np.append(x, np.zeros(sampAdd))
        numHops += 1 #Adding 1 to numFrames due to addition of dummy samples

    numFrames = numHops + 1

    dft = np.zeros((numFrames, N), dtype=np.complex64) #Initializing the dft matrix with zeros
    print dft

    for frame in range(numFrames):
        #print 'Calculating for frame number ' + str(frame)
        for k in range(N):
            temp = 0j
            for n in range(N):
                temp += x[n + H * frame] * np.exp(-1j * 2 * np.pi * k * n / N)

            #print 'Calculating for k=' + str(k) 

            dft[frame, k] = temp


    return dft



'''
This function calculates with DFT with the bute force method and with FFT method and stores them in a pickle files
'''
def DFT(x = None, fs = 44100, f = 440, t = 0.0058):
    if x == None:
        x = mixSinusoids(randPhase = True, t = t, fs = fs, f = f)

    xF = x[1]
    x2F = x[2]
    x = x[0]

    print len(xF)
    print len(x2F)
    print len(x)
    
    N = x.size

    if all(x == 0):
        return np.zeros(N)  # If all the elements of the list are zero, no need of the computation

    X = np.zeros(N, dtype = np.complex64)

    sTime = ti.default_timer()
    for k in range(N):
        temp = 0j
        for n in range(N):
            temp += x[n] * np.exp(-1j * 2 * np.pi * k * n / N) #Computation for sample 'n'

        X[k] = temp
    eTime = ti.default_timer()

    print 'The elapsed time in calculation of the DFT is:' + str(eTime - sTime)

    # DFT calculation by scipy fft pack

    sTime = ti.default_timer()
    Xfft = fft(x)
    
    eTime = ti.default_timer()
    print 'The elapsed time in calculation of the FFT is:' + str(eTime - sTime)

    fig = plt.figure()
    '''
    fwDFT = open('dft.pkl','w')
    fwFFT = open('fft.pkl','w')

    pickle.dump(X, fwDFT)
    pickle.dump(Xfft, fwFFT)

    fwDFT.close()
    fwFFT.close()
    '''
    '''
    # Plotting the fft and dft

    ax1 = fig.add_subplot('211')
    ax1.plot(np.abs(X))

    ax2 = fig.add_subplot('212')
    ax2.plot(np.abs(Xfft))
    '''
    #basisF = 

    # Plotting the multiplication with fundamental
    

    ax3 = fig.add_subplot(311)
    ax3.plot(x * xF)

    ax4 = fig.add_subplot(312)
    ax4.plot(x * x2F)

    numSamples = int(fs * t)
    fRand = random.randrange(0, 10000) + 1
    fRand = 3 * f
    sampRand = fs / (fRand)
    phase = random.randrange(0, sampRand) + 1
    amp2F = 1
    randBasis = sampleBased(sampRand, numSamples, amp2F, phase)
    ax5 = fig.add_subplot(313)
    ax5.plot(x * randBasis)
    print 'The random phase is:' + str(phase)
    print 'The random freq is:' + str(fRand)
    #print 'The bin number closest to f0 = ' + str(binNo3F)

    plt.show()

    return (X, Xfft)
