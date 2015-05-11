import os
import sys
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import q1
import numpy as np
import scipy.io.wavfile as wav

def part1():
    S = 2400

    sb600 = q1.sampleBased(N = 600, S = S, A = 1.0, phi = 0)
    sb600Half = q1.sampleBased(N = 600.5, S = S, A = 1.0, phi = 0)
    window = np.hanning(S)

    sb600HalfWindowed = window * sb600Half

    dft600 = np.fft.fft(sb600)
    dft600Half = np.fft.fft(sb600HalfWindowed)

    plt.plot(sb600)
    plt.plot(sb600HalfWindowed)
    plt.plot(np.log10(np.abs(dft600)))
    plt.plot(np.log10(np.abs(dft600Half)))

    plt.show()


def part2(filePath = '/home/swapnil/Dropbox/UVicMIR/assgn1/clarinet.wav'):
    if not os.path.isfile(filePath):
        print 'The file does not exist: ' + filePath
    
    fs, x = wav.read(filePath)

    print 'The sampling rate is =' + str(fs)
    print 'The number of samples =' + str(x.size)

    #get 2048 samples from the 0.5 second

    samp2048 = x[int(0.5 * fs) : int(0.5 * fs) + 2048]

    fig = plt.figure()

    dft2048 = fft(samp2048)

    ax1 = fig.add_subplot(521)
    ax2 = fig.add_subplot(522)
    ax1.plot(20 * np.log10(np.abs(dft2048)))
    ax2.plot(samp2048)

    samp256 = x[int(0.5 * fs) : int(0.5 * fs) + 256]
    window = np.hanning(256)
    
    #Calculating the DFT with 256 sample window
    dft256 = fft(samp256)
    dft256Wind = fft(samp256 * window)

    #Plotting 256 sample without windowing. 
    ax3 = fig.add_subplot(523)
    ax4 = fig.add_subplot(524)
    ax3.plot(20 * np.log10(np.abs(dft256)))
    ax4.plot(samp256)

    #Plotting 256 sample with windowing
    ax5 = fig.add_subplot(525)
    ax6 = fig.add_subplot(526)
    ax5.plot(20 * np.log10(np.abs(dft256Wind)))
    ax6.plot(samp256 * window)

    #Calculating with zero-padding
    samp2048Pad = np.append(samp256, np.zeros(2048 - 256))
    window2048 = np.hanning(2048)
    
    #Calculating the DFT with 256 sample window with zero padding
    dft2048Pad = fft(samp2048Pad)
    dft2048PadWind = fft(samp2048Pad * window2048)

    #Plotting 256 sample + zero-padding without windowing. 
    ax7 = fig.add_subplot(527)
    ax8 = fig.add_subplot(528)
    ax7.plot(20 * np.log10(np.abs(dft2048Pad)))
    ax8.plot(samp2048Pad)

    #Plotting 256 sample + zero-padding without windowing. 
    ax9 = fig.add_subplot(529)
    ax10 = fig.add_subplot(5,2,10)
    ax9.plot(20 * np.log10(np.abs(dft2048PadWind)))
    ax10.plot(samp2048Pad * window2048)
    
    fig.show()
