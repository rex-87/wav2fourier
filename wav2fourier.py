import numpy as np
import matplotlib.pyplot as plt 
from scipy.io import wavfile
from scipy.signal import butter, filtfilt 
import os

ScriptPath = os.path.dirname(os.path.abspath(__file__))

plt.style.use('seaborn-dark-palette')
pi = np.pi

if True:
    filename = r'1980s-Casio-Organ-C5.wav'
    # filename = r'Bass-Drum-1.wav'
    # filename = r'1980s-Casio-Flute-C5.wav'
    
    samples_directory = os.path.join(ScriptPath, r'wav_samples')
    filepath = os.path.join(samples_directory, filename)
    samplerate, wavread = wavfile.read(filepath)
    wavarray_full_ = np.array(wavread, dtype = float)/2**16

    # filt
    if True:
        b, a = butter(2, 1.0*40/(samplerate/2), btype = 'high')
        wavarray_full_filt = filtfilt(b, a, wavarray_full_)
        wavarray_full = wavarray_full_filt
    else:
        wavarray_full = wavarray_full_
    
    # num_chunks = 3
    # chunk_size = int(round(wavarray_full.size/num_chunks))
    chunk_size = int(round(0.1*samplerate)) # 0.1 => lower bound is 10 Hz
    startpoint = 0
    figindex = 0
    while startpoint < wavarray_full.size:
        wavarray = wavarray_full[startpoint:startpoint+chunk_size]
        
        rangemin = 0
        numpoints = wavarray.size
        rangemax = 1.0*(numpoints + 1)/samplerate
        t = np.linspace(start = rangemin, stop = rangemax, num = numpoints, endpoint = False)

        faxis = np.linspace(start = 0, stop = numpoints/rangemax, num = numpoints, endpoint = False) 
        ft_wavarray = np.fft.fft(wavarray)

        # plot
        numplot = 2
        
        plt.close()
        
        plt.subplot(numplot, 1, 1)
        plt.plot(t, wavarray, alpha = 0.5)
        plt.ylim([-0.5, 0.5])        
        plt.title('File: {}, chunk duration: {:.4} s'.format(filename, 1.0*chunk_size/samplerate))
        plt.subplot(numplot, 1, 2)
        plt.loglog(faxis[:numpoints/2], np.abs(ft_wavarray[:numpoints/2]), alpha = 0.5)
        spectrum_freq_max = 1.0*samplerate*(rangemax - rangemin)/4
        plt.ylim([spectrum_freq_max/10000, spectrum_freq_max])         
        
        startpoint += chunk_size
        
        # plt.show()
        plt.savefig(os.path.join(samples_directory, '{}{}.png'.format(filename, figindex)), format = 'png')
        figindex += 1
else:
    f = 500
    rangemin = 0
    rangemax = 1
    samplerate = 44100
    numpoints = int(round((rangemax - rangemin)*samplerate))
    t = np.linspace(start = rangemin, stop = rangemax, num = numpoints, endpoint = False)
    wavarray = 1.0/2*np.cos(2*pi*f*t) + np.cos(2*pi*2*f*t)

    faxis = np.linspace(start = 0, stop = numpoints/rangemax, num = numpoints, endpoint = False) 
    ft_wavarray = np.fft.fft(wavarray)

    # plt.plot(t, wavarray)
    plt.semilogx(faxis[:numpoints/2], np.abs(ft_wavarray[:numpoints/2]))
    plt.show()