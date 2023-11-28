# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt 
import scipy as sc
import pandas as pd


# Importing data
df_1 = pd.read_csv('piano.txt', header=None)
df_2 = pd.read_csv('trumpet.txt')
piano = df_1.to_numpy().transpose().reshape(100000)
trumpet = df_2.to_numpy().transpose().reshape(99999)
t_1 = np.linspace(0., 100000./44100., 100000)
t_2 = np.linspace(0., 99999./44100., 99999)


# Plotting piano waveform as a function of time
plt.plot(t_1, piano, '.')
plt.xlabel('Time ($s$)', fontsize=16)
plt.ylabel('Piano Waveform', fontsize=16)
plt.savefig('piano_waveform.png')
plt.cla()


# Plotting trumpet waveform as a function of time
plt.plot(t_2, trumpet, '.')
plt.xlabel('Time ($s$)', fontsize=16)
plt.ylabel('Trumpet Waveform', fontsize=16)
plt.savefig('trumpet_waveform.png')
plt.cla()


# Calculating FFT for the piano waveform 
n_piano = piano.size
piano_fft = sc.fft.rfft(piano)
piano_freq = sc.fft.rfftfreq(n_piano, d=1/44100)
plt.plot(piano_freq, np.abs(piano_fft))
plt.xlim(0,10000)
plt.xlabel('Frequency, (Hz)', fontsize=16)
plt.ylabel('Amplitude', fontsize=16)
plt.savefig('piano_fft.png')
plt.cla()


# Calculating FFT for the piano waveform 
n_trumpet = trumpet.size
trumpet_fft = sc.fft.rfft(trumpet)
trumpet_freq = sc.fft.rfftfreq(n_trumpet, d=1/44100)
plt.plot(trumpet_freq, np.abs(trumpet_fft))
plt.xlim(0,10000)
plt.xlabel('Frequency, (Hz)', fontsize=16)
plt.ylabel('Amplitude', fontsize=16)
plt.savefig('trumpet_fft.png')
plt.cla()


# Determining frequency from piano spectrum
max_piano_index = np.where(piano_fft == np.max(piano_fft))
piano_frequency = piano_freq[max_piano_index]

print('Frequency of piano note is', piano_freq[max_piano_index],'Hz, so the C one octave above middle C')


# Determining frequency from trumpet spectrum
max_trumpet_index = np.where(trumpet_fft == np.max(trumpet_fft))
trumpet_frequency = trumpet_freq[max_trumpet_index]

print('Frequency of trumpet note is', trumpet_frequency,'Hz, so the C 2 octaves above middle C')