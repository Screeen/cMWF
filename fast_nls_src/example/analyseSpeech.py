import sys

import librosa
from scipy.io import wavfile
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

sys.path.append('../')
import single_pitch

samplingFreq = 16000  # target sampling frequency [Hz]
maxLenSeconds = 5

file_name = 'roy.wav'  # 'roy.wav' double-bass-a-1.wav violin-d4.wav
[speechSignal, inputSamplingFreq] = librosa.load(file_name, sr=None)
if speechSignal.ndim > 1:
    speechSignal = speechSignal[:, 0]  # take only one channel
speechSignal = speechSignal[:maxLenSeconds * inputSamplingFreq]  # take only the first maxLenSeconds seconds
nData = speechSignal.shape[0]

# Resample to samplingFreq
if inputSamplingFreq != samplingFreq:
    speechSignal = sp.signal.resample(speechSignal, int(nData * samplingFreq / inputSamplingFreq))
    nData = speechSignal.shape[0]

# set up
segmentTime = 0.025  # seconds
segmentLength = round(segmentTime * samplingFreq)  # samples
nSegments = int(np.floor(nData / segmentLength))
f0Bounds = np.array([80, 400]) / samplingFreq
maxNoHarmonics = 20
minModelOrder = 4

f0Estimator = single_pitch.single_pitch(segmentLength, maxNoHarmonics, f0Bounds)

# do the analysis
idx = np.array(range(0, segmentLength))
f0Estimates = np.zeros((nSegments,))  # cycles/sample
modelOrders = np.zeros((nSegments,))
for ii in range(nSegments):
    speechSegment = np.array(speechSignal[idx], dtype=np.float64)
    f0Estimates[ii] = (samplingFreq / (2 * np.pi)) * f0Estimator.est(speechSegment, lnBFZeroOrder=0, eps=1e-5)
    modelOrders[ii] = f0Estimator.modelOrder()
    idx = idx + segmentLength

f0Estimates_copy = f0Estimates.copy()
modelOrders_copy = modelOrders.copy()

f0Estimates[f0Estimates < f0Bounds[0] * samplingFreq] = np.nan
f0Estimates[f0Estimates > f0Bounds[-1] * samplingFreq] = np.nan
f0Estimates[modelOrders < minModelOrder] = np.nan
modelOrders[modelOrders < minModelOrder] = np.nan

timeVector = np.array(range(1, nSegments + 1)) * segmentTime - segmentTime / 2

# compute the spectrogram of the signal
nOverlap = round(3 * segmentLength / 4)
[stftFreqVector, stftTimeVector, stft] = sp.signal.spectrogram(speechSignal,
                                                               fs=samplingFreq,
                                                               nperseg=segmentLength,
                                                               noverlap=nOverlap,
                                                               nfft=2048)
powerSpectrum = np.abs(stft) ** 2;

# plot the results
maxDynamicRange = 60  # dB
plt.pcolormesh(stftTimeVector, stftFreqVector, 10 * np.log10(powerSpectrum + 1e-12))
plt.scatter(timeVector, f0Estimates, c='b', s=1)
plt.title(file_name.strip('.wav'))
plt.xlabel('time [s]')
plt.ylabel('frequency [Hz]')
plt.show()

fig, ax = plt.subplots()
ax.plot(timeVector, f0Estimates, label='f0 estimate')
ax.set(xlabel='time [s]', ylabel='frequency [Hz]',
       title='f0 estimate')
ax.grid()

# Plot also model order with a different scale
ax2 = ax.twinx()
ax2.plot(timeVector, modelOrders, 'r', label='model order', alpha=0.5, linewidth=0.5)
ax2.set(ylabel='model order')
fig.legend()
fig.show()

# fig2, ax2 = plt.subplots()
# ax2.plot(speechSignal)
# ax2.set(xlabel='samples', ylabel='amplitude',
#        title='speech signal')
# ax2.grid()
# fig2.show()
