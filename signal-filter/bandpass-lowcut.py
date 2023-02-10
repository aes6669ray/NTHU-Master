import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import lfilter,butter,filtfilt
import pandas as pd

x=np.linspace(0, 1,5000)
y=np.sin(2*np.pi*x*600)+np.sin(2*np.pi*x*1250)
z=-1*np.sin(x)

df=pd.read_excel("test2.xlsx")

filter1=df["Event"] == "M3"
filter2=df["Event"] == "M4"
df2=df[filter1 | filter2]

sampling_rate=512
fft_size=len(df2.Time.index)

x=df2.Time
y=df2.BDV0

wind_y=y[:fft_size]
fft_y=np.fft.rfft(wind_y)/fft_size

freqs = np.linspace(0, sampling_rate/2, int(fft_size/2+1))
xfp = 20*np.log10(np.clip(np.abs(fft_y), 1e-20, 1e100))

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band',analog=False)
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=6):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_lowpass(cutOff, fs, order=5):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = butter(order, normalCutoff, btype='low', analog = False)
    return b, a

def butter_lowpass_filter(data, cutOff, fs, order=5):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = lfilter(b, a, data)
    return y


yy=butter_bandpass_filter(y,10, 100, 512)
wind_yy=yy[:fft_size]
fft_yy=np.fft.rfft(wind_yy)/fft_size

freqs1 = np.linspace(0, sampling_rate/2, int(fft_size/2+1))
xfp1 = 20*np.log10(np.clip(np.abs(fft_yy), 1e-20, 1e100))

zz=butter_lowpass_filter(y,0.5, 512)

wind_zz=zz[:fft_size]
fft_zz=np.fft.rfft(wind_zz)/fft_size

freqs2 = np.linspace(0, sampling_rate/2, int(fft_size/2+1))
xfp2 = 20*np.log10(np.clip(np.abs(fft_zz), 1e-20, 1e100))

print(len(xfp2))

plt.figure()
plt.subplot(321)
plt.plot(x[:300], y[:300])

plt.subplot(322)
plt.plot(freqs, xfp)

plt.subplot(323)
plt.plot(x[:300], yy[:300])

plt.subplot(324)
plt.plot(freqs1, xfp1)

plt.subplot(325)
plt.plot(x[:300], zz[:300])

plt.subplot(326)
plt.plot(freqs2, xfp2)

plt.show()