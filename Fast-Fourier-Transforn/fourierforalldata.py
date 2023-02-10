import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, lfilter,filtfilt
import os

path= r"C:\Users\aes66\OneDrive\桌面\python test\TEST\8_test"
dirs=os.listdir(path)

df=pd.DataFrame()
dic={}
dic2={}

for file in dirs:
    df=pd.read_excel(r"C:\Users\aes66\OneDrive\桌面\python test\TEST\8_test\\" + str(file))
    filter1=df["Event"] == "M3"
    filter2=df["Event"] == "M4"
    df2=df[filter1 | filter2]
    dic2={file:df2}
    dic.update(dic2)

a=0

fig, axs = plt.subplots(5,4,figsize=(20,60))
for i,n in enumerate(dic.values()):
    j=pd.DataFrame(n)
    sampling_rate=512
    fft_size=len(j.Time.index)
    yb=j.BDV0

    wind_yb=yb[:fft_size]
    fft_yb=np.fft.rfft(wind_yb)/fft_size

    freqsb = np.linspace(0, sampling_rate/2, int(fft_size/2+1))
    xfpb = 20*np.log10(np.clip(np.abs(fft_yb), 1e-20, 1e100))

    yg=j.GDV0

    wind_yg=yg[:fft_size]
    fft_yg=np.fft.rfft(wind_yg)/fft_size

    freqsg = np.linspace(0, sampling_rate/2, int(fft_size/2+1))
    xfpg = 20*np.log10(np.clip(np.abs(fft_yg), 1e-20, 1e100))

    axs[a,0].plot(j.Time,j.BDV0)
    axs[a,1].plot(freqsb,xfpb)
    
    axs[a,2].plot(j.Time,j.GDV0)
    axs[a,3].plot(freqsg,xfpg)
    a+=1

plt.show()


  
