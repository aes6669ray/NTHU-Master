import pandas as pd
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

df=pd.read_excel("test1.xlsx")

filter1=df["Event"] == "M3"
filter2=df["Event"] == "M4"

df2=df[filter1 | filter2]

y=df2["BDV0"]

y_filtered = savgol_filter(y, 11, 3)

plt.subplot(211)
plt.plot(df2.Time,df2.BDV0)

plt.subplot(212)
plt.plot(df2.Time,y_filtered)
plt.show()