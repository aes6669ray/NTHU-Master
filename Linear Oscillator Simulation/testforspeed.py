import numpy as np
from numpy.lib.function_base import percentile
import scipy
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model

x=np.linspace(0,16*np.pi,10000)
y=np.sin(x)

posweight=[]
for i in np.linspace(5,1,4000):
    posweight.append(i)

negweight=[]
for i in np.linspace(1,5,4000):
    negweight.append(i)

k=[1]*6000
posweight=posweight+k
negweight=k+negweight

table={"posweight":posweight,"negweight":negweight,"y":y,"Time":x}
df=pd.DataFrame(table)

df["posspeed"]=df.posweight*df.y
df["negspeed"]=df.negweight*df.y

usecols=["posspeed","negspeed","Time"]
df2=pd.DataFrame(df,columns=usecols)


a=[]
for i,n in enumerate(df.posspeed):
    if i == 0:
        continue
    else:
        j=(df.posspeed[i]-df.posspeed[i-1])/(x[i]-x[i-1])
        a.append(j)

b=[]
for i,n in enumerate(a):
    if i == 0:
        continue
    else:
        j=(a[i]-a[i-1])/(x[i]-x[i-1])
        b.append(j)

c=[]
for i,n in enumerate(df.negspeed):
    if i == 0:
        continue
    else:
        j=(df.negspeed[i]-df.negspeed[i-1])/(x[i]-x[i-1])
        c.append(j)

d=[]
for i,n in enumerate(c):
    if i == 0:
        continue
    else:
        j=(c[i]-c[i-1])/(x[i]-x[i-1])
        d.append(j)

a=a[:-1]
c=c[:-1]
df2=df2.iloc[:-2,:]
df2["de1pos"]=a
df2["de2pos"]=b
df2["de1neg"]=c
df2["de2neg"]=d

df2.drop(df2[df2.de2neg <-20 ].index, inplace=True)
df2.drop(df2[df2.de2pos > 20 ].index, inplace=True)



X=df2[["posspeed","de1pos"]]
Y=df2["de2pos"]

# X=df2[["negspeed","de1neg"]]
# Y=df2["de2neg"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
reg=linear_model.LinearRegression(fit_intercept=False)
reg.fit(X_train,y_train)
print("練習Score",reg.score(X_train,y_train))
print("測驗Score",reg.score(X_test,y_test))
print("相關係數",reg.coef_)

plt.figure()
plt.subplot(321)
plt.scatter(df2.Time,df2.posspeed)

plt.subplot(323)
plt.scatter(df2.Time,df2.de1pos)

plt.subplot(325)
plt.scatter(df2.Time,df2.de2pos)

plt.subplot(322)
plt.scatter(df2.Time,df2.negspeed)

plt.subplot(324)
plt.scatter(df2.Time,df2.de1neg)

plt.subplot(326)
plt.scatter(df2.Time,df2.de2neg)
plt.show()


