import numpy as np
from numpy.core.fromnumeric import shape, size
from numpy.random.mtrand import RandomState
import pandas as pd
from pandas.core.construction import array
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt
from scipy import signal

np.random.seed(5)
x=np.linspace(0,15,1000)
xa=np.linspace(0,15,999)
xb=np.linspace(0,15,998)
y=np.sin(x)+2*(np.sin(x/2))
z=np.random.normal(size=1000,loc=0,scale=0.1).round(4)
w=y+z

a=[]
for i,r in enumerate(y):
    if i == 0:
        continue
    else:
        j=(y[i]-y[i-1])/0.015
        a.append(j)
b=[]
for i,r in enumerate(a):
    if i == 0 and 1:
        continue
    else:
        j=(a[i]-a[i-1])/0.015
        b.append(j)

c=[]
for i,r in enumerate(w):
    if i == 0:
        continue
    else:
        j=(w[i]-w[i-1])/0.015
        c.append(j)
d=[]
for i,r in enumerate(c):
    if i == 0 and 1:
        continue
    else:
        j=(c[i]-c[i-1])/0.015
        d.append(j)

# plt.figure()

# plt.subplot(321)
# plt.scatter(x,y)
# plt.title("displacement")

# plt.subplot(323)
# plt.scatter(xa,a)
# plt.title("velocity")

# plt.subplot(325)
# plt.scatter(xb,b)
# plt.title("acceleration")

# plt.subplot(322)
# plt.scatter(x,w)
# plt.title("ndisplacement")

# plt.subplot(324)
# plt.scatter(xa,c)
# plt.title("nvelocity")

# plt.subplot(326)
# plt.scatter(xb,d)
# plt.title("nacceleration")

# plt.tight_layout()
# plt.show()

y=y[0:-2]
a=a[0:-1]
x=x[0:-2]
hold={"displacement":y,"velocity":a,"acceleration":b,"linespace":x}
df=pd.DataFrame(hold,columns=["displacement","velocity","acceleration","linespace"])

w=w[0:-2]
c=c[0:-1]
hold2={"ndisplacement":w,"nvelocity":c,"nacceleration":d,"linespace":x}
df2=pd.DataFrame(hold2,columns=["ndisplacement","nvelocity","nacceleration","linespace"])

df3=pd.read_excel("test1.xlsx")


GDV0Z_detrended = signal.detrend(df3["GDV0Z"])
BDV0Z_detrended = signal.detrend(df3["BDV0Z"])
df3["DEBDV0Z"]=BDV0Z_detrended
df3["DEGDV0Z"]=GDV0Z_detrended

DEBDV1Z=[]
for i,n in enumerate(df3.BDV0Z):
    j=(df3.DEBDV0Z[i+1]-df3.DEBDV0Z[i])/(df3.Time[i+1]-df3.Time[i])
    DEBDV1Z.append(j)
    if i == (len(df3.index)-2):
        break
DEBDV1Z.append(None)
df3["DEBDV1Z"]=DEBDV1Z

DEBDV2Z=[]
for i,n in enumerate(df3.DEBDV1Z):
    j=(df3.DEBDV1Z[i+1]-df3.DEBDV1Z[i])/(df3.Time[i+1]-df3.Time[i])
    DEBDV2Z.append(j)
    if i == (len(df3.index)-2):
        break
DEBDV2Z.append(None)
df3["DEBDV2Z"]=DEBDV2Z

df3.dropna(inplace=True)
tx1=df3[["BDV0Z","BDV1Z"]]
ty1=df3["BDV2Z"]
tx2=df3[["DEBDV0Z","DEBDV1Z"]]
ty2=df3["DEBDV2Z"]

# X=df[["displacement","velocity"]]
# Y=df["acceleration"]

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
# reg=linear_model.LinearRegression(fit_intercept=False)
# reg.fit(X_train,y_train)
# print("練習Score",reg.score(X_train,y_train))
# print("測驗Score",reg.score(X_test,y_test))
# print("相關係數",reg.coef_)

aa=np.arange(0.1,10.1,0.1)
# ridge=linear_model.RidgeCV(alphas=aa,cv=5,fit_intercept=False)
# ridge.fit(X_train,y_train)
# print("練習Score",ridge.score(X_train,y_train))
# print("測驗Score",ridge.score(X_test,y_test))
# print("相關係數",ridge.coef_)
# print("最佳參數",ridge.alpha_)

# X=df2[["ndisplacement","nvelocity"]]
# Y=df2["nacceleration"]


# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(tx2, ty2, test_size=0.3, random_state=2)
ridge=linear_model.RidgeCV(alphas=aa,cv=5,fit_intercept=False)
ridge.fit(X_train,y_train)
print("練習Score",ridge.score(X_train,y_train))
print("測驗Score",ridge.score(X_test,y_test))
print("相關係數",ridge.coef_)
print("最佳參數",ridge.alpha_)











