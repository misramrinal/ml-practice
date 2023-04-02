import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import ElasticNet, Lasso

df = pd.read_csv("assignment-1.csv")
df=df.replace(to_replace=['two','three','four','five','six','eight','twelve'],value=[2,3,4,5,6,8,12])
# print(df.head())

X = df.drop(['car_ID','symboling','CarName','fueltype','aspiration','carbody','drivewheel','enginelocation','fuelsystem','enginetype','price'], axis=1)
df.head()
# X = df[5].values.reshape(-1,1)
Y = df['price']
# Y.head()
x_train, x_test, y_train, y_test = tts(X, Y, train_size=0.2, random_state=0)

l = Lasso(alpha=0.05, max_iter=10000)
l.fit(x_train, y_train)
y_pred1=l.predict(x_test)
e = ElasticNet(alpha = 67, max_iter=10000)
e.fit(x_train, y_train)
y_pred2 = e.predict(x_test)
y_final=(y_pred1+y_pred2)/2
# plt.plot(x_test,y_pred)
mse(y_final, y_test, squared=False)
# X.head()
# y_final
# y_test
# l.coef_