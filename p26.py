import numpy as np
x=np.array([0,2,4,5,7,9,15,20,22])
y=np.array([-3,-1,4,3,5,9,5,6,7])

model=np.poly1d(np.polyfit(x,y,1))
ypred=model(x)

from sklearn.metrics import r2_score
print("r2 = ",r2_score(y,ypred))
import matplotlib.pyplot as plt
plt.scatter(x,y,label="data")
plt.plot(x,ypred,label="fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
