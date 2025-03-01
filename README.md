import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("/content/Student_Performance.csv")  # Update the path accordingly
df.head()

model = linear_model.LinearRegression()

X='Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced'
y='Performance Index'

model.fit(df[['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced']], df[['Performance Index']])

model.score(df[['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced']], df[['Performance Index']])

model.predict([[10, 85, 7, 5]])

model.coef_

model.intercept_

y=2.85342921*10+1.01858354*85+0.47633298*7+0.1951983*5-33.76372609

y



y_pred=model.predict
y

!pip install scikit-learn

#X = df.iloc[:, 0].values
#y = df.iloc[:, 1].values

print(X.shape)
print(y.shape)
class MereLR:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, x_train, y_train):
        x_train = np.insert(x_train, 0, 1, axis=1)
        betas = np.linalg.inv(x_train.T.dot(x_train)).dot(x_train.T).dot(y_train)
        self.intercept_ = betas[0]
        self.coef_ = betas[1:]

    def predict(self, x_test):
        x_test = np.insert(x_test, 0, 1, axis=1)
        return np.dot(x_test, np.append(self.intercept_, self.coef_))

#self.intercept_ = betas[0]`: The first value of `betas` represents the intercept (bias term) because we added a column of ones to `X_train`.
# self.coef_ = betas[1:]`: The remaining values are the feature coefficients.
# np.insert(X_train, 0, 1, axis=1)`: Adds a column of ones to include the intercept in matrix calculations. '''

from sklearn.model_selection import train_test_split
X_data = df[['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced']]
y_data = df['Performance Index']
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=2)
lr = MereLR()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test.values)

r2 = r2_score(y_test, y_pred)
r2
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)



from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(X_test['Hours Studied'], X_test['Previous Scores'], y_test, color='blue', label="Actual")
ax.scatter(X_test['Sleep Hours'], X_test['Sample Question Papers Practiced'], y_pred, color='red', label="Predicted")

ax.set_xlabel("Hours Studied / Sleep Hours")
ax.set_ylabel("Previous Scores / Sample Papers Practiced")
ax.set_zlabel("Performance Index")
ax.set_title("3D Visualization of Predictions")

plt.legend()
plt.show()



