import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib as plt

# The thousands part is saying to treat any ',' as 1000
df = pd.read_csv(r'C:\Users\Callum\Documents\ML Researcher\market-direction-model\data\Download Data - INDEX_UK_FTSE UK_UKX.csv', thousands = ',')
print(df.head())

# This converts to true or false based upon rise of fall via 1 or 0
df['Target'] = np.sign(df['Close'] - df['Open']).astype(int)

print(df.head())

# creating the X and y
X = df[['Open', 'Close']] # Remeber double [] creates a new df which is needed for scikitlearn
y = df['Target']

split_point = int(len(X) * 0.8) # This splits the data into 80% chunk.
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]

print(X_train)
print(y_train)

logreg = LogisticRegression(random_state = 16)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

# This is the calculation of the accuracy score used in classification models.
A_S = accuracy_score(y_test, y_pred)

print(f'The score is {A_S:.2f}')

# setting up the confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred)

