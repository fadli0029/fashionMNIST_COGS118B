import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Importing training data
df_train = pd.read_csv('fashion-mnist_train.csv')
y_train = df_train['label'].to_numpy()
X_train = df_train[df_train.columns[1:]].to_numpy()

# Importing testing data
df_test = pd.read_csv('fashion-mnist_test.csv')
y_test = df_test['label'].to_numpy()
X_test = df_test[df_train.columns[1:]].to_numpy()

# Normalizing data to be between 0-1
X_train = X_train/255
X_test = X_test/255

# Creating SVM model (SVC is the non-linear version of SVM)
non_linear_model = SVC(kernel='rbf')

# Fitting data to model
non_linear_model.fit(X_train, y_train)

# Testing data 
y_pred = non_linear_model.predict(X_test)

# Finding result accuracy
accuracy =  accuracy_score(y_true=y_test, y_pred=y_pred)

# Result 
print(confusion_matrix(y_true=y_test, y_pred=y_pred))
plt.imshow(confusion_matrix(y_true=y_test, y_pred=y_pred), cmap=plt.cm.hot)
plt.colorbar()
plt.title('Support Vector Machine\nAccuracy: ' + str((1 - accuracy)*100) + '% error')
plt.xlabel('Predicted')
plt.ylabel('True Value')
plt.show()