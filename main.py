import sklearn.model_selection as model_selection
import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn import datasets


def zad3(data):
    print("MAX: ")
    print(data.max())
    print("MIN: ")
    print(data.min())
    print("Ilość wartości: ")
    print(data.size)
    print("Wartość średnia: ")
    print(np.nanmean(data))
    print("Ilość wartości unikatowych: ")
    print(np.unique(data).size)
    print("Ilość nulli: ")
    print(np.count_nonzero(np.isnan(data)))
    print("Najczęstsza wartość: ")
    datas = np.array(data)
    print(mode(datas))
# Zadanie 1
wines = datasets.load_wine()
# Zadanie 2
x, y = wines.data, wines.target
X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.60, test_size=0.40)
pd.DataFrame(X_train).to_csv("out/x_train.csv")
pd.DataFrame(X_test).to_csv("out/X_test.csv")
pd.DataFrame(y_train).to_csv("out/y_train.csv")
pd.DataFrame(y_train).to_csv("out/y_test.csv")
# Zadania 3
zad3(X_train)
zad3(X_test)
zad3(y_train)
zad3(y_test)





