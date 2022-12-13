import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import os
from sklearn.model_selection import KFold
from sklearn import metrics
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier

def main():
    trainAuc = []
    testAuc = []
    for i in tqdm(range(10, 100+1, 10)):
        kf = KFold(n_splits=i)
        trainA = []
        testA = []
        for trainIndex, testIndex in KFold(n_splits=i, shuffle=True, random_state=1).split(xTrain):
            xTr = xTrain.iloc[trainIndex,:]
            yTr = yTrain.iloc[trainIndex,:]
            xTe = xTrain.iloc[testIndex,:]
            yTe = yTrain.iloc[testIndex,:]

            rf = RandomForestClassifier(n_estimators=250, max_depth=10, random_state=0)
            DecisionTreeClassifier(criterion="gini", max_depth=10, min_samples_leaf=1)
            rf.fit(xTr, yTr)
            yr = rf.predict(xTr)
            ye = rf.predict(xTe)

            trainA.append(calc_accuracy(yr, yTr['composer'].tolist()))
            testA.append(calc_accuracy(ye, yTe['composer'].tolist()))

        trainAuc.append(np.mean(trainA))
        testAuc.append(np.mean(testA))

    plt.plot(list(range(10, 101, 10)),trainAuc)
    plt.plot(list(range(10, 101, 10)),testAuc)
    plt.title('Random forest accuracy using 100 trees with maximum depth of 10')
    plt.legend(['training accuracy', 'testing accuracy'])
    plt.xlabel('Number of k in K-fold validation')
    plt.ylabel('model accuracy')
    plt.show()

    print(np.mean(trainAuc))
    print(np.mean(testAuc))

    rf = RandomForestClassifier(n_estimators=250, max_depth=10, random_state=0)
    rf.fit(xTrain, yTrain)
    yrf = rf.predict(xTest)
    print(calc_accuracy(yrf, yTest['composer'].tolist()))

if __name__ == "__main__":
    main()
