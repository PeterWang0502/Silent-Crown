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

def calc_accuracy(yHat, yTrue):
    a = 0
    for i in range(len(yTrue)):
        if yTrue[i] != yHat[i]:
            a += 1
    return 1.0-a/len(yHat)

def main():
    xTrain = pd.read_csv('filtered_xTrain.csv')
    yTrain = pd.read_csv('yTrain.csv')
    xTest = pd.read_csv('filtered_xTest.csv')
    yTest = pd.read_csv('yTest.csv')

    trainAuc = []
    testAuc = []
    for i in tqdm(range(10, 100+1, 10)):
        kf = KFold(n_splits=i)
        trainA = []
        testA = []
        for trainIndex, testIndex in KFold(n_splits=i, shuffle=True, random_state=1).split(xTrain):
            xTr = xTrain.iloc[trainIndex,:]
            yTr = yTrain.iloc[trainIndex]
            xTe = xTrain.iloc[testIndex,]
            yTe = yTrain.iloc[testIndex]

            dt = DecisionTreeClassifier(criterion="gini", max_depth=10, min_samples_leaf=1)
            dt.fit(xTr, yTr)
            yr = dt.predict(xTr)
            ye = dt.predict(xTe)

            trainA.append(calc_accuracy(yr, yTr['composer'].tolist()))
            testA.append(calc_accuracy(ye, yTe['composer'].tolist()))

        trainAuc.append(np.mean(trainA))
        testAuc.append(np.mean(testA))

    plt.plot(list(range(10, 101, 10)),trainAuc)
    plt.plot(list(range(10, 101, 10)),testAuc)
    plt.title('Decision tree accuracy using Gini index with maximum depth of 10')
    plt.legend(['training accuracy', 'testing accuracy'])
    plt.xlabel('Number of k in K-fold validation')
    plt.ylabel('model accuracy')
    plt.show()

    dt = DecisionTreeClassifier(criterion="gini", max_depth=10, min_samples_leaf=1)
    dt.fit(xTrain, yTrain)
    y = dt.predict(xTest)
    print(calc_accuracy(y, yTest['composer'].tolist()))
    fig = plt.figure(figsize=(75,60))
    _ = tree.plot_tree(dt,
                    feature_names=xTrain.columns,
                    class_names=y,
                    filled=True)

if __name__ == "__main__":
    main()
