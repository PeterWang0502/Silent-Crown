import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from sklearn.neighbors import KNeighborsClassifier


def calc_accuracy(yHat, yTrue):
    a = 0
    for i in range(len(yTrue)):
        if (yTrue[i] != yHat[i]):
            a += 1
    return 1.0-a/len(yHat)



def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("xTrain",
                        help="filename for features of the training data")
    parser.add_argument("yTrain",
                        help="filename for labels associated with training data")
    
    
    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    df = file_to_numpy(args.xTrain)
    y = file_to_numpy(args.yTrain)
 
	list_trainAuc = []
	list_testAuc = []


	list_neighbor = list(range(1,20))
	list_kfold = list(range(10, 105, 5))

	for num in list_kfold:
	    kf = KFold(n_splits=num)
	    trainA = []
	    testA = []
	    for trainIndex, testIndex in kf.split(df):

	        xTrain = df[trainIndex,:]
	        yTrain = y[trainIndex]
	        xTest = df[testIndex,]
	        yTest = y[testIndex]

	        neigh = KNeighborsClassifier(n_neighbors=17)
	        neigh.fit(xTrain, yTrain)
	        yTr = neigh.predict(xTrain)
	        yTe = neigh.predict(xTest)
	        trainA.append(calc_accuracy(yTr, yTrain))
	        testA.append(calc_accuracy(yTe, yTest))
	    list_trainAuc.append(mean(trainA))
	    list_testAuc.append(mean(testA))

	plt.plot(list_kfold, list_trainAuc)
	plt.plot(list_kfold, list_testAuc)
	plt.legend(["Training", "Test"])
	plt.xlabel("Number of K-Fold")
	plt.ylabel("Test Accuracy")
	plt.title("Auc For K-NN")
	plt.show()
	
if __name__ == "__main__":
    main()