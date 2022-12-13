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

def rhythmic_variance(df):
    return df['end_beat'].var()

def rhythmic_variety(df):
    num_durations = len(set(df['end_beat']))
    return 10 * num_durations / len(df.note.unique())

def leaps_ratio(df):
    leaps = 0
    steps = 0
    for i in range(len(df)-1):
        distance = abs(df['end_beat'][i+1]-df['end_beat'][i])
        if distance > 2:
            leaps += 1
        else:
            steps += 1
    return leaps/(leaps+steps)

def repeated_notes(df):
    return stats.mode(df['end_beat'])[1][0]

def chromaticism(df):
    is_chromatic = 0
    for element in df.instrument.unique():
        temp = list(df.note.loc[df.instrument == element])
        for i in range(len(temp)-2):
            if ((df.note[i]-df.note[i+1] == 1) and (df.note[i+1]-df.note[i+2] == 1)):
                if((df.end_time[i] == df.start_time[i+1]) and (df.end_time[i+1] == df.start_time[i+2])):
                    is_chromatic = 1
            if ((df.note[i+2]-df.note[i+1] == 1) and (df.note[i+1]-df.note[i] == 1)):
                if((df.end_time[i+2] == df.start_time[i+1]) and (df.end_time[i+1] == df.start_time[i+2])):
                    is_chromatic = 1
    return(is_chromatic)

def dimension_reduc(ensemble, beat_value, metadata, name):
    df = pd.DataFrame(columns=ensemble + ['instrument ' + s for s in [str(j) for j in range(1,129)]] + beat_value +
                      ['total notes', 'duration', 'rhythmic variance', 'rhythmic_variety', 'leaps_ratio', 'repeated_notes', 'chromaticism'])
    y = []
    for i in tqdm(os.listdir(name)):
        if i.endswith('.csv'):
            te = list(np.repeat(0, len(ensemble)))
            ens = list(metadata[metadata.id == int(i[0:4])].ensemble)[0]
            te[ensemble.index(ens)] = 1

            train_data = pd.read_csv(name + '/' + i)
            instrument = np.unique(train_data.instrument)
            ti = list(np.repeat(0, 128))
            for j in instrument:
                ti[j-1] = 1

            ta = list(np.repeat(0, 27))
            for j in range(len(beat_value)):
                ta[j] = len(np.unique(train_data[train_data.end_beat==beat_value[j]].note))
            ta[20] = len(np.unique(train_data.note))
            ta[21] = list(metadata[metadata.id == int(i[0:4])].seconds)[0]

            ta[22] = rhythmic_variance(train_data)
            ta[23] = rhythmic_variety(train_data)
            ta[24] = leaps_ratio(train_data)
            ta[25] = repeated_notes(train_data)
            ta[26] = chromaticism(train_data)

            temp = te + ti + ta
            df.loc[len(df)] = temp

            y.append(list(metadata[metadata.id == int(i[0:4])].composer)[0])
    return df, y

def main():
    metadata = pd.read_csv('musicnet_metadata.csv')
    beat_value = [0.0625, 0.0833333333333, 0.0833333333334, 0.125, 0.166666666667, 0.1875, 0.25, 0.375, 0.5, 0.666666666667,
                  0.75, 0.833333333333, 0.875, 1.0, 1.16666666667, 1.5, 2.0, 2.5, 3.0, 3.5]
    ensemble = list(np.unique(metadata.ensemble))
    xTrain, yTrain = dimension_reduc(ensemble, beat_value, metadata, 'train_data')
    xTest, yTest = dimension_reduc(ensemble, beat_value, metadata, 'test_data')

    yTrain = pd.DataFrame(yTrain, columns = ['composer'])
    yTest = pd.DataFrame(yTest, columns = ['composer'])

    #xTrain.to_csv('xTrain.csv', index=False, header=True)
    yTrain.to_csv('yTrain.csv', index=False, header=True)
    #xTest.to_csv('xTest.csv', index=False, header=True)
    yTest.to_csv('yTest.csv', index=False, header=True)

    li = []
    for i in range(len(xTrain.columns)):
        a = sum(xTrain.iloc[:,i] != 0)
        if a > 0:
            li.append(False)
        else:
            li.append(True)

    for i in xTrain.columns[li]:
        xTrain = xTrain.drop(i, axis=1)
        xTest = xTest.drop(i, axis=1)

    xTrain.to_csv('filtered_xTrain.csv', index=False, header=True)
    xTest.to_csv('filtered_xTest.csv', index=False, header=True)

if __name__ == "__main__":
    main()
