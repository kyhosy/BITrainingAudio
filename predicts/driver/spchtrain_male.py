#train_models.py
import datetime
import os
import librosa
import _pickle as cPickle
import numpy as np
from sklearn.mixture import GMM
import python_speech_features as mfcc
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")


def get_MFCC(sr,audio):
    # print('get_MFCC {}'.format(sr))

    features = mfcc.mfcc(audio,sr, 0.025, 0.01, 13,appendEnergy = False)
    features = preprocessing.scale(features)
    return features

def printNow():
    now = datetime.datetime.now()
    print(str(now))

def training(gender):
    printNow()
    source = "input files/train/{}".format(gender)
    dest = "input files/Ky/model/train1/"

    print('training {} {}'.format(source,dest))
    files    = [os.path.join(source,f) for f in os.listdir(source)]

    features = np.asarray(());
    for f in files:
        print(f)
        data,rate = librosa.core.load(f, 16000)
        # rate, data = read(f)

        # print('audio', rate,data)
        vector = get_MFCC(rate,data)
        # print(vector)
        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))

    gmm = GMM(n_components = 8, n_iter = 200, covariance_type='diag',
            n_init = 3)
    gmm.fit(features)
    print(f.split("/")[-2])
    picklefile = f.split("/")[-2]+".gmm"
    print(dest + picklefile)

    # model saved as .gmm
    cPickle.dump(gmm,open(dest + picklefile,'wb'))
    printNow()


print('BEGIN TRAINING {}'.format(printNow()))
training('male_central')
training('male_north')
training('male_south')
print('COMPLETED TRAINING {}'.format(printNow()))
