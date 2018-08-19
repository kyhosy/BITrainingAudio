#train_models.py

import os
import _pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GMM
import python_speech_features as mfcc
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")

def get_MFCC(sr,audio):
    features = mfcc.mfcc(audio,sr, 0.025, 0.01, 13,appendEnergy = False)
    features = preprocessing.scale(features)
    return features
# #path to training data
# source   = "D:\\pygender\\train_data\\youtube\\male\\"
# #path to save trained model
# dest     = "D:\\pygender\\"
#path to training data
# source   = "sources/train_data/male/_4oUfgt6Fzo_100-124-Male_0.wav"
def training(gender):
    source = "C:\\Users\\dell\\PycharmProjects\\BITrainingAudio\\sources\\train_data\\{}\\".format(gender)
    #path to save trained model
    dest = "C:\\Users\\dell\\PycharmProjects\\BITrainingAudio\\sources\\model\\"
    files    = [os.path.join(source,f) for f in os.listdir(source) if
                 f.endswith('.wav')]
    features = np.asarray(());

    for f in files:
        sr,audio = read(f)
        vector   = get_MFCC(sr,audio)
        print(vector)
        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))

    gmm = GMM(n_components = 8, n_iter = 200, covariance_type='diag',
            n_init = 3)
    gmm.fit(features)
    picklefile = f.split("\\")[-2].split(".wav")[0]+".gmm"
    print(dest + picklefile)

    # model saved as .gmm
    cPickle.dump(gmm,open(dest + picklefile,'wb'))
training('male')
training('female')