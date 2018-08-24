#test_gender.py
import pandas
import datetime
import os
import _pickle as cPickle
import numpy as np
import librosa
import python_speech_features as mfcc
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")
def printNow():
    now = datetime.datetime.now()
    print(str(now))

def get_MFCC(sr,audio):
    features = mfcc.mfcc(audio,sr, 0.025, 0.01, 13,appendEnergy = False)
    feat     = np.asarray(())
    for i in range(features.shape[0]):
        temp = features[i,:]
        if np.isnan(np.min(temp)):
            continue
        else:
            if feat.size == 0:
                feat = temp
            else:
                feat = np.vstack((feat, temp))
    features = feat;
    features = preprocessing.scale(features)
    return features

def test():
    printNow()
    csvTest = 'input files/Ky/test_metadata.csv'
    df = pandas.read_csv(csvTest)
    sourcepath = "input files/test/"
    modelpath = "input files/Ky/model/"


    gmm_files = [os.path.join(modelpath,fname) for fname in
                  os.listdir(modelpath) if fname.endswith('.gmm')]
    models    = [cPickle.load(open(fname,'rb')) for fname in gmm_files]
    genders   = [fname.split("/")[-1].split(".gmm")[0] for fname
                  in gmm_files]
    maxText = 50
    # print(genders)
    files     = [os.path.join(sourcepath,f) for f in df["fname"][:maxText]]

    index = 0
    for f in files:
        print(f.split("/")[-1])
        audio, sr = librosa.core.load(f, 16000)
        features   = get_MFCC(sr,audio)
        scores     = None
        log_likelihood = np.zeros(len(models))
        # scoreStr = ' score:'
        for i in range(len(models)):
            gmm    = models[i]         #checking with each model one by one
            scores = np.array(gmm.score(features))
            log_likelihood[i] = scores.sum()
            # scoreStr += genders[i] + "->" + log_likelihood[i] + ','
        winner = np.argmax(log_likelihood)

        # print("winner>>>",winner,log_likelihood)

        print(f.split("/")[-1],":detected as - ", genders[winner],"\n")
        df.loc[index, "test"] = genders[winner]
        index = index + 1


    df.to_csv("input files/Ky/result_training.csv",index=False, encoding='utf8')


print('BEGIN TEST {}'.format(printNow()))
test()
print('COMPLETED TEST {}'.format(printNow()))
