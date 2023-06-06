import numpy as np
import os
import soundfile as sf
import librosa
import librosa.display
from lab3_tools import *
from lab3_proto import *
from lab2_tools import *
from lab2_proto import *
from lab1_proto import *
from lab1_tools import *
from scipy.io import wavfile


# This function is used to read the data from the file

phenomenoes = [
    "sil","b","eh","d","er","t","ae","g","ao","n","aw","f","ay","v","r","hh","p","iy","s","l","ih","k","sh","m","ah","w","z","uw","y","ow"
    
]
sub_word_units = {
    "bed": ["sil","b", "eh", "d","sil"],
    "bird": ["sil","b", "er", "d","sil"],
    "cat": ["sil","k", "ae", "t","sil"],
    "dog": ["sil","d", "ao", "g","sil"],
    "down": ["sil","d", "aw", "n","sil"],
    "eight": ["sil","ey", "t","sil"],
    "five": ["sil","f", "ay", "v","sil"],
    "four": ["sil","f", "ao", "r","sil"],
    "go": ["sil","g", "ow","sil"],
    "happy": ["sil","hh", "ae", "p","iy","sil"],
    "house": ["sil","hh", "aw", "s","sil"],
    "left": ["sil","l", "eh", "f","t","sil"],
    "marvin": ["sil","m", "aa", "r","v","ih","n","sil"],
    "nine": ["sil","n", "ay", "n","sil"],
    "no": ["sil","n", "ow","sil"],
    "off": ["sil","ao", "f","sil"],
    "on": ["sil","ao", "n","sil"],
    "one": ["sil","w", "ah", "n","sil"],
    "right": ["sil","r", "ay", "t","sil"],
    "seven": ["sil","s", "eh", "v","ah","n","sil"],
    "sheila": ["sil","sh", "iy", "l","ah","sil"],
    "six": ["sil","s", "ih", "k","s","sil"],
    "stop": ["sil","s", "t","aa","p","sil"],
    "three": ["sil","th", "r","iy","sil"],
    "tree": ["sil","t", "r","iy","sil"],
    "two": ["sil","t", "uw","sil"],
    "up": ["sil","ah", "p","sil"],
    "wow": ["sil","w", "aw","sil"],
    "yes": ["sil","y", "eh", "s","sil"],
    "zero": ["sil","z", "ih", "r","ow","sil"]

}

def loadAudio(filename):
    """
    loadAudio: loads audio data from file using pysndfile

    Note that, by default soundfile converts the samples into floating point
    numbers and rescales them in the range [-1, 1]. This is avoided by specifying
    the option dtype=np.int16 which keeps both the original data type and range
    of values.
    """
    return sf.read(filename, dtype='int16')

def getData():
    fileNames=["bed","bird","cat","dog","down","eight","five","four","go","happy","house","left","marvin","nine","no","off","on","one","right","seven","sheila","six","stop","three","tree","two","up","wow","yes","zero"]
    data={}
    #read the data from the file speech_commands_v0.01 where every file is named like the word it contains
    path="speech_commands_v0.01/"
    for fileName in fileNames:
        data[fileName]=[]
        for file in os.listdir(path+fileName):
            print(path+fileName+"/"+file)
            if file.endswith(".wav"):
                sample, samplingrate = librosa.load(path+fileName+"/"+file)
                mfcc=librosa.feature.mfcc(y=sample,sr=samplingrate,n_mfcc=13)
                spec=librosa.feature.melspectrogram(y=sample,sr=samplingrate,n_mels=40)
                data[fileName].append({"sample":sample,"samplingrate":samplingrate,"lmfcc":mfcc,"mspec":spec})
    #pad the lmfcc with zeros to have the same lenght
    # for fileName in fileNames:
    #     for i in range(len(data[fileName])):
    #         data[fileName][i]["lmfcc"]=np.pad(data[fileName][i]["lmfcc"], ((0,max_lenght_lmfcc-data[fileName][i]["lmfcc"].shape[0]),(0,0)), 'constant', constant_values=0)
     #write to a file data.txt the data in the format: word, sample, samplingrate, lmfcc
    np.savez('data.npz', data=data)
    return data
    

def padding(data):
    for d in data:
        for s in range(len(d)):
            if data[d][s]['mspec'].shape[1]<44:
                 #pad and make all 128x44
                data[d][s]['mspec']=np.pad(data[d][s]['mspec'], ((0,0),(0,44-data[d][s]['mspec'].shape[1])), 'constant', constant_values=0)
            if data[d][s]['lmfcc'].shape[1]<44:
                data[d][s]['lmfcc']=np.pad(data[d][s]['lmfcc'], ((0,0),(0,44-data[d][s]['lmfcc'].shape[1])), 'constant', constant_values=0)
    return data


#create a function that add padding if the shape[1] is less than 44
def padding(mspec):
    return np.pad(mspec, ((0,0),(0,44-mspec.shape[1])), 'constant', constant_values=0)
#   



def getDataset(data):
    size=0
    for d in data:
        size+=len(data[d])
    dataset=np.zeros((size,13,44))
    i=0
    for word, samples in data.items():
        for i,sample in enumerate(samples):
            mspec = sample["lmfcc"]
            if mspec.shape[1]<44:
                mspec=padding(mspec)
            dataset[i]=mspec
            #check if dataset[i] is all 0 or there ar some Nan values
            if np.isnan(dataset[i]).any():
                print("Nan")
            if np.all(dataset[i]==0):
                print("all 0")
                i-=1

            #check if dataset[i] is all 0
            i+1 
    return dataset


               
