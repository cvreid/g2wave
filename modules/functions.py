import os
import re
import math
import random


import numpy as np
from gwpy.timeseries import TimeSeries



bdir = os.path.dirname(os.path.dirname(__file__))

def load_training_labels(fname="/training_labels.csv"):
    labels = {}
    with open(bdir + fname) as fh:
        lines = fh.readlines()
        for l in lines[1:]:
            arr = l.strip().split(",")
            labels[arr[0]] = int(arr[1])
    return labels

def load_training_batch(labels, add_dir):
    """
    Load a batch of training objects (one of the subfolders from training)

    @param
    labels: dict mapping id (in filename) to class (1 or 0).  

    @return
    data: np.arr 4 dim
    classes: np.arr 1 dim
    """
    
    num_files = len(os.listdir(bdir + "/" + add_dir))
        
    data = np.empty((num_files, 3, 4096)) # 3 signals, 2 seconds @ 2048 Hz
    
    classes = np.zeros((num_files,)) # Might as well default to 0
    #add dir sample like 0/0/
    counter = 0
    for f in os.listdir(bdir + "/" + add_dir):
        idx = f.split(".")[0]
        if idx in labels:

            data[counter] = np.load(bdir + "/" +add_dir +  f)
            classes[counter] = labels[idx]
        else:
            print("Missing label?", idx)
        counter += 1

    #print("Shapes: ", data.shape, classes.shape)
    return data, classes

def prepare_signal(signal, low_freq=35, high_freq=400, band = False):
    """
    Prepare the singal -- tukey window for whitening the spikes, ensuring no noise from incorrect window sample, etc.
    Bandpass -- makes the assumption that extreme frequencies are noise. Might be right, experiment with and without
    """
    whitened = signal.whiten(window=("tukey", 0.2)) #This is similar to what happens with window=None in spectrogram
    if band:
        banded = whitened.bandpass(low_freq, high_freq) #I might remove this tbh, dunno if anomalies happen here
        return banded
    return whitened

def q_transform(signal, qlow=16, qhigh=32, freqlow=30, freqhigh=450):
    """
    Run a q transform if not using wavelet transform inherently in model.
    """
    
    return signal.q_transform(qrange=(qlow, qhigh), frange=(freqlow, freqhigh), logf=True, whiten=False)


def data_to_image(data, bandpass=False, low_freq=35, high_freq=400):
    """
    Convert data to 3 separate 'images' (q_trans spectrogram like reps of signal) for convo net
    """
    
    #Setting shape to what I know the qtrans output to be...do this programatically later
    out_data = np.empty((data.shape[0], 3, 1000, 500))
    
    for i in range(data.shape[0]):
        #if i%300 == 0:
            #print("Fin:", i, data.shape[0])
        for j in range(3):
            #print("??", data[i][j].shape)
            out_data[i][j] = q_transform(prepare_signal(TimeSeries(data[i][j], sample_rate=2048)), freqlow=low_freq, freqhigh=high_freq)
    return out_data
    
    
    
    
    
    