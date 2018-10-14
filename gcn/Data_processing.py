import sklearn as skl
from sklearn.covariance import GraphLassoCV
import numpy as np
# import matplotlib.pyplot as plt
import sys
import csv

def data_reorder(Datadir):
    ### read fmri signal data (.npy) and DTI network data (matlab matrix)
    with open(Datadir+'miss_sub.txt','r') as f:
        filenames = f.read().splitlines()

    # Datadir = '/home/wen/Documents/gcn_kifp/Data/'

    fmri_signals = []
    DTI_connects = []
    for file in filenames:
        fmri_signal = np.load(Datadir+file+'_fmri.npy')
        fmri_signals.append(fmri_signal)
        DTI_connectivity = np.loadtxt(Datadir+file+'_fdt_matrix')

        ######################## need to fix, some subjects miss a brain region
        if DTI_connectivity.shape[0] == 246:
            DTI_connects.append(DTI_connectivity)

        ########################



    ### stack the data in the 3rd dimension
    fmri_signals=np.stack(fmri_signals,axis=2)
    DTI_connects=np.mean(np.stack(DTI_connects,axis=2),axis=2)
    DTI_connects[DTI_connects<50]=0
    DTI_connects = DTI_connects/(DTI_connects.sum(axis=1).transpose())
    DTI_connects = (DTI_connects.transpose()+DTI_connects)/2

    return fmri_signals, DTI_connects

def load_data(Datadir, labelscsv):  ### labels: a cvs file
    features, graph = data_reorder(Datadir)

    ### read cvs file [id, label]
    labels = []
    with open(labelscsv) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter = ',')
        line = 0
        for row in csv_reader:
            if line == 0:
                line = line +1
                continue
            labels.append(row[1])

    ### features: 3D array
    ### graph: 2D array
    ### labels: 1D list
    return features, graph, labels



if __name__ == '__main__':
    # data_reorder('/home/wen/Documents/gcn_kifp/Data/')
    load_data('../Data/', '../Data/labels.csv')