import sklearn as skl
from sklearn.covariance import GraphLassoCV
import numpy as np
import matplotlib.pyplot as plt
import sys

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
        DTI_connects.append(DTI_connectivity)

    ### stack the data in the 3rd dimension
    fmri_signals=np.stack(fmri_signals,axis=2)
    DTI_connects=np.stack(DTI_connects,axis=2)

    return fmri_signals, DTI_connects


if __name__ == '__main__':
    data_reorder('/home/wen/Documents/gcn_kifp/Data/')
