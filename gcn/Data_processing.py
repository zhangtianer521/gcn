import sklearn as skl
from sklearn.model_selection import train_test_split
import numpy as np
from random import sample
import csv
import pandas as pd

def data_reorder(Datadir):
    ### read fmri signal data (.npy) and DTI network data (matlab matrix)
    with open(Datadir+'DTI_passed_subj.txt','r') as f:
        filenames = f.read().splitlines()

    # Datadir = '/home/wen/Documents/gcn_kifp/Data/'

    fmri_signals = []
    DTI_connects = []
    for file in filenames:
        fmri_signal = np.load(Datadir+file+'.npy')
        fmri_signals.append(fmri_signal)
        # DTI_connectivity = np.loadtxt(Datadir+file+'_fdt_matrix')
        DTI_connectivity = np.load(Datadir+file+'_DTInetworks.npy')

        ######################## need to fix, some subjects miss a brain region
        if DTI_connectivity.shape[0] == 246:
            DTI_connects.append(DTI_connectivity)

        ########################



    ### stack the data in the 3rd dimension
    fmri_signals=np.stack(fmri_signals,axis=2)
    fmri_signals = np.transpose(fmri_signals,(2,1,0))
    DTI_connects=np.mean(np.stack(DTI_connects,axis=2),axis=2)
    DTI_connects[DTI_connects<50]=0
    DTI_connects = DTI_connects/(DTI_connects.sum(axis=1).transpose())
    DTI_connects = (DTI_connects.transpose()+DTI_connects)/2

    return fmri_signals, DTI_connects

def load_data(Datadir, labelscsv):  ### labels: a cvs file
    features, graph = data_reorder(Datadir)

    #normalize features
    # for ind in range(features.shape[0]):
    #     features[ind,:,:]*=255/features[ind,:,:].max()


    # read cvs file [id, label]
    labels = []
    with open(labelscsv) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter = ',')
        line = 0
        for row in csv_reader:
            if line == 0:
                line = line +1
                continue
            labels.append(row[1])

    # generate training and testing samples
    train_features, test_features, train_labels, test_labels = train_test_split(features,labels,test_size=0.2)


    # labels = np.asarray(labels)
    # indices = sample(range(features.shape[0]),int(features.shape[0]*0.8))
    # train_features = features[indices]
    # test_features = np.delete(features,indices)
    # train_labels = labels[indices]
    # test_labels = np.delete(labels,indices)


    ### features: 3D array, [#subjects, #nodes, #feature_per_node]
    ### graph: 2D array
    ### labels: 1D list
    return train_features, [graph], one_hot(train_labels), test_features, one_hot(test_labels)

def one_hot(labels):
    s = pd.Series(labels)
    return pd.get_dummies(s)


def creat_csv_Autism(idtxt, allcsvlist,outcsv):
    with open(idtxt,'r') as f:
        filenames = f.read().splitlines()

    with open(outcsv,'w') as outf:
        spamwriter = csv.writer(outf,delimiter=',')
        spamwriter.writerow(['id','labels'])
        for file in filenames:
            for csv_s in allcsvlist:
                with open(csv_s) as f_csv:
                    csv_reader = csv.reader(f_csv,delimiter = ',')
                    line = 0
                    for row in csv_reader:
                        if line ==0:
                            line +=1
                            continue
                        if file in row:
                            spamwriter.writerow([file ,row[3]])

def creat_csv_Schiz(idtxt, allcsvlist,outcsv):
    with open(idtxt,'r') as f:
        filenames = f.read().splitlines()

    with open(outcsv,'w') as outf:
        spamwriter = csv.writer(outf,delimiter=',')
        spamwriter.writerow(['id','labels'])
        for file in filenames:
            for csv_s in allcsvlist:
                with open(csv_s) as f_csv:
                    csv_reader = csv.reader(f_csv,delimiter = ',')
                    line = 0
                    for row in csv_reader:
                        if line ==0:
                            line +=1
                            continue
                        if file in row:
                            if 'No_Known_Disorder' in row[4]: label = 1
                            else: label = 2
                            spamwriter.writerow([file ,label])



if __name__ == '__main__':
    # data_reorder('/home/wen/Documents/gcn_kifp/Data/')
    # load_data('../Data/', '../Data/labels.csv')

    # # Autism
    # csv_dir = '/mnt/wzhan139/Image_Data/Network_data/Autism/'
    # csvfiles = [csv_dir + 'ABIDEII-BNI_1.csv',
    #             csv_dir + 'ABIDEII-IP_1.csv',
    #             csv_dir + 'ABIDEII-NYU_1.csv',
    #             csv_dir + 'ABIDEII-NYU_2.csv',
    #             csv_dir + 'ABIDEII-SDSU_1.csv',
    #             csv_dir + 'ABIDEII-TCD_1.csv']
    # creat_csv_Autism('/home/local/ASUAD/wzhan139/Dropbox (ASU)/Project_Code/GCN_kipf/Data/Autism/DTI_passed_subj.txt',
    #           csvfiles, '/home/local/ASUAD/wzhan139/Dropbox (ASU)/Project_Code/GCN_kipf/Data/Autism/labels.csv')

    # Schiz
    csvfiles = ['/mnt/easystore_8T/Wen_Data/Schizophrenia/deprecated_schizconnect_metaData_3640.csv']
    creat_csv_Schiz('/home/local/ASUAD/wzhan139/Dropbox (ASU)/Project_Code/GCN_kipf/Data/Schiz/DTI_passed_subj.txt',
              csvfiles, '/home/local/ASUAD/wzhan139/Dropbox (ASU)/Project_Code/GCN_kipf/Data/Schiz/labels.csv')