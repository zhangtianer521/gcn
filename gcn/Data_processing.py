import nilearn as nl
from nilearn.input_data import NiftiLabelsMasker
import nilearn.connectome as connectome
import sklearn as skl
from sklearn.covariance import GraphLassoCV
import numpy as np
import matplotlib.pyplot as plt
import sys

################## warnings
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
##################

with open(r'/mnt/easystore_8T/Wen_Data/Schizophrenia/subjects.txt','r') as f:
    filenames = f.read().splitlines()

dir = r'/mnt/easystore_8T/Wen_Data/Schizophrenia/COBRE_fmri/'

for file in filenames:

    print('Running subject '+file+' *******************')

    fmri_template = dir + 'sub-' + file +'_bold_2.feat/warp_func_atlas.nii.gz'
    fmri_image = dir + 'sub-' + file +'_bold_2.feat/filtered_func_data.nii.gz'

    masker = NiftiLabelsMasker(labels_img=fmri_template,standardize=True)
    time_series = masker.fit_transform(fmri_image)
    np.save(dir+'fmri_ROI_signal/'+file, time_series)

    fig, axs = plt.subplots(1,2)

    # print('correlation using GraphLassoCV')
    # estimator = GraphLassoCV()
    # graph = estimator.fit(time_series)
    # im1 = axs[0].matshow(graph.covariance_)
    # axs[0].set_title('GraphLassoCV')
    # plt.colorbar(im1)
    im1 = axs[0].matshow(time_series)
    axs[0].set_title('Signals')
    plt.colorbar(im1)

    print('correlation using corrf')
    estimator = connectome.ConnectivityMeasure()
    graph = estimator.fit_transform([time_series])[0]
    im2 = axs[1].matshow(graph)
    axs[1].set_title('covariance')
    plt.colorbar(im2)

    # print('correlation using partial_corrf')
    # estimator = connectome.ConnectivityMeasure(kind='partial correlation')
    # graph = estimator.fit_transform([time_series])[0]
    # plt.matshow(graph)
    # plt.colorbar()
    # plt.title('covariance')

    fig.savefig(dir+'fmri_ROI_signal_connect_image/'+file+'.png')
    plt.close(fig)
