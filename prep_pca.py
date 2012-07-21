"""
Aaron Berndsen

Try decomposing the data into pca first

"""
#from sklearn.cross_validation import train_test_split
#from sklearn.grid_search import GridSearchCV
#from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC
import numpy as np

def pca_decomp(dftr, ncomp=120):
    """
    decompose data

    """
    n_features = np.array(dftr.data).shape[1]
    n_classes = np.unique(dftr.target)

    n_components = 80
    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(dftr.train_data)
    #hardcoded to 32x32 features
    h = 32
    w = 32
    eigenfeat = pca.components_.reshape((n_components, h, w))

    dftr.train_data_pca = pca.transform(dftr.train_data)
    dftr.test_data_pca = pca.transform(dftr.test_data)

