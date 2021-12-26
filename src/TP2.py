from tp2_aux import images_as_matrix
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif
from pandas.plotting import scatter_matrix


def extract_features(data):
    pca = PCA(n_components=6)
    x_pca = pca.fit_transform(data)
    print('Shape of data after PCA:', x_pca.shape)

    tsne = TSNE(n_components=6, method='exact')
    x_tsne = tsne.fit_transform(data)
    print('Shape of data after TSNE:', x_tsne.shape)
    
    isomap = Isomap(n_components=6)
    x_isomap = isomap.fit_transform(data)
    print('Shape of data after Isomap:', x_isomap.shape)

    extracted_features = np.concatenate((x_pca,x_tsne,x_isomap), axis=1)

    return extracted_features

def heatmap(features):
    """ 
    Another method to check the correlation between features
    """
    sns.heatmap(
    abs(features), 
    annot=True, 
    cmap=plt.cm.Reds)
    plt.savefig("heatmap.png")
    plt.figure(figsize=(12,10))
    plt.show()
    plt.close()

def scatter_plot(features):
    """ 
    Another method to check the correlation between features
    """
    plt.figure()
    scatter_matrix(
        features, 
        alpha=0.5,
        figsize=(15,10), 
        diagonal='kde')
    plt.savefig("scatter_plot.png")
    plt.show()  
    plt.close()

def select_features(features):

    # Remove features that are statistically independent of the class 
    labels = np.loadtxt('labels.txt', delimiter=',', usecols = 1)

    labeled_features = features[labels!=0,:] 
    nonzerolabels=labels[labels!=0]

    f,prob = f_classif(labeled_features, nonzerolabels)
    dependent_vars=np.where(prob < 0.01)[0].tolist()
    s_features = features[:,dependent_vars]

    df = pd.DataFrame(s_features)

    #Pearson correlation matrix of selected features
    cor = df.corr()

    # Return 1, 2, 3 and 4 because 0 and 3 are correlated and one of them is thus redundant so we skip 0 because it had the highest probability
    return s_features[:,[1, 2, 3, 4]]

def main():
    data = images_as_matrix()

    features_already_extracted = True
    # 1. Feature extraction using 3 different methods each extracting 6 features for total of 18 features
    if not features_already_extracted:
        extracted_features = extract_features(data)
        np.savetxt('features.txt', extracted_features)

    extracted_features = np.loadtxt('features.txt')

    # 2. Feature selection
    s_features = select_features(extracted_features)

    # 3. Clustering algorithms
    # 4. Evaluation ? 

if __name__ == '__main__':
    main()

