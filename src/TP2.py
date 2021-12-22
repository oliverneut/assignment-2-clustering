from tp2_aux import images_as_matrix
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap

def extract_features(data):
    pca = PCA(n_components=6)
    x_pca = pca.fit_transform(data)
    print('Shape of data after PCA:', x_pca.shape)
    np.savez('pca_data', x_pca)

    tsne = TSNE(n_components=6, method='exact')
    x_tsne = tsne.fit_transform(data)
    print('Shape of data after TSNE:', x_tsne.shape)
    np.savez('tsne_data', x_tsne)
    
    isomap = Isomap(n_components=6)
    x_isomap = isomap.fit_transform(data)
    print('Shape of data after Isomap:', x_isomap.shape)
    np.savez('isomap_data', x_isomap)

def select_features(data):
    print('Select features -------')

def main():
    data = images_as_matrix()
    print('Initial shape of data :', data.shape)

    # 1. Feature extraction using 3 different methods each extracting 6 features for total of 18 features
    extracted_features = extract_features(data)

    # 2. Feature selection
    s_features = select_features(extracted_features)

    # 3. Clustering algorithms
    # 4. Evaluation ? 

if __name__ == '__main__':
    main()

