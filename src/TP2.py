from tp2.tp2_aux import images_as_matrix
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap

def main():
    arr = images_as_matrix()
    print(arr)
    print(arr.shape)

    # Feature extraction using 3 different methods 
    # Clustering algorithms 
    # Evaluation ?? 

if __name__ == '__main__':
    main()

