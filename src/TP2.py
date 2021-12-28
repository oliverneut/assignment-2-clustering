from re import S
from tp2_aux import images_as_matrix, report_clusters
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif
from pandas.plotting import scatter_matrix
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier


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

def select_features(features, labels):
    means = np.mean(features,axis=0)
    std = np.std(features,axis=0)
    features= (features-means)/std

    # Remove features that are statistically independent of the class 
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

def confusion_matrix(labels, pred):
    distances_labels=pairwise_distances(labels.reshape(-1,1),labels.reshape(-1,1),metric="hamming")
    distances_predictions=pairwise_distances(pred.reshape(-1,1),pred.reshape(-1,1),metric="hamming")
    TN=sum(distances_predictions[distances_labels==1])/2
    FP=(len(distances_predictions[distances_labels==1])-sum(distances_predictions[distances_labels==1]))/2
    TP=(len(distances_predictions[distances_labels==0])-distances_predictions.shape[0]-sum(distances_predictions[distances_labels==0]))/2
    FN=sum(distances_predictions[distances_labels==0])/2
    return(TP, TN, FP, FN)

def metric_estimation(s_features, pred, labeled_pred, nonzerolabels):
    # Compute performance (internal index)
    silhouette = silhouette_score(s_features, pred)

    # Compute performance (external index)
    TP, TN, FP, FN = confusion_matrix(labeled_pred, nonzerolabels)

    rand_index = (TP + TN) / (TP + TN + FP + FN)

    precision = (TP)/(TP+FP)

    recall = (TP)/(TP+FN)

    F1 = 2*precision*recall/(precision+recall)

    adjusted_RS = adjusted_rand_score(labels_true=nonzerolabels, labels_pred=labeled_pred)
    
    return (silhouette, rand_index, precision, recall, F1, adjusted_RS)


def k_means_clustering(s_features, labels, k):
    model_kmeans = KMeans(n_clusters=k)
    pred = model_kmeans.fit_predict(s_features)

    labeled_pred = pred[labels!=0]
    nonzerolabels = labels[labels!=0]

    return metric_estimation(s_features, pred, labeled_pred, nonzerolabels)

def plot_distances(data, num_neighbors):
    neighbours = KNeighborsClassifier(n_neighbors=num_neighbors).fit(data, np.ones(563))
    distances = neighbours.kneighbors()
    distances=distances[0]
    # print(distances)
    print('distances: ')
    sorted_distances = sorted(distances[:,num_neighbors-1], reverse=True)
    plt.figure(figsize=(8,6))
    plt.scatter(
        range(1,data.shape[0]+1),
        sorted_distances,
        s=0.1)
    plt.xlabel("Data points")
    plt.ylabel("Distance to 5th neighbour")
    plt.savefig(f'distances_plot_epsilon.png')
    plt.show()
    plt.close()

def dbscan_clustering(s_features, labels, eps):
    model_kmeans = DBSCAN(eps=eps)
    pred = model_kmeans.fit_predict(s_features)

    labeled_pred = pred[labels!=0]
    nonzerolabels = labels[labels!=0]

    return metric_estimation(s_features, pred, labeled_pred, nonzerolabels)

def plot_metrics(range, isKMeans:bool, labels, s_features):
    silhouettes = []
    rand_idxs = []
    precisions = []
    recalls = []
    F1s = []
    adjusted_RSs = []
    total = []
    for i in range:
        if(isKMeans):
            (silhouette, rand_index, precision, recall, F1, adjusted_RS) = k_means_clustering(s_features, labels, k=i)
        else:
            (silhouette, rand_index, precision, recall, F1, adjusted_RS) = dbscan_clustering(s_features, labels, eps=i)
        silhouettes.append(silhouette)
        rand_idxs.append(rand_index)
        precisions.append(precision)
        recalls.append(recall)
        F1s.append(F1)
        adjusted_RSs.append(adjusted_RS)

    plt.plot(range,silhouettes, linewidth=1.0,label="silhouette score")
    plt.plot(range,rand_idxs, linewidth=1.0,label="Rand index")
    plt.plot(range,F1s, linewidth=1.0, label='F1 score')
    plt.plot(range,adjusted_RSs, linewidth=1.0, label='adjusted rand score')
    plt.plot(range,precisions, linewidth=1.0, label='precision')
    plt.plot(range,recalls, linewidth=1.0, label= 'recall')
    plt.ylabel("Scores")
    plt.legend()
    if(isKMeans):
        plt.xlabel("K")
        plt.title("K_Means_clustering")
        plt.savefig("K_Means_clustering.png")
    else: 
        plt.xlabel("Eps")
        plt.title("DBSCAN_clustering")
        plt.savefig("DBSCAN_clustering.png")
    plt.show()
    plt.close()
    plt.show()

def main():
    data = images_as_matrix()

    features_already_extracted = True  # toggle this to false to extract features
    # 1. Feature extraction using 3 different methods each extracting 6 features for total of 18 features
    if not features_already_extracted:
        extracted_features = extract_features(data)
        np.savetxt('features.txt', extracted_features)

    extracted_features = np.loadtxt('features.txt')

    #get the labels
    labels = np.loadtxt('labels.txt', delimiter=',', usecols = 1)

    # 2. Feature selection
    s_features = select_features(extracted_features, labels)

    # 3. Clustering algorithms

    # loop over different values of k
    K_range = np.arange(2,12,1)
    plot_metrics(K_range, True, labels=labels, s_features=s_features)

    # We find a optimal epsilon value at 0.85 using the following function
    #plot_distances(s_features, 5)

    Eps_range = np.arange(0.3, 1, 0.05)
    plot_metrics(Eps_range, False, labels=labels, s_features=s_features)
    dbscan_clustering(s_features, labels, eps=0.85)

    ids = np.loadtxt('labels.txt', delimiter=',', usecols = 0)

    # KMeans clustering using k = 4
    k = 4
    model_kmeans = KMeans(n_clusters=k)
    pred = model_kmeans.fit_predict(s_features)
    report_clusters(ids, pred, f'KMeans_k={k}.html')

    # DBSCAN clustering using eps = 0.5
    eps = 0.55
    model_dbscan = DBSCAN(eps=eps)
    pred = model_dbscan.fit_predict(s_features)
    report_clusters(ids, pred, f'DBSCAN_eps={eps}.html')

    # Aglomerative clustering using n = 4
    n = 4
    model_agglomerative = AgglomerativeClustering(n_clusters=n)
    pred = model_agglomerative.fit_predict(s_features)
    report_clusters(ids, pred, f'AgglomerativeClustering_n={n}.html')



if __name__ == '__main__':
    main()

