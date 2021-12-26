#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 10:32:54 2020

@author: duartesilveira
"""

#Import necessary libraries
import numpy as np
import pandas as pd
import tp2_aux as aux
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.feature_selection import f_classif
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from pandas.plotting import scatter_matrix
from sklearn.cluster import DBSCAN, SpectralClustering,KMeans,AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics import pairwise_distances
from numpy import genfromtxt

#Set seed 
np.random.seed(42)

#-------------------------------FEATURE EXTRACTION---------------------------------------------------------

#Transform the images in the images folder in a 2D numpy array with on image per row and one pixel per column
data=aux.images_as_matrix()

#Extract 6 features using Principal Component Analysis
PCA_features = PCA(n_components=6).fit_transform(data)

#Extract 6 features using t-Distributed Stochastic Neighbor Embedding
TSNE_features = TSNE(n_components=6,method="exact").fit_transform(data)

#Extract 6 features using Isometric mapping with Isomap
ISOMAP_features = Isomap(n_components=6).fit_transform(data)

#Save the 18 extracted features into one feature matrix 
matrix=np.concatenate((PCA_features,TSNE_features,ISOMAP_features), axis=1)
np.savez('featureextration.npz',matrix)

#-------------------------------FEATURE SELECTION---------------------------------------------------------

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

def heatmap(features):
    """ 
    Another method to check the correlation between features
    """
    sns.heatmap(
    abs(features), 
    annot=True, 
    cmap=plt.cm.Reds)
    plt.figure(figsize=(12,10))
    plt.savefig("heatmap.png")
    plt.show()
    plt.close()

#LOAD THE 18 FEATURES
file=np.load('featureextration.npz')
features=file['arr_0']

#STANDARDIZE THE FEATURES
means = np.mean(matrix,axis=0)
std = np.std(matrix,axis=0)
features= (matrix-means)/std

#LOAD LABELS REALTIVE TO CELL CYCLE PHASE (0 NOT LABELED)
labels=np.loadtxt('labels.txt', delimiter=',')[:,1]

#GET THE LABELED DATA
labeled=features[labels!=0,:]
nonzerolabels=labels[labels!=0]

#USE ANOVA F-test, THAT ESTIMATES THE DEGREE OF LINEAR DEPENDENCY BETWEEN EACH FEATURE AND THE LABELS.
#RETAIN THE FEATURES FOR WHICH THE HYPOTHESIS OF INDEPENDENCE WITH THE LABELS IS REJECTED AT 1% SIGNIFICANCE LEVEL.
f,prob = f_classif(labeled,nonzerolabels)
features_Findexes=np.where(prob < 0.01)[0].tolist()
order=prob[features_Findexes].argsort().tolist()
features_Findexes=np.array(features_Findexes)[order].tolist()
features_after_F=features[:,features_Findexes]

#CHECK FOR CORRELATIONS WITHIN THE OBTAINED FEATURES

#CONVERT FEATURES INTO A DATAFRAME
df = pd.DataFrame(features_after_F)

#SCATTER PLOT (CHECK FOR NON-LINEAR CORRELATIONS)
scatter_plot(df)

#PEARSON CORRELATION MATRIX (LINEAR CORRELATIONS)
cor = df.corr()

#HEATMAP
heatmap(cor)

#CHOOSE FEATURES THAT ARE NOT CORRELATED TO OTHER IN THE FINAL SET (PRIVILEGE TO HIGHER F-VALUES)
final_features=features_after_F[:,[0,2,3,5]]





#----------------------OBTAIN THE CLUSTERS USING THE PREVIOUS FEATURES-----------------

def get_labeled_data(labels,pred):
    """Get labeles
    Returns:
    Labels
    """
    return labels[labels!=0],pred[labels!=0]


def confusion_matrix(labels,pred):
    """ Returns the numbers of true positives, true
    negatives, false positives and false negatives

    Returns:
    TP - number of points which are in the same cluster and have the same label
    TN - number of points which are in different clusters and have diferent labels
    FP - number of points which are in the same cluster and have the different labels
    FN - number of points which are in different cluster and have the same label
    """
    matrix_labels=pairwise_distances(labels.reshape(-1,1),labels.reshape(-1,1),metric="hamming")
    matrix_pred=pairwise_distances(pred.reshape(-1,1),pred.reshape(-1,1),metric="hamming")
    TN=sum(matrix_pred[matrix_labels==1])/2
    FP=(len(matrix_pred[matrix_labels==1])-sum(matrix_pred[matrix_labels==1]))/2
    TP=(len(matrix_pred[matrix_labels==0])-matrix_pred.shape[0]-sum(matrix_pred[matrix_labels==0]))/2
    FN=sum(matrix_pred[matrix_labels==0])/2 
    return(TP, TN, FP, FN)


def RI(labels,pred):
    """ Funtion that returns the rand index. The Rand index measure the
    similarity between the clusters.
    Returns:
    Rand_index
    """
    TP,TN,FP,FN = confusion_matrix(labels,pred)
    RI = (TP+TN)/(TP+FP+FN+TN)
    return RI

def precision(labels,pred):
    """ Funtion that returns the precision. The precision is the ability of 
    the classifier to label as positive a sample that is positive.
    Returns:
    precision
    """
    TP,TN,FP,FN = confusion_matrix(labels,pred)
    precision = (TP)/(TP+FP)
    return precision

def recall(labels,pred):
    """ Funtion that returns the recall. The recall is the ability of the 
    classifier to find all positives.
    Returns:
    recall
    """
    TP,TN,FP,FN = confusion_matrix(labels,pred)
    recall = (TP)/(TP+FN)
    return recall

def F1(labels,pred):
    """ Funtion that returns the F1 metric. F1 metric is a weighted average 
    of the precision and recall.
    Returns:
    F1 metric
    """
    prec=precision(labels,pred)
    rec=recall(labels,pred)
    F1=2*prec*rec/(prec+rec)
    return F1


def method_performance(features, labels,range_values,method):
    """
    Function that returns all the metrics for Kmeans and DBSCAN algorithms
    (or others) varying the k clusters and eps.
    
    Parameters:
    features - Array with the selected final features.
    labels: Actual labels from the biologists
    range_values: range of value to train and get results from different metrics
    method: Method in study 
    
    Returns:
    silhouette_scores - All values of the Silhouette Score varying the parameter to be optimized. 
    rand_ixs - All values of the Rand index varying the parameter to be optimized. 
    f1_scores - All values of the F1 Score varying the parameter to be optimized. 
    precisions -  All values of the Precision varying the parameter to be optimized. 
    recalls -  All values of the Recall varying the parameter to be optimized. 
    adjusted_rand_ixs - All values of the Adjusted Rand Index varying the parameter to be optimized. 
    """
    
    silhouette_scores =[]
    f1_scores = []
    rand_ixs=[]
    precisions =[]
    recalls= []
    adjusted_rand_ixs=[]
    range_values=range_values
    for i in range_values:  
        if method == "KMEANS":
            model = KMeans(n_clusters=i)     
        elif method == "DBSCAN":
            model = DBSCAN(eps=i) 
        elif method == "AgglomerativeClustering":
            model = AgglomerativeClustering(n_clusters=i)  
        elif method == "GaussianMixture":
            model = GaussianMixture(n_components=i)
        else:
            model = method(i)
        pred = model.fit_predict(features)
        labelled_labels,labelled_preds = get_labeled_data(labels,pred)
        silhouette_scores.append(silhouette_score(features,pred))  
        rand_ixs.append(RI(labelled_labels,labelled_preds))
        f1_scores.append(F1(labelled_labels,labelled_preds))
        precisions.append(precision(labelled_labels,labelled_preds))
        recalls.append(recall(labelled_labels,labelled_preds))
        adjusted_rand_ixs.append(adjusted_rand_score(labelled_labels,labelled_preds))
        
    return range_values,silhouette_scores,rand_ixs,f1_scores,precisions,recalls,adjusted_rand_ixs

    
def plot_measures(range_values,method,xlabel):
    """
    Function to observe the data provided by the metrics and the range of the 
    parameters to be optimized.
    Parameters:
    range_values - interval of different values of the parameter
    method - the method in study
    xlabel: parameter to be optimized
    """
    range_values = range_values
    plt.plot(range_values,silhouette_scores, linewidth=1.0,label="silhouette_scores")
    plt.plot(range_values,rand_ixs, linewidth=1.0,label="Rand Scores")
    plt.plot(range_values,f1_scores, linewidth=1.0, label='F1')
    plt.plot(range_values,adjusted_rand_ixs, linewidth=1.0, label='adjusted random scores')
    plt.plot(range_values,precisions, linewidth=1.0, label='precisions')
    plt.plot(range_values,recalls, linewidth=1.0, label= 'recalls')
    plt.xlabel(f"{xlabel}")
    plt.ylabel("Scores")
    plt.title(f"{method}")
    plt.legend()
    plt.savefig(f"{method}.png")
    plt.show()
    plt.close()
    plt.show()

def clusters_report(best_values, method):
    """
    Function to report the html using the auxiliar report_clusters 
    to visually see the clusters provided by the method in study.
    Parameters:
    best_values - interval of the best values of the parameter observed using the plot.
    method - the method in study
    """
    ids = genfromtxt('labels.txt', delimiter=',')[:,0]

    for i in best_values:
        if method == "KMEANS":
            model = KMeans(n_clusters=i)     
        elif method == "DBSCAN":
            model = DBSCAN(eps=i)    
        elif method == "GaussianMixture":
            model = GaussianMixture(n_components=i)
        elif method == "AgglomerativeClustering":
            model = AgglomerativeClustering(n_clusters=i)  
        else:
             model = method(i)
        prediction = model.fit_predict(final_features) 
        aux.report_clusters(ids, prediction, f'{method}_{i}.html')
    
#----------------------------K MEANS------------------------------------------#

#EXAMINING THE PERFORMANCE OF KMEANS FOR DIFFERENT VALUES OF K
#FIRST WE SET A RANGE OF K
np.random.seed(67)
k_clusters = np.arange(4,13,1)
#CALCULATE THE METRICS 
[range_values,silhouette_scores,rand_ixs,f1_scores,precisions,recalls,adjusted_rand_ixs]=method_performance(final_features,labels, k_clusters,"KMEANS" )
#PLOT THE MEASURES FOR THE DIFFERENT VALUES OF K
plot_measures(range_values,"KMEANS","Clusters")

#REPORT THE CLUSTERS 
np.random.seed(67)
best_values = [7,9,11]
clusters_report(best_values, "KMEANS")



#----------------------------DBSCAN-------------------------------------------#

def k_dist(X,n_neighbors):
    """ 
    Plots the k-dist graph (our case 5), to check the best eps by the 
    elbow method accordingly to the paper. 
    Parameters:
    X : Array with the selected final features.
    n_neighbors : Minimum number of neighboors to become a core point
    """
    neighbours = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, np.zeros(563))
    dist = neighbours.kneighbors(X)
    distances_sorted = sorted(dist[0][:,n_neighbors-1], reverse=True)
    plt.figure(figsize=(8,6))
    plt.scatter(
        range(1,X.shape[0]+1),
        distances_sorted, 
        s=0.1)
    plt.ylim((0, 1.5)) 
    plt.xlabel("Points")
    plt.ylabel("5-dist")
    plt.savefig(f'elbow_eps_{n_neighbors}_dist.png')
    plt.show() 
    plt.close()

# ACCORDINGLY TO THE PAPER THE BEST VALUE FOR EPS IS LOCATED IN THE EDGE OF
# ELBOW THUS:
k_dist(final_features,5)
#BY OBSERVING THE K-DIST WE MAY SAY THAT THE OPTIMUM EPS VALUE IS AROUND 0.8
eps=0.8
#GET IMAGES IDS
ids = genfromtxt('labels.txt', delimiter=',')[:,0]
#GETTING DBSCAN WITH OPTIMAL EPS ACCORDINGLY TO PAPER
model = DBSCAN(eps=eps,min_samples=5)    
prediction = model.fit_predict(final_features) 
#REPORTING TO SEE THE CLUSTERS
aux.report_clusters(ids, prediction, f'DBSCAN_{eps}.html')

#EXAMINING THE PERFORMANCE OF DBSCAN FOR DIFFERENT VALUES OF EPS
esp = np.arange(0.3,0.9,0.1)
#TO DETERMINE THE BEST VALUE FOR EPS, WE COMPUTE THE INDICES FOR MULTIPLES EPS
[range_values,silhouette_scores,rand_ixs,f1_scores,precisions,recalls,adjusted_rand_ixs]=method_performance(final_features,labels, esp, "DBSCAN" )

#PLOT THE MEASURES FOR DIFFERENT EPS
plot_measures(range_values,"DBSCAN","eps")

#REPORT RESULTS WITH BEST OBSERVED VALUES FOR EPS
best_values = [0.5,0.6,0.7]
clusters_report(best_values, "DBSCAN")




#------------------------------------Q8---------------------------------------#



#GAUSSIAN MIXTURE MODELS
n_components =  np.arange(3,12,1)
[range_values,silhouette_scores,rand_ixs,f1_scores,precisions,recalls,adjusted_rand_ixs] = method_performance(final_features,labels, n_components,"GaussianMixture")
plot_measures(range_values,"GaussianMixture","n_components")
#REPORT THE CLUSTERS 
best_values = [4,6,9]
clusters_report(best_values, "GaussianMixture")

#AGGLOMERATIVE CLUSTERING 
k_clusters = np.arange(3,12,1)
#CALCULATE THE METRICS 
[range_values,silhouette_scores,rand_ixs,f1_scores,precisions,recalls,adjusted_rand_ixs]=method_performance(final_features,labels, k_clusters,"AgglomerativeClustering" )
#PLOT THE MEASURES FOR THE DIFFERENT VALUES OF K
plot_measures(range_values,"AgglomerativeClustering","Clusters")

#REPORT THE CLUSTERS 
best_values = [4,7]
clusters_report(best_values, "AgglomerativeClustering")





#----------------------------OPTIONAL EXERCISE--------------------------------#

def Bisecting_K_Means(features,n_clusters):
    """
    Function that implements the bissecting Kmeans algorithm and returns the
    list of list in the format specified in the project descryption
    Parameters:
    features- the features used to divide the points (in our case the final features after selection)
    cluster_numbers - array with the desired numbers of clusters
    """
    output_list = [ [] for i in range(features.shape[0]) ]
    number_list=[]
    index_list=[]
    indexes=np.array([ i for i in range(features.shape[0]) ])
    aux_features=features
    model=KMeans(n_clusters=2)
    for i in range(n_clusters-1):
        
      pr=model.fit_predict(aux_features)
      class0=len(pr[pr==0])
      class1=len(pr[pr==1])
      number_list.append(class0)
      number_list.append(class1)
        
      if class0>=class1:
         biggest_cluster=0
      else: 
         biggest_cluster=1
        
      index_list.append(indexes[pr==biggest_cluster])
      index_list.append(indexes[pr!=biggest_cluster])
        
      for i in range(len(pr)):
         output_list[indexes[i]].append(pr[i])
        
      maximum=max(number_list)

      if maximum<2:
         break
      indexes=index_list[number_list.index(max(number_list))]
      del index_list[number_list.index(max(number_list))]
      del number_list[number_list.index(max(number_list))]
        
      aux_features=features[indexes,:]
    
    
    return output_list

def hierirchal_clusters_report(cluster_numbers):
    """
    Function to report the html using the auxiliar report_clusters_hierarchical
    to visually see the clusters provided by the method in study.
    Parameters:
    cluster_numbers - array with the desired numbers of clusters
    """
    ids = genfromtxt('labels.txt', delimiter=',')[:,0]

    for i in best_values:
        
        clusters=Bisecting_K_Means(final_features,i)
        aux.report_clusters_hierarchical(ids, clusters, f'hierarchicalKmeans_{i}.html')
        

#REPORT THE CLUSTERS 
np.random.seed(42)
best_values = [7,9,11]
hierirchal_clusters_report(best_values)     
        
        