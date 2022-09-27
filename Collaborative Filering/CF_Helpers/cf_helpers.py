import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import itertools
from sklearn.metrics import silhouette_samples, silhouette_score


# function to obtain silhouette score fitted number of clusters
def silhouette_score_by_cluster(k, data):
    kmeans = KMeans(n_clusters = k).fit(data) 
    predictions = kmeans.predict(data)
    silhouette_avg = silhouette_score(data, predictions)
    return silhouette_avg

# function to add value labels
def addlabels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i]//2, y[i], ha = 'center')
        
#function to filter df given cluster ID
def filter_df_cluster(df , x): #x represents cluster number
    return df[df.cluster == x]

#function to filter 
def round_to_nearest_grade(x):
    return round(x * 4) / 4

# input student data (list of strings) of user j (index number as integer)
# this method only applies for students of political science
# educational science does not have such clear indication 
def filter_T1_courses(student_data): 
    courses_not_taken = list(student_data.columns[student_data.isna().all()])
    prefix = 'T1'
    for course in courses_not_taken[:]:
        if course.startswith(prefix):
            courses_not_taken.remove(course)
    selection = pd.DataFrame()
    selection[''] = courses_not_taken
    selection = selection.set_index([''])
    return selection
# helper function to create metric table per cluster

def create_metric_df(clusters, list_of_df, df_cluster):
    df_fin_list = []
    for i in clusters:
        for dfi in range(len(list_of_df)):
            df = list_of_df[dfi]
            a = df[df['ClusterID'] == i].mean().tolist()
            column = df_cluster.columns[dfi]
        
            df_cluster.loc['Recall'][dfi] = a[1]
            df_cluster.loc['Precision'][dfi] = a[2]
            df_cluster.loc['F1'][dfi] = a[3]
    
        iterables = [[i] ,df_cluster.index.tolist()]
        index = pd.MultiIndex.from_product(iterables, names=["ClusterID", 'Metrics'])
        temp = df_cluster.set_index(index)
        df_fin_list.append(temp)
    
    df_fin = pd.concat(df_fin_list, axis = 0)
    return df_fin