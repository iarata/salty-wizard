# I assume the way the clusters are going to be encoded is

# image/path/name.png -> cluster number

# STEPS

# - Group image names cluster wise (just in case)
# - Calculate average distance and centroids of clusters
# - Calculate distance of centroids to closest other centroids
# - Calculate ratio of the average distance between within cluster and the centroid of the cluster to the distance between the centroids of the cluster and the nearest cluster.
# - Calculate the average of the ratios for all clusters.

# HOW TO IMPLEMENT

# Use from sklearn.metrics import davies_bouldin_score

import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score

from functools import reduce
from collections import defaultdict

import math
from sklearn.metrics import adjusted_rand_score

# Add parent directory to path to import brain_text_model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from brain_text_model.tokenizer import CharTokenizer, TokenizerConfig

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

# Initialize tokenizer
tokenizer = CharTokenizer(TokenizerConfig())

def group_image_paths(labels: dict):
    """
    Function that turns a dicitonary mapping an image path to a cluster lable,
    into mapping cluster label to list of paths
    """
    groups = {}

    for key, value in labels.items():
        if value not in groups:
            groups[value] = []
        groups[value].append(key)

    return groups
    
def get_image_data(cluster_paths: list, super_path: str = ''):
    """
    Function that load the images form the paths provided in the list
    """

    data = []

    for path in cluster_paths:
        # Load image
        img = plt.imread(super_path + '_'.join(path.split('_')[:2]))
        data.append(img)

    return data

def get_sentence_data_from_image_path(cluster_paths: list):
    """
    Extracts the sentences used to generate the images from the provided list of paths
    """
    sentences = []
    
    for path in cluster_paths:
        s = path.split("\\")[1].split("_")[0]
        s = ' '.join(s.split('-'))
        sentences.append(s)

    return sentences

def get_sentence_data(cluster_paths: list, data_path: str = ''):
    """
    Extracts the sentences from HDF5 files based on the provided paths.
    Each path is split/date/trial, where trial is the row ID in the HDF5 file.
    """
    import h5py
    from collections import defaultdict
    
    # Group paths by file to minimize file I/O
    file_to_trials = defaultdict(list)
    path_order = []  # Track original order
    
    for path in cluster_paths:
        split, date, trial = path.split("/")
        file_path = os.path.join(data_path, date, f'data_{split}.hdf5')
        
        if trial not in file_to_trials[file_path]:
            file_to_trials[file_path].append((trial, len(path_order)))
            path_order.append(None)  # Placeholder
    
    # Read files once and extract required rows
    for file_path, trial_list in file_to_trials.items():
        with h5py.File(file_path, 'r') as f:

            for trial_id, original_idx in trial_list:
                # Access the row with the given trial_id                
                trial_id = '_'.join(trial_id.split('_')[:2])  # Adjust trial_id format
                
                group = f[trial_id]
                attrs = {key: group.attrs[key] for key in group.attrs}
                try:
                    sentence = attrs.get("sentence_label")
                except KeyError:
                    sentence = []
                # Decode if bytes
                if isinstance(sentence, bytes):
                    sentence = sentence.decode('utf-8')
                # Remove punctuation if any
                sentence = sentence.replace('.', '').replace(',', '').replace('!', '').replace('?', '')
                # Remove stop words
                sentence = ' '.join([word for word in sentence.split() if word.lower() not in stop_words])
                path_order[original_idx] = tokenizer.decode(tokenizer.encode(sentence)).split()
                # print(f"Extracted sentence: {path_order[original_idx]}")
    return path_order

def word_frequency(sentences: list):
    """
    Gets list of sentences and uses map reduce to count the number of occurences of each word
    """

    # Flatten the list of lists
    # print("Calculating word frequency for sentences:", sentences[0:3], "...")

    flattened_words = []

    for sentence in sentences:
        if sentence is None:
            continue
        flattened_words.extend(sentence)

    mapped_words = map(lambda w: (w, 1), flattened_words)

    def reducer(acc, pair):
        word, count = pair
        acc[word] += count
        return acc
    
    # Use reduce to aggregate word counts
    word_freq = reduce(reducer, mapped_words, defaultdict(int))

    # Convert defaultdict to a regular dictionary
    return dict(word_freq)

def tf_idf(cluster_data: dict):
    """
    Function that calculates the tf-idf index of words in clusters

    Input:
    - cluster_data: dictionary of cluster number and a list of sentences in that cluster
    """
    # Get word frequency of all clusters
    freqs = {}
    for cluster, sentences in cluster_data.items():
        freqs[cluster] = word_frequency(sentences)

    # Calculate Term Frequency in each cluster
    #   count of word / max count of word
    tf = {}
    idf = {}

    for cluster, w_freqs in freqs.items():

        max_f = w_freqs[max(w_freqs, key=w_freqs.get)]

        cluster_tfs  = {}
        cluster_idfs = {}

        for word, count in w_freqs.items():
            cluster_tfs[word] = count/max_f
            
            count = 0
            for _, words_dict in freqs.items():
                if word in words_dict:
                    count += 1

            cluster_idfs[word] = math.log(len(freqs.keys())/count, 2)
        
        tf[cluster]  = cluster_tfs
        idf[cluster] = cluster_idfs

    # Calculate TF-IDF
    tf_idf = {}
    for cluster in tf:
        cluster_tf_idf = {}
        for word in tf[cluster]:
            cluster_tf_idf[word] = tf[cluster][word] * idf[cluster][word]
        tf_idf[cluster] = cluster_tf_idf
    
    return tf_idf

def cluster_trial_frequency(cluster_data: dict):
    """
    Function that calculates the frequency of trials in each cluster

    Input:
    - cluster_data: dictionary of cluster number and a list of sentences in that cluster
    """
    trial_freq_clusters = {}

    for cluster in cluster_data:
        cluster_paths = cluster_data[cluster]
        
        trial_freq = {}

        for path in cluster_paths:
            _, date, trial = path.split("/")
            trial_id = '_'.join(trial.split('_')[:2])
            key = '/'.join([_, date, trial_id])
            if key not in trial_freq:
                trial_freq[key] = 0
            trial_freq[key] += 1

        trial_freq_clusters[cluster] = trial_freq

    return trial_freq_clusters

def davies_bouldin_index(data: list, labels: list):
    """
    Function that calculates the davies bouldin index of the provided data and labels
    """
    return davies_bouldin_score(data, labels)

def adjusted_rand_index_calc(cluster_data1: dict, cluster_data2: dict):
    """
    Function that calculates the Adjusted Rand Index (ARI) between two clusterings.

    Inputs:
    - cluster_data1: dict mapping cluster_id -> list of item identifiers (e.g. image paths)
    - cluster_data2: dict with the same structure and the same set of item identifiers

    Returns:
    - float: Adjusted Rand Index
    """

    def invert(clusters: dict):
        mapping = {}
        for label, items in clusters.items():
            for it in items:
                mapping[it] = label
        return mapping

    m1 = invert(cluster_data1)
    m2 = invert(cluster_data2)

    if set(m1.keys()) != set(m2.keys()):
        raise ValueError("Both clusterings must contain the same set of items.")

    items = sorted(m1.keys())  # deterministic ordering
    labels1 = [m1[it] for it in items]
    labels2 = [m2[it] for it in items]

    return adjusted_rand_score(labels1, labels2)

def box_plot_trial_frequency(trial_freq: dict):
    """
    Function that creates a box plot of the trial frequencies in each cluster

    Input:
    - cluster_data: dictionary of cluster number and a list of sentences in that cluster
    """

    data = []
    labels = []

    for cluster, freq_dict in trial_freq.items():
        freqs = list(freq_dict.values())
        data.append(freqs)
        labels.append(str(cluster))

    plt.boxplot(data, labels=labels)
    plt.xlabel('Cluster')
    plt.ylabel('Trial Count')
    plt.title('Number of same trial bins in clusters')
    plt.show()

def main(cluster_file: str = 'src\\clustering\\across_trial_clusters_chunked.pkl'):

    # Load per_trial_clusters_raw.pkl
    with open(cluster_file, 'rb') as f:
        import pickle
        per_trial_clusters = pickle.load(f)
    
    clusters = per_trial_clusters

    print("Number of clusters:", len(clusters))
    print(clusters.keys())
    
    path = 'src/data/hdf5_data_final'

    trial_freq = cluster_trial_frequency(clusters)

    box_plot_trial_frequency(trial_freq)
    
    new_clusters = defaultdict(list)
    # Keep trial in cluster if most frequent in reverse order to remove items
    for cluster, freq_dict in trial_freq.items():
        for trial, freq in freq_dict.items():
        
            other_max = 0
            for cluster2 in trial_freq.keys():
                if cluster2 == cluster:
                    continue
                try:
                    if trial_freq[cluster2][trial] > other_max:
                        other_max = trial_freq[cluster2][trial]
                except KeyError:
                    continue
            if freq > other_max:
                new_clusters[cluster].append(trial)


    data_img = {}
    data_txt = {}
    for cluster, paths in new_clusters.items():
        # data_img[cluster] = get_image_data(paths, super_path=path)
        data_txt[cluster] = get_sentence_data(paths, data_path=path)

    # Sanity check
    # print(data_txt)

    tf_idf_indices = tf_idf(data_txt)
    
    # SOrt words by tf-idf value and print top 5
    for cluster, indices in tf_idf_indices.items():
        sorted_words = sorted(indices.items(), key=lambda item: item[1], reverse=True)
        print(f"Cluster {cluster} top words:")
        for word, score in sorted_words[:5]:
            print(f"  {word}: {score:.4f}")
    
    # print("Davies-Bouldin Index:")
    # db_index = davies_bouldin_index(
    #     data=[img.flatten() for cluster in data_img.values() for img in cluster],
    #     labels=[int(cluster) for cluster, paths in clusters.items() for _ in paths]
    # )
    # print(db_index)
    return new_clusters

if __name__ == "__main__":
    new_cluster = main("")

    for cluster, paths in new_cluster.items():
        print(f"Cluster {cluster} has {len(paths)} items.")