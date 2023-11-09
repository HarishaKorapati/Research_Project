from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import mysql.connector
from sklearn.decomposition import PCA
import difflib
import sys


def get_database_results(query):
    try:
        conn = mysql.connector.connect(**db_config)
        if conn.is_connected():
            print("Connected to the MySQL db")
        cursor_obj = conn.cursor()
        cursor_obj.execute(query)
        rows = cursor_obj.fetchall()
        return rows
    except mysql.connector.Error as error:
        print(f"Error: {error}")
    finally:
        if "cursor_obj" in locals() and cursor_obj is not None:
            cursor_obj.close()
        if conn.is_connected():
            conn.close()
            print("Db connection closed")


def read_code_files(db_code_file_names, directory_path, code_contents):
    file_paths = os.listdir(directory_path)
    for file_path in file_paths:
        if file_path in db_code_file_names:
            with open(directory_path + "/" + file_path, 'r', encoding='utf-8') as file:
                code_contents.append(file.read())


def tokenization(code_contents):
    return [' '.join([token.lower() for token in code.split()]) for code in code_contents]


def tfidf_vectorization(tokenized_code):
    tfidf_vectorizer = TfidfVectorizer()
    return tfidf_vectorizer.fit_transform(tokenized_code)


def calculate_similarity(vectorized_matrix):
    similarities = cosine_similarity(vectorized_matrix, vectorized_matrix)
    scaler = StandardScaler()
    return scaler.fit_transform(similarities)


def perform_clustering(scaled_similarities, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    return kmeans.fit_predict(scaled_similarities)


def get_file_names_same_cluster(directory_path, cluster_labels, target_filename):
    target_cluster = -1
    count = 1
    file_paths = os.listdir(directory_path)
    for file_path, cluster_label in zip(file_paths, cluster_labels):
        count = count + 1
        if target_filename.__eq__(file_path):
            target_cluster = cluster_label

    count = 1
    target_files = []
    for file_path, cluster_label in zip(file_paths, cluster_labels):
        if target_cluster.__eq__(cluster_label):
            target_files.append(file_path)
            print(f"{count}. {file_path} -> Cluster {cluster_label}")
            count = count + 1

    return target_files


def cluster_visualization(scaled_similarities, cluster_labels):
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(scaled_similarities)
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels)
    plt.title('Cluster Visualization')
    plt.show()


def get_file_name_max_score(rows, target_files):
    file = ''
    max_score = 0.0
    for row in rows:
        if row[0] in target_files:
            if max_score < row[1]:
                max_score = row[1]
                file = row[0]
    print(f"file {file} -> max_score {max_score}")
    return file


def find_diff_code_files(target_file_name, file):
    with open('C:/Users/gangu/OneDrive/Desktop/295A/pr1/' + target_file_name, 'r') as target_file_name:
        with open('C:/Users/gangu/OneDrive/Desktop/295A/pr1/' + file, 'r') as file:
            diff = difflib.unified_diff(
                target_file_name.readlines(),
                file.readlines(),
                fromfile='target_file_name',
                tofile='file',
            )
            for line in diff:
                sys.stdout.write(line)


# Execution starts here

# Step 1: Read the code file names and measures from the database
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "password",
    "database": "clpdata"
}
rows = get_database_results(
    "select code, measure from submissions where assignment_id in (select id from assignments where name = 'pr1') limit 10000")
# Save code file names
db_code_file_names = []
for row in rows:
    db_code_file_names.append(row[0])

# Step 2: Read files content from the directory where file name matches with database file name
code_contents = []
directory_path = "C:/Users/gangu/OneDrive/Desktop/295A/pr1"
read_code_files(db_code_file_names, directory_path, code_contents)

# Step 3: Tokenization and Preprocessing
tokenized_code = tokenization(code_contents)

# Step 4: Vectorization (TF-IDF)
vectorized_matrix = tfidf_vectorization(tokenized_code)

# Step 5: Similarity Calculation (Cosine Similarity)
scaled_similarities = calculate_similarity(vectorized_matrix)

# Step 6: Clustering (K-Means)
num_of_clusters = 6
cluster_labels = perform_clustering(scaled_similarities, num_of_clusters)

# Step 7: Visualize the clusters
cluster_visualization(scaled_similarities, cluster_labels)

# Step 8: Get all the file names from the cluster where target file belongs
target_filename = "55434aca8ac2457ab6b03d17a08b367b_bkm_v3.py"
target_files = get_file_names_same_cluster(directory_path, cluster_labels, target_filename)

# Step 9: Find the file name with max score in the same cluster where target file belongs
file = get_file_name_max_score(rows, target_files)

# Step 10: provide the feedback
find_diff_code_files(target_filename, file)
