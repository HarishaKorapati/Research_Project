import openai
from flask import Flask, flash, redirect, render_template, request, url_for
from radon.visitors import ComplexityVisitor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import mysql.connector
import difflib
import sys
import ast

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'random string'
app.config['MESSAGE_FLASHING_OPTIONS'] = {'duration': 5}
openai.api_key = "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"

# Create the uploads folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def submit():
    name = request.form.get("name")
    uploaded_file = request.files["file"]

    if uploaded_file:
        target_filename = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(target_filename)

        flash(f"Hello, {name}! Your file '{uploaded_file.filename}' has been successfully uploaded and processed.")
        # shutil.rmtree(app.config['UPLOAD_FOLDER'])

        # Step 1: Read the code file names and measures from the database
        rows = get_database_results(
            "select code, measure from submissions where assignment_id in (select id from assignments where name = 'pr1') limit 10000")
        # Save code file names
        db_code_file_names = []
        for row in rows:
            db_code_file_names.append(row[0])

        # Step 2: Read files content from the directory where file name matches with database file name
        code_contents = []
        directory_path = "pr1"
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

        # Step 7: Get all the file names from the cluster where target file belongs
        target_files = get_file_names_same_cluster(db_code_file_names, cluster_labels, uploaded_file.filename)

        # Step 8: Find the file name with max score in the same cluster where target file belongs
        max_score_code_file = get_file_name_max_score(rows, target_files)

        # Step 9: provide the feedback
        files = get_code_file_names_to_compare(db_code_file_names, cluster_labels, uploaded_file.filename, scaled_similarities, rows)
        if max_score_code_file not in files:
            files.append(max_score_code_file)
        suggestions = compare_files_with_target(uploaded_file.filename, files)

        return render_template("result.html", comment=suggestions)

    flash("No file selected.")
    return redirect(url_for("index"))


def get_database_results(query):
    try:
        db_config = {
            "host": "localhost",
            "user": "root",
            "password": "password",
            "database": "clpdata"
        }
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
    for db_code_file_name in db_code_file_names:
        if db_code_file_name in file_paths:
            with open(directory_path + "/" + db_code_file_name, 'r', encoding='utf-8') as file:
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


def get_file_names_same_cluster(db_code_file_names, cluster_labels, target_filename):
    target_cluster = -1
    count = 1
    for file_path, cluster_label in zip(db_code_file_names, cluster_labels):
        count = count + 1
        if target_filename.__eq__(file_path):
            target_cluster = cluster_label

    count = 1
    target_files = []
    for file_path, cluster_label in zip(db_code_file_names, cluster_labels):
        if target_cluster.__eq__(cluster_label):
            target_files.append(file_path)
            # print(f"{count}. {file_path} -> Cluster {cluster_label}")
            count = count + 1

    return target_files


def get_file_name_max_score(rows, target_files):
    max_score = 0.0
    file = ''
    for row in rows:
        if row[0] in target_files:
            if max_score < row[1]:
                max_score = row[1]
                file = row[0]
    #print(f"file {file} -> max_score {max_score}")
    return file


def find_diff_code_files(target_file_name, file):
    with open('pr1/' + target_file_name, 'r') as target_file_name:
        with open('pr1/' + file, 'r') as file:
            diff = difflib.unified_diff(
                target_file_name.readlines(),
                file.readlines(),
                fromfile='target_file_name',
                tofile='file',
            )
            for line in diff:
                sys.stdout.write(line)


def read_file(filename):
    with open("pr1/" + filename, 'r') as file:
        return file.read()


def extract_modules(source_code):
    modules = []
    try:
        tree = ast.parse(source_code)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                modules.append((node.name, ast.get_source_segment(source_code, node)))

        return modules
    except SyntaxError as syntaxError:
        syntaxError
    finally:
        return modules


def calculate_module_weightage(source_code):
    modules = extract_modules(source_code)
    module_weights = []
    if modules is not None:
        module_weights = {module[1]: ComplexityVisitor.from_code(module[1]).total_complexity for module in modules}
    return module_weights


def compare_files_with_target(target_file, files_to_compare):
    target_code = open("uploads/" + target_file, 'r', encoding="utf-8").read()
    target_weights = calculate_module_weightage(target_code)
    if target_weights.__len__() == 0:
        return []

    suggestions = []

    for file in files_to_compare:
        code = read_file(file)
        weights = calculate_module_weightage(code)

        if weights.__len__() != 0:
            # Calculate similarity using difflib
            similarity = difflib.SequenceMatcher(None, list(target_weights.values()), list(weights.values())).ratio()

            # Suggest modules to remove (with less weightage) and add (with more weightage)
            modules_to_remove = [module for module, weight in target_weights.items() if weight < min(target_weights.values())]
            modules_to_add = [module for module, weight in weights.items() if weight > max(target_weights.values())]

            if modules_to_add.__len__() != 0:
                suggestions.append((file, modules_to_add))

    return suggestions


def display_suggestions(suggestions):
    if suggestions.__len__() == 0:
        print("No suggestions for the code submission, this maybe due to the submission code parsing issues")
    for file, data in suggestions.items():
        print(f"\nSuggestions for comparing with {file}:")
        print(f"Similarity: {data['similarity']}")
        if data['modules_to_remove'].__len__() != 0:
            print("Modules to Remove:", data['modules_to_remove'])
        if data['modules_to_add'] != 0:
            print("Modules to Add:", data['modules_to_add'])


def get_code_file_names_to_compare(db_code_file_names, cluster_labels, target_filename, scaled_similarities, rows):
    target_cluster = -1
    count = 1
    for file_path, cluster_label in zip(db_code_file_names, cluster_labels):
        count = count + 1
        if target_filename.__eq__(file_path):
            target_cluster = cluster_label

    target_files = []
    for file_path, scaled_similarity in zip(db_code_file_names, scaled_similarities):
        if file_path.__eq__(target_filename):
            for file_path1, scaled_similarity1, cluster_label in zip(db_code_file_names, scaled_similarity, cluster_labels):
                if target_cluster.__eq__(cluster_label):
                    if scaled_similarity1 > 0.5:
                        target_files.append(file_path1)

    score = 0.0
    final_target_files = []
    count = 0
    for row in rows:
        if row[0].__eq__(target_filename):
            score = row[1]
    for row in rows:
        if row[0] in target_files:
            if score < row[1]:
                for final_file in final_target_files:
                    if final_file.__contains__(row[0].split("_")[1]):
                        count = count+1
                if count == 0:
                    final_target_files.append(row[0])

    return final_target_files


if __name__ == "__main__":
    app.run(port=8000)
