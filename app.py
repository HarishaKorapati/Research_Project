import shutil

import openai
from flask import Flask, flash, redirect, render_template, request, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import os

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
    taskType = request.form.get('option')

    if uploaded_file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(filename)
        top_3_results = compare_files(filename, taskType)
        comment = generate_code_comment(open(filename, 'r', encoding="utf-8").read())

        flash(f"Hello, {name}! Your file '{uploaded_file.filename}' has been successfully uploaded and processed.")
        # shutil.rmtree(app.config['UPLOAD_FOLDER'])
        return render_template("result.html", top_3_results=top_3_results, comment=comment)

    flash("No file selected.")
    return redirect(url_for("index"))


def compare_files(file_name, task):
    result = []
    reference_folder = "old_programs"
    user_code = open(file_name, 'r', encoding="utf-8").read()
    for filename in os.listdir(os.path.join(reference_folder, task)):
        if filename.endswith(".py"):
            with open(os.path.join(reference_folder, task, filename), "r", encoding="utf-8") as file:
                reference_code = file.read()
                similarity = difflib.SequenceMatcher(None, user_code, reference_code).ratio()
                result.append((filename, round(similarity, 4)))

    # Sort the results by similarity score (in descending order)
    result.sort(key=lambda x: x[1], reverse=True)

    top_3_results = result[:3]
    return top_3_results


def similarityMeasure(file_name, task):
    referenceCodes = []
    reference_folder = "old_programs"
    for filename in os.listdir(os.path.join(reference_folder, task)):
        if filename.endswith(".py"):
            with open(os.path.join(reference_folder, task, filename), "r", encoding="utf-8") as file:
                referenceCodes.append(file.read())

    # Initialize TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Transform reference code into TF-IDF vectors
    tfidf_matrix = tfidf_vectorizer.fit_transform(referenceCodes)

    # Transform user's code into a TF-IDF vector
    user_code = open(file_name, 'r', encoding="utf-8").read()
    user_tfidf = tfidf_vectorizer.transform([user_code])

    # Calculate cosine similarities
    similarities = cosine_similarity(user_tfidf, tfidf_matrix)[0]

    # Pair each similarity score with the corresponding file name
    file_similarities = list(zip(os.listdir(os.path.join(reference_folder, task)), similarities))

    # Sort by similarity in descending order
    file_similarities.sort(key=lambda x: x[1], reverse=True)

    # Get the top-3 most relevant files with scores
    top_3_results = file_similarities[:3]
    return top_3_results


def generate_code_comment(user_code_text):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Act as if you are a professor and Please provide comments and suggest what needs to be done"
                   f" for the following code:\n\n{user_code_text}",
            max_tokens=500,
            stop=None
        )
        print(response)
        comment = response.choices[0].text.strip()
    except Exception as e:
        print(f"OpenAI GPT request error: {str(e)}")
        comment = "An error occurred while generating comments."

    return comment


if __name__ == "__main__":
    app.run(port=8000)
