#!/usr/bin/env python
# coding: utf-8

# In[11]:


import os
import nltk
from nltk.tokenize import RegexpTokenizer


# In[12]:


# 1. Tokenization
tokenizer = RegexpTokenizer(r'\w+')

def tokenize_code(code):
    return set(tokenizer.tokenize(code))


# In[13]:


# 2. Compute Jaccard Similarity
def compute_jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union

def analyze_folder_for_token_similarity(folder_path):
    codes = {}
    similarity_matrix = {}
    
    # Tokenize each Python file
    for filename in os.listdir(folder_path):
        if filename.endswith('.py'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                code_content = file.read()
                codes[filename] = tokenize_code(code_content)

    # Compute and store similarity scores
    for file1, tokens1 in codes.items():
        similarity_matrix[file1] = {}
        for file2, tokens2 in codes.items():
            similarity_matrix[file1][file2] = compute_jaccard_similarity(tokens1, tokens2)

    return similarity_matrix

folder_path = "/Users/harishak/Documents/CMPE295/Dataset/CMPE255-03-fa21/pr1"
similarity_scores = analyze_folder_for_token_similarity(folder_path)

# Display similarity scores
for file1, scores in similarity_scores.items():
    print(f"Similarity scores for {file1}:")
    for file2, score in scores.items():
        print(f" - {file2}: {score:.2f}")


# In[14]:


# Assuming 'file 1' is the first key in the similarity_scores dictionary
file_1 = list(similarity_scores.keys())[0]

# Extract similarity scores for 'file 1'
file_1_scores = similarity_scores[file_1]

# Sort the files based on similarity score with 'file 1', in descending order
top_5_similar_files = sorted(file_1_scores, key=file_1_scores.get, reverse=True)[:5]

# Print the top 5 similar files
print(f"Top 5 similar files to {file_1}: {top_5_similar_files}")


# In[15]:


import csv

def read_rmse_scores_from_csv(csv_path):
    rmse_scores = {}
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rmse_scores[row['Filename']] = float(row['rmse_score'])
    return rmse_scores

rmse_csv_path = '/Users/harishak/Documents/CMPE295/Dataset/rmse_data.csv'
rmse_scores = read_rmse_scores_from_csv(rmse_csv_path)


# In[16]:


def rank_files_based_on_rmse(rmse_scores):
    # Sort files based on RMSE scores in ascending order (lower RMSE is better)
    return sorted(rmse_scores.keys(), key=lambda x: rmse_scores[x])

ranked_files = rank_files_based_on_rmse(rmse_scores)


# In[17]:


# Assuming 'file 1' is either explicitly known or the first key in the dictionary
file_1 = list(similarity_scores.keys())[0]
file_1_scores = similarity_scores[file_1]

# Sort and get top 5 files based on similarity score, excluding 'file 1' itself
top_5_similar_files = sorted(file_1_scores, key=file_1_scores.get, reverse=True)
top_5_similar_files.remove(file_1)  # Remove 'file 1' if you don't want it in the list
top_5_similar_files = top_5_similar_files[:5]

# Saving the top 5 similar files to an array
file_paths = top_5_similar_files

# Rank the top 5 similar files based on RMSE scores
ranked_files_based_on_rmse = sorted(top_5_similar_files, key=lambda x: rmse_scores[x])

# Print the ranked files
print(f"Top 5 similar files to {file_1}, ranked by RMSE: {ranked_files_based_on_rmse}")


# In[18]:


import ast

class CodeBlockVisitor(ast.NodeVisitor):
    def __init__(self):
        self.blocks = []

    def visit_FunctionDef(self, node):
        self.blocks.append(node)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.blocks.append(node)
        self.generic_visit(node)

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def compare_ast(node1, node2):
    if type(node1) != type(node2):
        return 0

    similarity_score = 0

    if hasattr(node1, 'name') and hasattr(node2, 'name'):
        similarity_score += (node1.name == node2.name)

    children1 = list(ast.iter_child_nodes(node1))
    children2 = list(ast.iter_child_nodes(node2))
    similarity_score += (len(children1) == len(children2))

    return similarity_score

def find_blocks(file_path):
    content = read_file(file_path)
    tree = ast.parse(content)
    visitor = CodeBlockVisitor()
    visitor.visit(tree)
    return visitor.blocks

def find_unique_blocks_across_files(file_paths):
    all_file_blocks = [(file_path, find_blocks(file_path)) for file_path in file_paths]

    unique_blocks_across_files = []

    for i, (file_path1, blocks1) in enumerate(all_file_blocks):
        for block1 in blocks1:
            is_unique = True
            for j, (file_path2, blocks2) in enumerate(all_file_blocks):
                if i != j:
                    if any(compare_ast(block1, block2) >= threshold for block2 in blocks2):
                        is_unique = False
                        break
            
            if is_unique:
                unique_blocks_across_files.append((file_path1, block1))

    # Print unique blocks
    print("Unique blocks across all files:")
    for file_path, block in unique_blocks_across_files:
        print(f"\nUnique block in {file_path}:\n{ast.unparse(block)}")

# List of file paths to compare
#file_paths = ['/Users/harishak/Documents/CMPE295/Dataset/CMPE255-03-fa21/pr1/0a53e3d2c329490fabbd1fa085215b05_bisecting_kmeans.py', '/Users/harishak/Documents/CMPE295/Dataset/CMPE255-03-fa21/pr1/0a755d8fce094c2ba5f57a95f9d022b5_v5.py', '/Users/harishak/Documents/CMPE295/Dataset/CMPE255-03-fa21/pr1/0ae152bcb8cf4e2ea5b583faa6be4e9d_firstSubmission.py']
threshold = 1

find_unique_blocks_across_files(file_paths)


# In[20]:


import ast

class CodeBlockVisitor(ast.NodeVisitor):
    def __init__(self):
        self.blocks = []

    def visit_FunctionDef(self, node):
        self.blocks.append(node)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.blocks.append(node)
        self.generic_visit(node)

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def compare_ast(node1, node2):
    if type(node1) != type(node2):
        return 0

    similarity_score = 0

    if hasattr(node1, 'name') and hasattr(node2, 'name'):
        similarity_score += (node1.name == node2.name)

    children1 = list(ast.iter_child_nodes(node1))
    children2 = list(ast.iter_child_nodes(node2))
    similarity_score += (len(children1) == len(children2))

    return similarity_score

def find_blocks(file_path):
    content = read_file(file_path)
    tree = ast.parse(content)
    visitor = CodeBlockVisitor()
    visitor.visit(tree)
    return visitor.blocks

def find_similar_blocks_in_first_file(file_paths):
    all_file_blocks = [(file_path, find_blocks(file_path)) for file_path in file_paths]

    similar_blocks_in_first_file = []

    first_file_blocks = all_file_blocks[0][1]
    other_files_blocks = [blocks for _, blocks in all_file_blocks[1:]]

    for block1 in first_file_blocks:
        is_similar = False
        for blocks in other_files_blocks:
            if any(compare_ast(block1, block2) >= threshold for block2 in blocks):
                is_similar = True
                break
        
        if is_similar:
            similar_blocks_in_first_file.append(block1)

    # Print similar blocks from the first file
    print("Similar blocks:")
    for block in similar_blocks_in_first_file:
        print(ast.unparse(block))

# List of file paths to compare
#file_paths = ['/Users/harishak/Documents/CMPE295/Dataset/CMPE255-03-fa21/pr1/0a53e3d2c329490fabbd1fa085215b05_bisecting_kmeans.py', '/Users/harishak/Documents/CMPE295/Dataset/CMPE255-03-fa21/pr1/0a755d8fce094c2ba5f57a95f9d022b5_v5.py', '/Users/harishak/Documents/CMPE295/Dataset/CMPE255-03-fa21/pr1/0ae152bcb8cf4e2ea5b583faa6be4e9d_firstSubmission.py']
threshold = 1

find_similar_blocks_in_first_file(file_paths)


# In[ ]:




