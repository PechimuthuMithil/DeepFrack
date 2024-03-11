import numpy as np
import string
import json


def save_dictionary_to_file(dictionary, filename):
    with open(filename, 'w') as file:
        json.dump(dictionary, file)

def load_dictionary_from_file(filename):
    with open(filename, 'r') as file:
        dictionary = json.load(file)
    return dictionary

# Function to normalize a string (convert to lowercase and remove punctuation)
def normalize_string(s):
    return s.translate(str.maketrans('', '', string.punctuation)).lower()

# Function to calculate cosine similarity between two strings
def cosine_similarity(s1, s2):
    # Tokenize strings
    tokens = set(s1.split()) | set(s2.split())
    # Create vectors
    v1 = [s1.split().count(token) for token in tokens]
    v2 = [s2.split().count(token) for token in tokens]
    # Calculate cosine similarity
    similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return similarity

def jaccard_similarity(s1, s2):
    set1 = set(s1)
    set2 = set(s2)
    return len(set1.intersection(set2)) / len(set1.union(set2))

# Main function
def main():
    logo = '''
 _____                      _______               _     
(____ \                    (_______)             | |    
 _   \ \ ____ ____ ____     _____ ____ ____  ____| |  _ 
| |   | / _  ) _  )  _ \   |  ___) ___) _  |/ ___) | / )
| |__/ ( (/ ( (/ /| | | |  | |  | |  ( ( | ( (___| |< ( 
|_____/ \____)____) ||_/   |_|  |_|   \_||_|\____)_| \_)
                  |_|                                   

'''
    print(logo)

    # Database dictionary
    database = load_dictionary_from_file("./Datasets.json")

    # Ask the user for the required Workload
    workload_query = input("Enter the required Workload: ")
    workload_norm = normalize_string(workload_query)

    # Ask the user for the required Architecture
    architecture_query = input("Enter the required Architecture: ")
    architecture_norm = normalize_string(architecture_query)

    # Find the closest match in the database
    max_sim = 0
    workload_match = ''
    for workload in database["Workloads"]:
        norm = normalize_string(workload)
        sim = jaccard_similarity(norm,workload_norm)
        if (sim > max_sim):
            max_sim = sim
            workload_match = workload

    if (max_sim <= 0):
        ans_w = 'n'
        workload_match = workload_query 
    else:
        ans_w = input(f"We found {workload_match} in our datasets. Do you want to use this? [y/N]: \n")
    
    max_sim = 0
    architecture_match = ''
    for arch in database["Architectures"]:
        norm = normalize_string(arch)
        sim = jaccard_similarity(norm,architecture_norm)
        if (sim > max_sim):
            max_sim = sim
            architecture_match = arch
    
    if (max_sim <= 0):
        ans_a = 'n'
        architecture_match = architecture_query
    else:
        ans_a = input(f"We found {architecture_match} in our datasets. Do you want to use this? [y/N]: \n")

    if (ans_w.lower() == 'n'):
        workload_path = input("No match in datasets, Enter the path to the folder of the workload YAML files: \n")
        database["Workloads"][workload_query] = workload_path

    if (ans_a.lower() == 'n'):
        architecture_path = input("No match in datasets, Enter the path to the folder of the architecture YAML files: \n")
        database["Architectures"][architecture_query] = architecture_path

    final_workload = database["Workloads"][workload_match]
    final_architecture = database["Architectures"][architecture_match]
    
    print("Use the following while runnign the Benchmarker.py: \n")
    print("folder_path: ", final_workload, "\n")
    print("architecture description parent folder: ", final_architecture, "\n")


# Execute the main function
if __name__ == "__main__":
    main()
