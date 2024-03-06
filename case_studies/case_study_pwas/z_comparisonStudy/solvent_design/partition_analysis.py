import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from scipy.cluster import hierarchy


# Function to calculate Jaccard similarity between two numerical sets
def jaccard_similarity(set1, set2):
    set1 = set(set1.flatten())
    set2 = set(set2.flatten())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def order_map(similarity_matrix):
    # Perform hierarchical clustering using linkage
    linkage_matrix = hierarchy.linkage(similarity_matrix, method='single')

    order = hierarchy.leaves_list(linkage_matrix)
    clustered_matrix = similarity_matrix[:, order][order, :]

    # Create a mapping between original and reordered indices
    original_indices = np.arange(len(similarity_matrix))
    index_mapping = dict(zip(original_indices,order))
    return clustered_matrix, index_mapping,order


matrix_excel_file = 'partition_analysis.xlsx'
sheet_name_solvent = 'solvent'
sheet_name_partition = 'partition'
sheet_name_yield = 'yield'
solvents = np.array(pd.read_excel(matrix_excel_file, sheet_name = sheet_name_solvent).iloc[0:, 1:])
partitions = np.array(pd.read_excel(matrix_excel_file, sheet_name = sheet_name_partition).iloc[0:, 1])
yield_solvent = np.array(pd.read_excel(matrix_excel_file, sheet_name = sheet_name_yield).iloc[0:, 1])
num_vectors = solvents.shape[0]

# Calculate similarity matrix using Jaccard similarity
similarity_matrix_partition = np.zeros((num_vectors, num_vectors))
for i in range(num_vectors):
    for j in range(num_vectors):
        similarity_matrix_partition[i, j] = jaccard_similarity(partitions[i], partitions[j])


similarity_matrix_solvent = cosine_similarity(solvents)
# similarity_matrix_partition = cosine_similarity(partitions.reshape(-1,1))
# similarity_matrix_yield = euclidean_distances(yield_solvent)
similarity_matrix_yield = np.corrcoef(yield_solvent, yield_solvent)[0, 1]

clustered_matrix_solvent, index_mapping_solvent, order_solvent = order_map(similarity_matrix_solvent)
# clustered_matrix_partition, index_mapping_partition = order_map(similarity_matrix_partition)
clustered_matrix_partition = similarity_matrix_partition[:, order_solvent][order_solvent, :]
clustered_matrix_yield = similarity_matrix_yield[:, order_solvent][order_solvent, :]

# Create a mapping between original and reordered indices for partition
original_indices_partition = np.arange(len(similarity_matrix_partition))
index_mapping_partition = dict(zip(original_indices_partition, order_solvent))
index_mapping_yield = dict(zip(original_indices_partition, order_solvent))

# Create a heatmap using seaborn
sns.set(style="white")  # Set the style of the heatmap
plt.figure(figsize=(8, 6))  # Set the size of the heatmap

# Create a heatmap using seaborn's heatmap function
sns.heatmap(clustered_matrix_solvent, annot=False, cmap="YlGnBu",
            xticklabels=([f"{index_mapping_solvent[i]+1}" for i in range(len(similarity_matrix_solvent))]),
            yticklabels=([f"{index_mapping_solvent[i]+1}" for i in range(len(similarity_matrix_solvent))]),
)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.title('Similarity Heatmap of Solvent Structure (Group contribution)')
plt.xlabel('Solvent iteration',fontsize=12)
plt.ylabel('Solvent iteration',fontsize=12)
plt.savefig('similarity_matrix_solvent.png', dpi=400)

sns.set(style="white")  # Set the style of the heatmap
plt.figure(figsize=(8, 6))  # Set the size of the heatmap

# Create a heatmap using seaborn's heatmap function
sns.heatmap(clustered_matrix_partition, annot=False, cmap="YlGnBu",
            xticklabels=([f"{index_mapping_partition[i]+1}" for i in range(len(similarity_matrix_partition))]),
            yticklabels=([f"{index_mapping_partition[i]+1}" for i in range(len(similarity_matrix_partition))]),
)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.title('Similarity Heatmap of Partition')
plt.xlabel('Solvent iteration',fontsize=12)
plt.ylabel('Solvent iteration',fontsize=12)
plt.savefig('similarity_matrix_partition.png', dpi=400)


sns.set(style="white")  # Set the style of the heatmap
plt.figure(figsize=(8, 6))  # Set the size of the heatmap

# Create a heatmap using seaborn's heatmap function
sns.heatmap(clustered_matrix_yield, annot=False, cmap="YlGnBu",
            xticklabels=([f"{index_mapping_yield[i]+1}" for i in range(len(similarity_matrix_yield))]),
            yticklabels=([f"{index_mapping_yield[i]+1}" for i in range(len(similarity_matrix_yield))]),
)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.title('Similarity Heatmap of Yield')
plt.xlabel('Solvent iteration',fontsize=12)
plt.ylabel('Solvent iteration',fontsize=12)
plt.savefig('similarity_matrix_yield.png', dpi=400)


