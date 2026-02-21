import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import librosa
import math
from matplotlib import font_manager

def find_adjusted_max_indices_and_probabilities(attention_map):
    num_rows, num_cols = attention_map.shape
    indices = np.zeros(num_rows, dtype=int)
    probabilities = np.zeros(num_rows, dtype=float)

    # max index
    max_indices = np.argmax(attention_map, axis=1)
    indices[0] = max_indices[0]
    probabilities[0] = attention_map[0, max_indices[0]]

    # every max index
    for i in range(1, num_rows):
        prev_index = indices[i-1]
        current_max_index = max_indices[i]

        if current_max_index >= prev_index or current_max_index >= prev_index - 4:
            indices[i] = current_max_index
        else:
            start = max(0, prev_index - 3)
            end = min(num_cols, prev_index + 6)
            current_max_index = np.argmax(attention_map[i, start:end]) + start
            indices[i] = current_max_index

        probabilities[i] = attention_map[i, current_max_index]

    return indices, probabilities

def find_adjusted_max_indices_and_probabilities_batch_np(attention_map):
    batch_size, num_rows, num_cols = attention_map.shape
    all_indices = np.zeros((batch_size, num_rows), dtype=int)
    all_probabilities = np.zeros((batch_size, num_rows), dtype=float)

    for b in range(batch_size):
        indices = np.zeros(num_rows, dtype=int)
        probabilities = np.zeros(num_rows, dtype=float)

        max_indices = np.argmax(attention_map[b], axis=1)
        indices[0] = max_indices[0]
        probabilities[0] = attention_map[b, 0, max_indices[0]]

        for i in range(1, num_rows):
            prev_index = indices[i-1]
            current_max_index = max_indices[i]

            if current_max_index >= prev_index or current_max_index >= prev_index - 4:
                indices[i] = current_max_index
            else:

                start = max(0, prev_index - 3)
                end = min(num_cols, prev_index + 6)
                current_max_index = np.argmax(attention_map[b, i, start:end]) + start
                indices[i] = current_max_index

            probabilities[i] = attention_map[b, i, current_max_index]

        all_indices[b] = indices
        all_probabilities[b] = probabilities

    return all_indices, all_probabilities

def find_adjusted_max_indices_and_probabilities_batch(attention_map):
    batch_size, num_rows, num_cols = attention_map.shape
    all_indices = torch.zeros((batch_size, num_rows), dtype=torch.int64).cuda()
    all_probabilities = torch.zeros((batch_size, num_rows), dtype=torch.float32).cuda()

    max_indices = torch.argmax(attention_map, dim=2)
    all_indices[:, 0] = max_indices[:, 0]
    all_probabilities[:, 0] = attention_map[torch.arange(batch_size), 0, max_indices[:, 0]]

    for i in range(1, num_rows):
        prev_indices = all_indices[:, i-1]
        current_max_indices = max_indices[:, i]

        valid_mask = (current_max_indices >= prev_indices) | (current_max_indices >= prev_indices - 4)

        all_indices[:, i] = torch.where(valid_mask, current_max_indices, all_indices[:, i])

        for b in range(batch_size):
            if not valid_mask[b]:
                start = max(0, prev_indices[b] - 3)
                end = min(num_cols, prev_indices[b] + 6)
                all_indices[b, i] = torch.argmax(attention_map[b, i, start:end]) + start

        all_probabilities[:, i] = attention_map[torch.arange(batch_size), i, all_indices[:, i]]

    return all_indices, all_probabilities

def calculate_cross(tensor1, tensor2, mask1 = None, mask2 = None):

    dot_product = torch.bmm(tensor1, tensor2.transpose(1, 2))

    tensor1_norm = tensor1.norm(dim=2, keepdim=True)
    tensor2_norm = tensor2.norm(dim=2, keepdim=True)

    norm_product = torch.bmm(tensor1_norm, tensor2_norm.transpose(1, 2))

    cosine_sim_matrix = dot_product / norm_product
    if mask1 is not None:
        mask_matrix = torch.bmm(mask1, mask2.transpose(1, 2))
        cosine_sim_matrix = cosine_sim_matrix * mask_matrix
        # cosine_sim_matrix = cosine_sim_matrix.masked_fill(mask == 0, float('-inf'))
    
    return cosine_sim_matrix

def calculate_probability(data, mask1):
    masked_data = data * mask1.squeeze(-1)

    valid_counts = mask1.sum(dim=1, keepdim=True)
    mean = masked_data.sum(dim=1) / valid_counts.squeeze()
    
    mask1 = mask1.squeeze(-1)
    variance = ((masked_data - mean.unsqueeze(1)) ** 2 * mask1).sum(dim=1) / valid_counts.squeeze()
    std = torch.sqrt(variance)
    
    score = mean - std
    return mean, std, score

def plot_sns(cosine_sim_matrix):
    font_path = '/train20/intern/permanent/kwli2/wwsnet/wwsnet-ts/tools/Times_New_Roman.ttf'
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 25
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cosine_sim_matrix, annot=False, cmap='viridis', cbar=True)
    plt.title('Cosine Similarity Attention Map')
    plt.xlabel('Tensor2')
    plt.ylabel('Tensor1')
    plt.savefig('g.png')
    plt.show()

# cosine_sim_matrix = calculate_cross(tensor1, tensor2, mask1, mask2)
# # cosine_sim_matrix = cosine_sim_matrix.cpu().detach().numpy()
# print(cosine_sim_matrix)

# indices, probabilities = find_adjusted_max_indices_and_probabilities_batch(cosine_sim_matrix)
# print("Indices:", indices)
# print("Probabilities:", probabilities)


# mean, std, score = calculate_probability(probabilities, mask1)

# print("mean:", mean)
# print("std:", std)

# print("score:", score)
