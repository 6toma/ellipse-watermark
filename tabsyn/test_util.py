import os
import json
import numpy as np
import torch

from tabsyn.process_syn_dataset import process_data, preprocess_syn

def get_input_train(dataname):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = f'{curr_dir}/../data/{dataname}/'

    with open(f'{dataset_dir}/info.json', 'r') as f:
        info = json.load(f)

    ckpt_dir = f'{curr_dir}/ckpt/{dataname}/'
    embedding_save_path = f'{curr_dir}/vae/ckpt/{dataname}/train_z.npy'
    train_z = torch.tensor(np.load(embedding_save_path)).float()
    print(train_z.size())

    train_z = train_z[:, 1:, :]
    B, num_tokens, token_dim = train_z.size()
    print(train_z.size())

    in_dim = num_tokens * token_dim
    train_z = train_z.view(B, in_dim)
    print(train_z.size())

    return train_z, curr_dir, dataset_dir, ckpt_dir, info


if __name__ =='__main__':
    get_input_train('magic')
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = f'{curr_dir}/ckpt/magic'

    ori_noise = np.load(f'{ckpt_path}/ori_noisy_latent.npy')
    print(np.max(ori_noise), np.min(ori_noise))
    reversed_noise = np.load(f'{ckpt_path}/reversed_noisy_latent.npy')
    print(np.max(reversed_noise), np.min(reversed_noise))

    # Compute squared differences
    squared_diff = np.abs(ori_noise - reversed_noise)

    # Compute mean squared error
    mse = np.mean(squared_diff)
    print(mse)


