import os
import torch

import argparse
import warnings
import time
import numpy as np
import json

from tabsyn.model import MLPDiffusion, DDIMModel, DDIMScheduler
from tabsyn.latent_utils import get_input_generate, recover_data, split_num_cat_target  # get_encoder_latent
# from tabsyn.watermark_utils import get_watermarking_mask, inject_watermark, get_watermarking_pattern, eval_watermark
from tabsyn_old_cide.watermark_utils_old import get_noise, detect

from utils_train import preprocess
from tabsyn.vae.model import Encoder_model

# from tabsyn.process_syn_dataset import process_data, preprocess_syn

warnings.filterwarnings('ignore')


def main(args):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    save_path_arg = args.save_path
    # save_dir = f'{curr_dir}/../{save_path_arg}'
    # if not os.path.exists(save_dir):
    #     # If it doesn't exist, create it
    #     os.mkdir(save_dir)

    dataname = args.dataname
    device = args.device
    steps = args.steps

    watermark = args.wm

    train_z, _, _, ckpt_path, info, num_inverse, cat_inverse = get_input_generate(args)
    in_dim = train_z.shape[1]

    mean = train_z.mean(0)

    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)

    # Score-based
    # model = Model(denoise_fn = denoise_fn, hid_dim = train_z.shape[1]).to(device)

    # DDIM
    model = DDIMModel(denoise_fn).to(device)

    model.load_state_dict(torch.load(f'{ckpt_path}/model.pt', map_location=torch.device('cpu')))

    '''
        Generating samples    
    '''
    start_time = time.time()

    num_samples = train_z.shape[0]
    sample_dim = in_dim
    # torch.manual_seed(i)
    # init_latents = torch.randn([num_samples, sample_dim], device=device)
    # if watermark is not 'None':
    #     watermarked_latents, key, channel, radius = get_noise([num_samples, sample_dim], pattern=watermark)
    # else:
    #     print('No watermark')
    #     watermarked_latents, key, channel, radius = get_noise([num_samples, sample_dim], pattern='rand')

    # Score-based
    # x_next = sample(model.denoise_fn_D, latents, args=args)

    # DDIM
    # with (open(f'results_marked_{watermark}.txt', 'w') as f):
    #     for i in range(100):
    init_latents = torch.randn([num_samples, sample_dim], device=device)
    if watermark is not 'None':
        watermarked_latents, key, channel, radius = get_noise([num_samples, sample_dim], pattern=watermark)
    else:
        print('No watermark')
        watermarked_latents, key, channel, radius = get_noise([num_samples, sample_dim], pattern='rand')
    # watermarked_latents, key, channel, radius = get_noise([num_samples, sample_dim], pattern='rand')
    noise_scheduler = DDIMScheduler(num_train_timesteps=1000)

    if watermark is not 'None':
        latents = watermarked_latents
    else:
        latents = init_latents
    x_next = noise_scheduler.generate(
        model.noise_fn,
        latents=latents,
        num_inference_steps=steps,
        eta=0.0)

    x_next = x_next * 2 + mean.to(device)
    syn_data = x_next.float().cpu().numpy()
    syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, cat_inverse, args.device)

    syn_df = recover_data(syn_num, syn_cat, syn_target, info)

    idx_name_mapping = info['idx_name_mapping']
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

    syn_df.rename(columns=idx_name_mapping, inplace=True)
    # save_path = f'{save_dir}/{args.method}.csv'
    save_path = args.save_path
    syn_df.to_csv(save_path, index=False)

    end_time = time.time()
    print('Time:', end_time - start_time)
    print('Saving sampled data to {}'.format(save_path))

    # Loading diffusion model for inverse process
    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    model = DDIMModel(denoise_fn).to(device)
    model.load_state_dict(torch.load(f'{ckpt_path}/model.pt', map_location=torch.device('cpu')))
    noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
    # Inverse process
    # TODO
    # DDIM reverse to get the initial noise latent used for synthesizing

    dataset_dir_syn = f'data/{dataname}_ring'
    dataset_dir_real = f'data/{dataname}'
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    with open(f'{dataset_dir_real}/info.json', 'r') as f:
        info = json.load(f)

    task_type = info['task_type']
    x_num_1, x_cat_1, categories_1, d_numerical_1, num_i, cat_i = preprocess(dataset_dir_syn, task_type=task_type,
                                                                         inverse=True)

    dataset_dir_syn_csv = f'synthetic/{dataname}1/adult.csv'
    # x_cat, x_num = preprocess_new_dataset(dataset_dir_syn_csv, info)

    # print('x_num', x_num)
    # print('x_cat', x_cat)
    # print('num_i', num_i)
    # print('cat_i', cat_i)

    x_num_2, x_cat_2, categories_2, d_numerical_2, _, _ = preprocess(dataset_dir_real, task_type=task_type,
                                                                     inverse=True)

    print(d_numerical_1)
    print(d_numerical_2)
    # print('COMPARE -------------')
    # print(f'xnum: {np.array_equal(x_num_1, x_num_2)}\n')
    # print(f'xcat: {np.array_equal(x_cat_1, x_cat_2)}\n')
    # print('VAL -----------')
    # print(x_num_1)
    # print('VAL 2 --------------')
    # print(x_num_2)
    # print(f'categories: {categories_1 == categories_2}\n')
    # print(f'd numerical: {d_numerical_1 == d_numerical_2}\n')
    x_1, x_2 = x_num_1
    x_3, x_4 = x_cat_1
    # print(type(x_1))
    # print(type(x_2))
    # print(x_1.shape)
    # print(x_2.shape)
    # print(x_3.shape)
    x_num = np.concatenate((x_1, x_2), axis=0)
    x_cat = np.concatenate((x_3, x_4), axis=0)
    x_num = torch.tensor(x_num).float().to('cpu')
    x_cat = torch.tensor(x_cat).long().to('cpu')

    encoder = Encoder_model(2, d_numerical_2, categories_2, 4, n_head=1, factor=32)
    encoder_save_path = f'{curr_dir}/vae/ckpt/{dataname}/encoder.pt'

    saved_encoder_dict = torch.load(encoder_save_path, map_location=torch.device('cpu'))
    # embedding_weight = saved_encoder_dict['Tokenizer.category_embeddings.weight']
    # clipped_embedding_weight = embedding_weight[:85, :]
    # saved_encoder_dict['Tokenizer.category_embeddings.weight'] = clipped_embedding_weight

    encoder.load_state_dict(saved_encoder_dict)

    # x_num = num_i(x_num)
    # x_cat = cat_i(x_cat)
    inverted_latents = encoder(x_num, x_cat)

    # print('d_numerical_1', d_numerical_1)
    # print('x_num[0]', len(x_num[0]))
    # x_cat_tensor = torch.tensor(x_cat, dtype=torch.long, device=device)
    # x_num_tensor = torch.tensor(x_num, dtype=torch.float, device=device)

    # categories = info['cat_col_idx'] + info['target_col_idx']
    # cats = get_categories(x_cat_tensor)
    # print('categories', categories)
    # print('cats', cats)
    # encoder = Encoder_model(2, len(x_num[0]), cats, 4, n_head=1, factor=32)
    # encoder_save_path = f'tabsyn/vae/ckpt/{args.dataname}/encoder.pt'
    # encoder.load_state_dict(torch.load(encoder_save_path, map_location=device))

    # x_cat_tensor = torch.tensor(x_cat, dtype=torch.long, device=device)
    # x_num_tensor = torch.tensor(x_num, dtype=torch.float, device=device)
    # inverted_latents = encoder(x_num_tensor, x_cat_tensor)

    # print(inverted_latents.shape)
    inverted_latents = inverted_latents.detach().cpu()
    inverted_latents = inverted_latents[:, 1:, :]
    B, num_tokens, token_dim = inverted_latents.size()
    in_dim = num_tokens * token_dim

    inverted_latents = inverted_latents.view(B, in_dim)
    # print(x_next.shape)
    # print(inverted_latents.shape)

    reversed_noise = noise_scheduler.gen_reverse(model.noise_fn,
                                                 # latents=train_z,
                                                 latents=inverted_latents,
                                                 num_inference_steps=1)
    # is_watermarked, distance = detect(reversed_noise, key, channel, radius)
    is_watermarked = detect(reversed_noise, key, channel, radius)
    # print(f'Is watermarked {is_watermarked} \nDistance {distance}')
    # f.write(str(distance) + '\n')
    # f.write(f'Is watermarked: {is_watermarked}\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generation')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--epoch', type=int, default=None, help='Epoch.')
    parser.add_argument('--steps', type=int, default=None, help='Number of function evaluations.')
    parser.add_argument('--wm', type=str, default='None', help='The type of watermarking.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'
