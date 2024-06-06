import os
import torch

import argparse
import warnings
import time
import copy
import numpy as np
import hashlib

import wandb

from tabsyn.model import MLPDiffusion, Model, DDIMModel, DDIMScheduler
from tabsyn.latent_utils import get_input_generate, recover_data, split_num_cat_target, get_encoder_latent
# from tabsyn.watermark_utils import get_watermarking_mask, inject_watermark, get_watermarking_pattern, eval_watermark
from tabsyn.watermark_utils_2 import get_noise, detect
from tabsyn.process_syn_dataset import process_data, preprocess_syn, is_processed

warnings.filterwarnings('ignore')


def main(args):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    save_path_arg = args.save_path
    # TREERING
    # save_dir = f'{curr_dir}/../{save_path_arg}/treering/{i}'
    # if not os.path.exists(save_dir):
        # If it doesn't exist, create it
        # os.makedirs(save_dir, exist_ok=True)

    dataname = args.dataname
    device = args.device
    steps = args.steps
    # steps = 1
    device = "mps"
    with_w = args.wm

    train_z, _, _, ckpt_path, info, num_inverse, cat_inverse = get_input_generate(args)
    in_dim = train_z.shape[1] 

    mean = train_z.mean(0)

    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    
    # Score-based
    # model = Model(denoise_fn = denoise_fn, hid_dim = train_z.shape[1]).to(device)

    # DDIM
    model = DDIMModel(denoise_fn).to(device)

    model.load_state_dict(torch.load(f'{ckpt_path}/model.pt', map_location=torch.device('mps')))

    '''
        Generating samples    
    '''
    start_time = time.time()

    num_samples = train_z.shape[0]
    sample_dim = in_dim
    # torch.manual_seed(i)
    print(device)
    with (open(f'distances_syn_{with_w}.txt', 'w') as f):
        with (open(f'distances_real_{with_w}.txt', 'w') as g):
            for i in range(1):
                init_latents = torch.randn([num_samples, sample_dim], device=device)

                # watermarking
                # if with_w == 'treering':
                if with_w is not None:
                    latents_1 = init_latents.to(device)

                    print('sampling with watermark - ' + with_w)
                    # change from two-dimensional table into watermark size [1, c, l, w]
                    init_latents = init_latents.unsqueeze(0).unsqueeze(0)
                    init_latent_w = copy.deepcopy(init_latents).to(device)

                    # # ground-truth patch
                    # gt_patch = get_watermarking_pattern(args, device, shape=init_latent_w.shape, seed=i)
                    # # get watermarking mask
                    # watermarking_mask = get_watermarking_mask(init_latent_w, args, device)
                    # # inject watermark
                    # latents = inject_watermark(init_latent_w, watermarking_mask, gt_patch, args)
                    watermarked_latents, key, channel, radius = get_noise(init_latent_w.shape, pattern=with_w)
                    key.to(device)
                    # back to two-dimensional
                    latents = watermarked_latents.squeeze(0).squeeze(0)
                    latents_2 = latents.to(device)
                    # latents_2 = watermarked_latents.to(device)
                    # keys = torch.load(f'{save_path_arg}/keys.pt', map_location=torch.device('cpu'))
                    #
                    # hex_dig = hashlib.sha256(key.numpy().tobytes()).hexdigest()
                    # key_name = "_".join([hex_dig, str(channel), str(radius), with_w])
                    #
                    # keys[key_name] = key
                    #
                    # torch.save(keys, f'{save_path_arg}/keys.pt')

                    # saving gt_patch and watermarking_mask
                    # np_gt_patch = gt_patch.detach().cpu().numpy()
                    # np_watermarking_mask = watermarking_mask.detach().cpu().numpy()
                    # np.save(f'{save_dir}/gt_patch.npy', np_gt_patch)
                    # np.save(f'{save_dir}/watermarking_mask.npy', np_watermarking_mask)
                    # is_watermarked, dist = detect(watermarked_latents, key, channel, radius)
                    # dist = torch.abs(init_latent_w - latents_2).mean().item()
                    # print('Distance after wm: ', dist)
                else:
                    print('sampling without watermark')
                    latents_1 = init_latents.to(device)

                # Score-based
                # x_next = sample(model.denoise_fn_D, latents, args=args)

                # DDIM no watermark
                noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
                x_next_1 = noise_scheduler.generate(
                        model.noise_fn,
                        latents_1,
                        num_inference_steps=steps,
                        eta=0.0)

                # DDIM watermark
                x_next_2 = noise_scheduler.generate(
                        model.noise_fn,
                        latents_2,
                        num_inference_steps=steps,
                        eta=0.0)



                # is_watermarked, dist = detect(x_next_2, key, channel, radius)
                # dist = torch.abs(init_latent_w - x_next_2).mean().item()
                # print('Distance after wm after DDIM: ', dist)
                # x_next_2 = x_next_2 * 2 + mean.to(device)

                #
                # x_next_2 = (x_next_2 - mean.to(device)) / 2


                #
                # recovered_latent = noise_scheduler.gen_reverse(
                #     model.noise_fn,
                #     x_next_2,
                #     num_inference_steps=steps,
                #     eta=0.0
                # )
                #
                # # recovered_latent = (recovered_latent - mean.to(device)) / 2
                #
                # is_watermarked, dist = detect(recovered_latent, key, channel, radius)
                # # dist = torch.abs(init_latent_w - recovered_latent).mean().item()
                # print('Distance after wm after DDIM after DDIM Inverse: ', dist)

                # Saving the synthetic csv
                x_next_dict = {'no-w': x_next_1, 'w': x_next_2}
                # x_next_dict = {'w': x_next_2}
                distances = {}
                for k in x_next_dict.keys():
                    print(k)
                    # save_path = f'{save_dir}/{k}-{args.method}.csv'
                    if with_w is None:
                        save_dir = f'{save_path_arg}/{k}'
                    else:
                        save_dir = f'{save_path_arg}/{k}-{with_w}'
                    if not os.path.exists(save_dir):
                        # If not, create it
                        os.makedirs(save_dir)
                    save_path = f'{save_dir}/{dataname}.csv'
                    x_next = x_next_dict[k]
                    #
                    # # recovered_latent = noise_scheduler.gen_reverse(
                    # #     model.noise_fn,
                    # #     x_next,
                    # #     num_inference_steps=steps,
                    # #     eta=0.0
                    # # )
                    # # recovered_latent = recovered_latent.unsqueeze(0).unsqueeze(0)
                    # # is_watermarked = detect(recovered_latent, key, channel, radius)
                    # # recovered_latent = recovered_latent.squeeze(0).squeeze(0)
                    # # print('is_watermarked: ', is_watermarked)
                    #
                    x_next = x_next * 2 + mean.to(device)
                    #
                    # # recovered_latent = noise_scheduler.gen_reverse(
                    # #     model.noise_fn,
                    # #     x_next,
                    # #     num_inference_steps=steps,
                    # #     eta=0.0
                    # # )
                    # # recovered_latent = recovered_latent.unsqueeze(0).unsqueeze(0)
                    # # is_watermarked = detect(recovered_latent, key, channel, radius)
                    # # recovered_latent = recovered_latent.squeeze(0).squeeze(0)
                    # # print('is_watermarked: ', is_watermarked)
                    #
                    # print('before saving csv')
                    syn_data = x_next.float().cpu().numpy()
                    # print('before splitting')
                    syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, cat_inverse, args.device)

                    # print('before recovering data')
                    syn_df = recover_data(syn_num, syn_cat, syn_target, info)

                    idx_name_mapping = info['idx_name_mapping']
                    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

                    # print('before saving to file')
                    syn_df.rename(columns = idx_name_mapping, inplace=True)
                    syn_df.to_csv(save_path, index = False)

                    # encode and check distance
                    # save_dir = 'synthetic/adult/real'
                    # save_path = 'synthetic/adult/real/adult.csv'
                    if not is_processed(save_dir):
                        # process_data(dataname, save_path, save_dir, k)
                        process_data(dataname, save_path, save_dir)

                    # x_num, x_cat = preprocess_syn(save_dir, k=k)
                    x_num, x_cat = preprocess_syn(save_dir)

                    encoded_latents = get_encoder_latent(x_num, x_cat, info, device)

                    # encoded_latents = train_z

                    # encoded_latents = (encoded_latents - mean.to(device)) / 2

                    # is_watermarked, dist = detect(encoded_latents, key, channel, radius)
                    # dist = torch.abs(init_latent_w - x_next_2).mean().item()
                    # print('Distance after wm after DDIM before inversion: ', dist)

                    encoded_latents = (encoded_latents - mean.to(device)) / 2

                    # print('look here v')
                    # _, _ = detect(encoded_latents, key, channel, radius)

                    recovered_latent = noise_scheduler.gen_reverse(
                        model.noise_fn,
                        encoded_latents,
                        num_inference_steps=steps,
                        eta=0.0
                    )

                    # recovered_latent = (recovered_latent - mean.to(device)) / 2
                    recovered_latent = recovered_latent.unsqueeze(0).unsqueeze(0)
                    is_watermarked, dist_1 = detect(recovered_latent, key, channel, radius)
                    # is_watermarked, dist_1 = detect(recovered_latent, key, channel, radius)
                    print(f'{k} is_watermarked: ', is_watermarked)

                    distances[k] = dist_1

                    # if 'no' not in k:
                    # f.write(str(dist) + '\n')
                    # else:
                    #     g.write(str(dist) + '\n')

                    # save_dir = 'synthetic/adult/real'
                    # save_path = 'synthetic/adult/real/adult.csv'
                    # if not is_processed(save_dir):
                    #     # process_data(dataname, save_path, save_dir, k)
                    #     process_data(dataname, save_path, save_dir)
                    #
                    # # x_num, x_cat = preprocess_syn(save_dir, k=k)
                    # x_num, x_cat = preprocess_syn(save_dir)
                    #
                    # encoded_latents = get_encoder_latent(x_num, x_cat, info, device)
                    #
                    # # encoded_latents = train_z
                    #
                    # encoded_latents = (encoded_latents - mean.to(device)) / 2
                    #
                    # recovered_latent = noise_scheduler.gen_reverse(
                    #     model.noise_fn,
                    #     encoded_latents,
                    #     num_inference_steps=steps,
                    #     eta=0.0
                    # )
                    #
                    # # recovered_latent = (recovered_latent - mean.to(device)) / 2
                    # # recovered_latent = recovered_latent.unsqueeze(0).unsqueeze(0)
                    # # is_watermarked, dist_2 = detect(recovered_latent, key, channel, radius)
                    # is_watermarked, dist_2 = detect(recovered_latent, key, channel, radius)
                    # print(f'{k} is_watermarked: ', is_watermarked)

                    # g.write(str(dist) + '\n')

                    end_time = time.time()
                    print('Time:', end_time - start_time)
                    print('Saving sampled data to {}'.format(save_path))

                print('\n')
                print(f'Wm latents distance: {distances["w"]}')
                print(f'Non wm latents distance: {distances["no-w"]}')

                if distances['w'] < distances['no-w']:
                    print('Congratulations! GOOD JOB')
                else:
                    print('THIS IS BAD')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generation')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--epoch', type=int, default=None, help='Epoch.')
    parser.add_argument('--steps', type=int, default=None, help='Number of function evaluations.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'mps'

    args.device = 'mps'