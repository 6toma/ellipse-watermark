import os
import torch

import argparse
import warnings
import time
import copy
import hashlib


from tabsyn.model import MLPDiffusion, Model, DDIMModel, DDIMScheduler
from tabsyn.latent_utils import get_input_generate, recover_data, split_num_cat_target
from tabsyn.watermark_utils import get_noise

warnings.filterwarnings('ignore')


def main(args):
    save_path_arg = args.save_path
    dataname = args.dataname
    device = args.device
    steps = args.steps
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
    init_latents = torch.randn([num_samples, sample_dim], device=device)

    # watermarking
    if with_w is not None:
        latents_1 = init_latents.to(device)

        print('sampling with watermark - ' + with_w)
        # change from two-dimensional table into watermark size [1, c, l, w]
        init_latents = init_latents.unsqueeze(0).unsqueeze(0)
        init_latent_w = copy.deepcopy(init_latents).to(device)

        watermarked_latents, key, channel, radius = get_noise(init_latent_w.shape, pattern=with_w)
        key.to(device)

        # back to two-dimensional
        latents = watermarked_latents.squeeze(0).squeeze(0)
        latents_2 = latents.to(device)

        # saving gt_patch and watermarking_mask
        keys = torch.load(f'{save_path_arg}/keys_tree.pt', map_location=torch.device('cpu'))

        hex_dig = hashlib.sha256(key.numpy().tobytes()).hexdigest()
        key_name = "_".join([hex_dig, str(channel), with_w])

        keys[key_name] = key

        torch.save(keys, f'{save_path_arg}/keys_tree.pt')
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

    # Saving the synthetic csv
    x_next_dict = {'no-w': x_next_1, 'w': x_next_2}
    for k in x_next_dict.keys():
        if with_w is None:
            save_dir = f'{save_path_arg}/{k}'
        else:
            save_dir = f'{save_path_arg}/{k}-{with_w}'

        if not os.path.exists(save_dir):
            # If not, create it
            os.makedirs(save_dir)

        save_path = f'{save_dir}/{dataname}.csv'
        x_next = x_next_dict[k]
        x_next = x_next * 2 + mean.to(device)

        syn_data = x_next.float().cpu().numpy()
        syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, cat_inverse, args.device)
        syn_df = recover_data(syn_num, syn_cat, syn_target, info)

        idx_name_mapping = info['idx_name_mapping']
        idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

        syn_df.rename(columns = idx_name_mapping, inplace=True)
        syn_df.to_csv(save_path, index = False)

        end_time = time.time()
        print('Time:', end_time - start_time)
        print('Saving sampled data to {}'.format(save_path))



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