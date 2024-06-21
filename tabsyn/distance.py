import os
import torch

import argparse
import warnings
import pandas as pd

from tabsyn.model import MLPDiffusion, DDIMModel, DDIMScheduler
from tabsyn.latent_utils import get_input_generate, get_encoder_latent
from tabsyn.watermark_utils import detect
from tabsyn.process_syn_dataset import process_data, preprocess_syn
from tabsyn.attacks import add_random_noise_cat, add_random_noise_num, delete_columns, delete_rows
import time


warnings.filterwarnings('ignore')


def main(args):
    dataname = args.dataname
    device = args.device
    data_dir = args.data_dir
    steps = args.steps
    steps = 1000
    device = 'mps'

    with_w = args.wm

    train_z, _, _, ckpt_path, info, num_inverse, cat_inverse = get_input_generate(args)
    in_dim = train_z.shape[1]
    num_samples = train_z.shape[0]

    mean = train_z.mean(0)

    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)

    # DDIM
    model = DDIMModel(denoise_fn).to(device)

    model.load_state_dict(torch.load(f'{ckpt_path}/model.pt', map_location=torch.device('cpu')))

    noise_scheduler = DDIMScheduler(num_train_timesteps=1000)

    # data_path = f'{data_dir}/{dataname}.csv'

    start_time = time.time()

    attacks = [
        'add_noise_num',
        'add_noise_cat',
        'delete_rows',
        'delete_cols'
    ]

    num_cols = {
        'abalone': 9,
        'adult': 15,
        'default': 24,
        'diabetes': 9
    }

    for attack in attacks:
        for attack_strength in [0.1, 0.25]:
            with (open(f'dist_syn_wm_tree_{with_w}_{attack}_{attack_strength}.txt', 'w') as g):
                for i in range(100):
                    data_path = f'{data_dir}/{dataname}.csv'
                    print('iteration --- ' + str(i) + ' --- attack --- ' + attack +
                          ' ' + str(attack_strength) + '--- ' + with_w)
                    torch.manual_seed(i)

                    # attack here
                    df = pd.read_csv(data_path)
                    if attack == 'add_noise_num':
                        df = add_random_noise_num(df, attack_strength)
                        print(' -- added random noise num')
                    if attack == 'add_noise_cat':
                        df = add_random_noise_cat(df, attack_strength)
                        print(' -- added random noise cat')
                    if attack == 'delete_rows':
                        df = delete_rows(df, attack_strength, i)
                    if attack == 'delete_cols':
                        df = delete_columns(df, attack_strength, i)

                    new_data_path = f'{data_dir}/{dataname}_attacked.csv'
                    df.to_csv(new_data_path, index=False)
                    data_path = new_data_path
                    # end attacks here

                    num_rows = min(1000, num_samples)
                    process_data(dataname, data_path, data_dir, expected_shape=[num_rows, num_cols[dataname]])

                    task_type = 'regression' if 'abalone' in dataname else 'binclass'
                    x_num, x_cat = preprocess_syn(data_dir, task_type)

                    encoded_latents = get_encoder_latent(x_num, x_cat, info, device)

                    recovered_latent = noise_scheduler.gen_reverse(
                        model.noise_fn,
                        encoded_latents,
                        num_inference_steps=steps,
                        eta=0.0
                    )

                    recovered_latent = recovered_latent.unsqueeze(0).unsqueeze(0)

                    thresholds_path = os.path.join(data_dir, '../thresholds.pt')
                    thresholds_path = os.path.normpath(thresholds_path)

                    thresholds = torch.load(thresholds_path, map_location=torch.device('cpu'))

                    keys_path = os.path.join(data_dir, '../keys_tree.pt')
                    keys_path = os.path.normpath(keys_path)

                    keys = torch.load(keys_path, map_location=torch.device('cpu'))

                    wm_present = False
                    w_pattern = with_w
                    for key_name, w_key in keys.items():
                        w_channel = key_name.split("_")[1]
                        pattern = key_name.split("_")[2]
                        if w_pattern != pattern:
                            continue
                        threshold = thresholds[w_pattern]

                        is_watermarked, dist = detect(recovered_latent, w_key, w_channel, threshold)
                        if is_watermarked:
                            wm_present = True
                            break

                    print('Is watermarked: ', wm_present)
                    g.write(str(dist) + '\n')

                    end_time = time.time()
                    print('Time:', end_time - start_time)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generation')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--data_dir', type=str, default=None, help='Path to data directory.')
    parser.add_argument('--steps', type=int, default=1, help='Number of function evaluations.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'
