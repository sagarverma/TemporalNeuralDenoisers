import os
import argparse
import pickle
from copy import deepcopy
from scipy.io import savemat

from timedenoiser.utils.predict_utils import (load_model, load_data,
                                              predict)


def get_arg_parse():
    parser = argparse.ArgumentParser(description='Test on custom benchmark.')
    parser.add_argument('--speed_model_file', required=True, type=str)
    parser.add_argument('--torque_model_file', required=True, type=str)
    parser.add_argument('--benchmark_file', type=str,
                        required=True, help='benchmark file')
    parser.add_argument('--window', type=int,
                        required=True, help='input window')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='directory where results are saved')
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--noise', type=bool, default=False)
    parser.add_argument('--out_name', type=str, required=True)
    args = parser.parse_args()
    return args


args = get_arg_parse()
speed_model, torque_model = load_model(args)
data = load_data(args)
out = deepcopy(data)

speed_denormed, torque_denormed, speed_ml_metrics, torque_ml_metrics = \
        predict(speed_model, torque_model, data, args.window)

print(args.speed_model_file.split('/')[-1][:30], args.benchmark_file.split('/')[-1])


save_dir = os.path.join(args.save_dir, args.benchmark_file.split('/')[-1].split('.')[0])
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

to_dump = {'pred_speed': speed_denormed,
           'pred_torque': torque_denormed}
# fout = open(os.path.join(save_dir, args.out_name + '.pkl'), 'wb')
# pickle.dump({**to_dump, **out}, fout)
# fout.close()

savemat(os.path.join(save_dir, args.out_name + '.mat'), {**to_dump, **out})
