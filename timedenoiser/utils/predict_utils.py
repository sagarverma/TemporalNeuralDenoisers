import pickle

import numpy
import torch
import torch.nn as nn

from scipy.io import loadmat

from timedenoiser.utils.dataloader import normalize, denormalize
from timedenoiser.models.ffnn import ShallowFNN, DeepFNN
from timedenoiser.models.cnn import ShallowCNN, DeepCNN
from timedenoiser.models.rnn import ShallowRNN, DeepRNN
from timedenoiser.models.encdec import (EncDecSkip,
                                        EncDecRNNSkip,
                                        EncDecBiRNNSkip,
                                        EncDecDiagBiRNNSkip)

from motormetrics.ml import *
from motormetrics.ee import *


def get_loader_transform_types(model):
    if isinstance(model, ShallowFNN) or isinstance(model, DeepFNN):
        return 'flat', 'flat'
    if isinstance(model, ShallowRNN) or isinstance(model, DeepRNN) or \
       isinstance(model, EncDecSkip) or isinstance(model, EncDecRNNSkip) or \
       isinstance(model, EncDecBiRNNSkip) or \
       isinstance(model, EncDecDiagBiRNNSkip):
        return 'seq', 'seq'
    if isinstance(model, ShallowCNN) or isinstance(model, DeepCNN):
        return 'seq', 'flat'


def predict(speed_model, torque_model, data, window):
    metadata = {"min": {"voltage_d": -300,
                        "voltage_q": -300,
                        "current_d": -30,
                        "current_q": -30,
                        "noisy_voltage_d": -300,
                        "noisy_voltage_q": -300,
                        "noisy_current_d": -30,
                        "noisy_current_q": -30,
                        "torque": -120,
                        "speed": -80,
                        "statorPuls": -80},
                "max": {"voltage_d": 300,
                        "voltage_q": 300,
                        "current_d": 30,
                        "current_q": 30,
                        "noisy_voltage_d": 300,
                        "noisy_voltage_q": 300,
                        "noisy_current_d": 30,
                        "noisy_current_q": 30,
                        "torque": 120,
                        "speed": 80,
                        "statorPuls": 80}}
                        
    inp_trf_typ, out_trf_typ = get_loader_transform_types(speed_model)

    inp_quants = ['noisy_voltage_d', 'noisy_voltage_q', 'noisy_current_d', 'noisy_current_q']

    inp_data = []
    for inp_quant in inp_quants:
        quantity = data[inp_quant]
        minn = metadata['min'][inp_quant]
        maxx = metadata['max'][inp_quant]
        inp_data.append(normalize(quantity, minn, maxx))

    inp_data = np.asarray(inp_data)

    out_data = []
    for out_quant in ['speed', 'torque']:
        quantity = data[out_quant]
        minn = metadata['min'][out_quant]
        maxx = metadata['max'][out_quant]
        out_data.append(normalize(quantity, minn, maxx))

    out_data = np.asarray(out_data)

    samples = []
    for i in range(inp_data.shape[1]):
        if (i + window) < inp_data.shape[1]:
            inp_data_mod = inp_data[:, i:i+window]
            if inp_trf_typ == 'flat':
                inp_data_mod = inp_data_mod.flatten()
            samples.append(inp_data_mod)

    speed_preds = []
    torque_preds = []
    for i in range(0, len(samples), 1000):
        batch_inp = torch.tensor(np.asarray(samples[i:i+1000])).float().cuda(0)
        speed_out = speed_model(batch_inp)
        torque_out = torque_model(batch_inp)
        speed_preds.append(speed_out.data.cpu().numpy())
        torque_preds.append(torque_out.data.cpu().numpy())

    speed_preds = np.concatenate(speed_preds, axis=0)
    torque_preds = np.concatenate(torque_preds, axis=0)

    if out_trf_typ == 'seq':
        speed_preds = speed_preds[:, 0, window//2].flatten()
        torque_preds = torque_preds[:, 0, window//2].flatten()
    if out_trf_typ == 'flat':
        speed_preds = speed_preds[:, 0].flatten()
        torque_preds = torque_preds[:, 0].flatten()

    speed_true = out_data[0, :].flatten()
    torque_true = out_data[1, :].flatten()

    speed_preds = np.concatenate((speed_true[:window//2], speed_preds,
                                  speed_true[-1 * window//2:]), axis=0)
    torque_preds = np.concatenate((torque_true[:window//2], torque_preds,
                                   torque_true[-1 * window//2:]), axis=0)

    minn = metadata['min']['speed']
    maxx = metadata['max']['speed']
    speed_denormed = denormalize(speed_preds, minn, maxx)
    speed_true = denormalize(speed_true, minn, maxx)

    minn = metadata['min']['torque']
    maxx = metadata['max']['torque']
    torque_denormed = denormalize(torque_preds, minn, maxx)
    torque_true = denormalize(torque_true, minn, maxx)

    speed_ml_metrics = {}
    speed_ml_metrics['smape'] = smape(speed_true, speed_denormed)
    speed_ml_metrics['r2'] = r2(speed_true, speed_denormed)
    speed_ml_metrics['rmse'] = rmse(speed_true, speed_denormed)
    speed_ml_metrics['mae'] = mae(speed_true, speed_denormed)

    torque_ml_metrics = {}
    torque_ml_metrics['smape'] = smape(torque_true, torque_denormed)
    torque_ml_metrics['r2'] = r2(torque_true, torque_denormed)
    torque_ml_metrics['rmse'] = rmse(torque_true, torque_denormed)
    torque_ml_metrics['mae'] = mae(torque_true, torque_denormed)

    return speed_denormed, torque_denormed, speed_ml_metrics, torque_ml_metrics


def load_data(opt):
    data = loadmat(opt.benchmark_file)
    return data


def load_model(opt):
    speed_model = torch.load(opt.speed_model_file).cuda(0)
    speed_model = speed_model.eval()

    torque_model = torch.load(opt.torque_model_file).cuda(0)
    torque_model = torque_model.eval()

    return speed_model, torque_model
