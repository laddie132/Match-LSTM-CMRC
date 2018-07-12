#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import yaml
import logging
import argparse
from train import train
from test import test
from dataset import PreprocessCMRC
from utils import init_logging, read_config, ensemble_ans

init_logging()
logger = logging.getLogger(__name__)

EN_MODEL_NUM = 18
TMP_PATH = 'logs/'


def preprocess(config_path, test_json_path=None):
    logger.info('------------Preprocess CMRC dataset--------------')
    logger.info('loading config file...')
    global_config = read_config(config_path)

    logger.info('preprocess data...')
    pdata = PreprocessCMRC(global_config)

    if test_json_path is None:
        pdata.run()
    else:
        pdata.run_test(test_json_path)


def test_en(config_path, ans_out_path):
    logging.info('-----------ENSEMBLE MODEL PREDICT-----------')

    global_config = read_config(config_path)
    tmp_config_path = TMP_PATH + 'tmp_config.yaml'

    tmp_ans_pathes = []
    for i in range(1, EN_MODEL_NUM+1):
        logging.info('---------------MODEL No.%d/%d----------------' % (i, EN_MODEL_NUM))

        model_weight_path = 'data/ensemble-model/model-weight.pt-' + str(i)
        global_config['data']['model_path'] = model_weight_path
        with open(tmp_config_path, 'w') as f:
            yaml.dump(global_config, f)

        tmp_ans_path = TMP_PATH + 'ans.json-' + str(i)
        tmp_ans_pathes.append(tmp_ans_path)

        test(tmp_config_path, tmp_ans_path)
    ensemble_ans(tmp_ans_pathes, ans_out_path)


parser = argparse.ArgumentParser(description="preprocess/train/test the model")
parser.add_argument('mode', help='preprocess or train or test')
parser.add_argument('--config', '-c', required=False, dest='config_path', default='config/global_config.yaml')
parser.add_argument('--input', '-i', required=False, dest='in_path')
parser.add_argument('--output', '-o', required=False, dest='out_path')
args = parser.parse_args()


if args.mode == 'preprocess':
    preprocess(args.config_path)
elif args.mode == 'train':
    train(args.config_path)
elif args.mode == 'test':
    test(config_path=args.config_path, out_path=args.out_path)
elif args.mode == 'test_en':
    test_en(args.config_path, args.out_path)
elif args.mode == 'test_raw':
    preprocess(args.config_path, test_json_path=args.in_path)
    test(config_path=args.config_path, out_path=args.out_path)
elif args.mode == 'test_raw_en':
    preprocess(args.config_path, test_json_path=args.in_path)
    test_en(args.config_path, args.out_path)
else:
    raise ValueError('Unrecognized mode selected.')

