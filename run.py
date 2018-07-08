#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import logging
import argparse
from train import train
from test import test
from dataset import PreprocessCMRC
from utils.load_config import init_logging, read_config

init_logging()
logger = logging.getLogger(__name__)


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
elif args.mode == 'test_raw':
    preprocess(args.config_path, test_json_path=args.in_path)
    test(config_path=args.config_path, out_path=args.out_path)
else:
    raise ValueError('Unrecognized mode selected.')

