#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"



import os
import torch
import logging
import argparse
import time
import json
from tornado import web, ioloop

from models import *
from dataset import Dataset, DocTextCh
from utils.load_config import init_logging, read_config
from utils.functions import count_parameters

init_logging()
logger = logging.getLogger(__name__)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def use_logging(method):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            logger.info('========== Calling {} Start ========'.format(method))
            rtn = func(*args, **kwargs)
            end = time.time()
            logger.info('========== Calling {} End, time cost: {} (s) ========'.format(method, end-start))
            return rtn
        return wrapper

    return decorator


class MRCService(metaclass=Singleton):
    def __init__(self, enable_cuda=False):
        self.init_env(enable_cuda)
        self._construct_model()
        self._load_checkpoint()

    def init_env(self, enable_cuda):
        logger.info('loading config file...')
        self.config = read_config('config/global_config.yaml')

        # set random seed
        seed = self.config['global']['random_seed']
        torch.manual_seed(seed)

        torch.set_grad_enabled(False)  # make sure all tensors below have require_grad=False
        self.device = torch.device("cuda" if enable_cuda else "cpu")

        logger.info('reading dataset...')
        self.dataset = Dataset(self.config)

    def _construct_model(self):
        logger.info('constructing model...')
        dataset_h5_path = self.config['data']['dataset_h5']
        self.model = MatchLSTMPlus(dataset_h5_path)
        self.model.eval()  # let training = False, make sure right dropout
        self.model.to(self.device)

        logging.info('model parameters count: %d' % count_parameters(self.model))

    def _load_checkpoint(self):
        logger.info('loading model weight...')
        model_weight_path = self.config['data']['model_path']
        is_exist_model_weight = os.path.exists(model_weight_path)
        assert is_exist_model_weight, "not found model weight file on '%s'" % model_weight_path

        weight = torch.load(model_weight_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(weight, strict=False)

    def _example_to_batch(self, context, question):
        preprocess_config = self.config['preprocess']
        context_doc = DocTextCh(context, preprocess_config)
        question_doc = DocTextCh(question, preprocess_config)

        context_doc.update_em(question_doc)
        question_doc.update_em(context_doc)

        context_id_char = None
        question_id_char = None

        context_id, context_f = context_doc.to_id(self.dataset.meta_data)
        question_id, question_f = question_doc.to_id(self.dataset.meta_data)

        bat_input = [context_id, question_id, context_id_char, question_id_char, context_f, question_f]
        bat_input = [x.unsqueeze(0).to(self.device) if x is not None else x for x in bat_input]

        return bat_input

    @use_logging('mrc_predict')
    def predict(self, context, question):
        batch_input = self._example_to_batch(context, question)
        out_ans_prop, out_ans_range, vis_param = self.model.forward(*batch_input)

        out_ans_range = out_ans_range.cpu().numpy()
        start = out_ans_range[0][0]
        end = out_ans_range[0][1] + 1

        context_id = batch_input[0][0].cpu().numpy()
        out_answer_id = context_id[start:end]
        out_answer = self.dataset.sentence_id2word(out_answer_id)

        out_answer = ''.join(out_answer)
        return out_answer


class MRCHandler(web.RequestHandler):
    def initialize(self, cuda):
        self.service = MRCService(enable_cuda=cuda)

    def get(self):
        context = self.get_argument('context')
        question = self.get_argument('question')

        answer = self.service.predict(context, question)

        rtn = json.dumps({'answer': answer}, ensure_ascii=False)
        self.write(rtn)

    def post(self):
        context = self.get_body_argument('context')
        question = self.get_body_argument('question')

        answer = self.service.predict(context, question)

        rtn = json.dumps({'answer': answer}, ensure_ascii=False)
        self.write(rtn)


class IndexHandler(web.RequestHandler):
    def get(self):
        self.write('Hello World :)')


def add_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Machine Reading Comprehension Service\n'
                                                 'Contact: liuhan132@foxmail.com')
    parser.add_argument('-P', '--port', help='port', default=9999, required=False)
    parser.add_argument('-A', '--address', help='address', default='0.0.0.0', required=False)
    parser.add_argument('-L', '--log', help='log path', default='logs/',
                        required=False)
    parser.add_argument('--cuda', help='enable_cuda', action='store_true', default=False)

    args = parser.parse_args()
    return args


def run():
    logger.info('-------------MRC SERVICE--------------')
    args = add_args()

    handlers = [
        (r'/api/mrc', MRCHandler, dict({'cuda': args.cuda})),
        (r'^/$', IndexHandler),
    ]

    app = web.Application(handlers)

    logger.info('listening at "http://%s:%d"' % (args.address, args.port))
    server = app.listen(args.port, args.address)
    ioloop.IOLoop.instance().start()


if __name__ == '__main__':
    run()
