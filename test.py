#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import json
import os
import torch
import logging
import argparse
from dataset import Dataset
from models import *
from utils import init_logging, read_config, eval_on_model, beam_search
from models.loss import MyNLLLoss

logger = logging.getLogger(__name__)


def test(config_path, out_path):
    logger.info('------------MODEL PREDICT--------------')
    logger.info('loading config file...')
    global_config = read_config(config_path)

    # set random seed
    seed = global_config['global']['random_seed']
    torch.manual_seed(seed)

    enable_cuda = global_config['test']['enable_cuda']
    device = torch.device("cuda" if enable_cuda else "cpu")
    if torch.cuda.is_available() and not enable_cuda:
        logger.warning("CUDA is avaliable, you can enable CUDA in config file")
    elif not torch.cuda.is_available() and enable_cuda:
        raise ValueError("CUDA is not abaliable, please unable CUDA in config file")

    torch.set_grad_enabled(False)  # make sure all tensors below have require_grad=False,

    logger.info('reading dataset...')
    dataset = Dataset(global_config)

    logger.info('constructing model...')
    dataset_h5_path = global_config['data']['dataset_h5']

    model = MatchLSTMPlus(dataset_h5_path)
    model = model.to(device)
    model.eval()  # let training = False, make sure right dropout
    criterion = MyNLLLoss()

    model_rerank = None
    rank_k = global_config['global']['rank_k']
    if global_config['global']['enable_rerank']:
        model_rerank = ReRanker(dataset_h5_path)
        model_rerank = model_rerank.to(device)
        model_rerank.eval()
        criterion = torch.nn.NLLLoss()

    # load model weight
    logger.info('loading model weight...')
    model_weight_path = global_config['data']['model_path']
    assert os.path.exists(model_weight_path), "not found model weight file on '%s'" % model_weight_path

    weight = torch.load(model_weight_path, map_location=lambda storage, loc: storage)
    if enable_cuda:
        weight = torch.load(model_weight_path, map_location=lambda storage, loc: storage.cuda())
    model.load_state_dict(weight, strict=False)

    if global_config['global']['enable_rerank']:
        rerank_weight_path = global_config['data']['rerank_model_path']
        assert os.path.exists(rerank_weight_path), "not found rerank model weight file on '%s'" % rerank_weight_path
        logger.info('loading rerank model weight...')
        weight = torch.load(rerank_weight_path, map_location=lambda storage, loc: storage)
        if enable_cuda:
            weight = torch.load(rerank_weight_path, map_location=lambda storage, loc: storage.cuda())
        model_rerank.load_state_dict(weight, strict=False)

    # forward
    logger.info('forwarding...')

    batch_size = global_config['test']['batch_size']
    num_workers = global_config['global']['num_data_workers']

    batch_dev_data = dataset.get_dataloader_dev(batch_size, num_workers)

    # to just evaluate score or write answer to file
    if out_path is None:
        score_em, score_f1, sum_loss = eval_on_model(model=model,
                                                     criterion=criterion,
                                                     batch_data=batch_dev_data,
                                                     epoch=None,
                                                     device=device,
                                                     model_rerank=model_rerank,
                                                     rank_k=rank_k)
        logger.info("test: ave_score_em=%.2f, ave_score_f1=%.2f, sum_loss=%.5f" % (score_em, score_f1, sum_loss))
    else:
        context_right_space = dataset.get_all_ct_right_space_dev()
        predict_ans = predict_on_model(model=model,
                                       batch_data=batch_dev_data,
                                       device=device,
                                       id_to_word_func=dataset.sentence_id2word,
                                       right_space=context_right_space,
                                       model_rerank=model_rerank,
                                       rank_k=rank_k)
        samples_id = dataset.get_all_samples_id_dev()
        ans_with_id = dict(zip(samples_id, predict_ans))

        logging.info('writing predict answer to file %s' % out_path)
        with open(out_path, 'w') as f:
            json.dump(ans_with_id, f)

    logging.info('finished.')


def predict_on_model(model, batch_data, device, id_to_word_func, right_space, model_rerank, rank_k):
    batch_cnt = len(batch_data)
    answer = []

    cnt = 0
    for bnum, batch in enumerate(batch_data):
        batch = [x.to(device) if x is not None else x for x in batch]
        bat_context = batch[0]
        bat_answer_range = batch[-1]

        # forward
        batch_input = batch[:len(batch) - 1]
        tmp_ans_prop, tmp_ans_range, _ = model.forward(*batch_input)

        # if model_rerank is not None:
        cand_ans_range, cand_ans_prop = beam_search(tmp_ans_prop, k=rank_k)

            # context = batch_input[0]
            # question = batch_input[1]
            # cand_score, tmp_ans_range = model_rerank(context, question, cand_ans_range)

        tmp_context_ans = zip(bat_context.cpu().data.numpy(),
                              cand_ans_range.cpu().data.numpy())

        # generate initial answer text
        i = 0
        for c, k_a in tmp_context_ans:
            cur_no = cnt + i

            k_ans = []
            for k in range(k_a.shape[0]):
                a = k_a[k]
                tmp_ans = id_to_word_func(c[a[0]:(a[1] + 1)])
                cur_space = right_space[cur_no][a[0]:(a[1] + 1)]

                cur_ans = ''
                for j, word in enumerate(tmp_ans):
                    cur_ans += word
                    if cur_space[j]:
                        cur_ans += ' '
                k_ans.append(cur_ans.strip())
            k_ans_prop = dict(zip(k_ans, cand_ans_prop[i].cpu().data.numpy().tolist()))
            answer.append(k_ans_prop)
            i += 1
        cnt += i
        logging.info('batch=%d/%d' % (bnum, batch_cnt))

    return answer


if __name__ == '__main__':
    init_logging()

    parser = argparse.ArgumentParser(description="evaluate on the model")
    parser.add_argument('--config', '-c', required=False, dest='config_path', default='config/global_config.yaml')
    parser.add_argument('--output', '-o', required=False, dest='out_path')
    args = parser.parse_args()

    test(config_path=args.config_path, out_path=args.out_path)
