#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import torch
from models.layers import *
from utils import generate_mask


class ReRanker(torch.nn.Module):
    """
    match-lstm+ model for machine comprehension
    Args:
        - global_config: model_config with types dictionary

    Inputs:
        context: (batch, seq_len)
        question: (batch, seq_len)
        candidate_answer: (batch, k, ans_len)

    Outputs:
        candidate_prop: (batch, k)
        out_answer: (batch, ans_len)
    """

    def __init__(self, dataset_h5_path):
        super(ReRanker, self).__init__()

        word_embedding_size = 300
        hidden_size = 150
        hidden_mode = 'GRU'
        emb_dropout_p = 0.1
        dropout_p = 0.2

        encoder_bidirection = True
        encoder_direction_num = 2 if encoder_bidirection else 1

        measure_bidirection = True
        measure_direction_num = 2 if measure_bidirection else 1

        self.embedding = GloveEmbedding(dataset_h5_path=dataset_h5_path)
        self.encoder = MyRNNBase(mode=hidden_mode,
                                 input_size=word_embedding_size,
                                 hidden_size=hidden_size,
                                 bidirectional=encoder_bidirection,
                                 dropout_p=emb_dropout_p)
        encoder_out_size = hidden_size * encoder_direction_num

        self.seq_to_seq_att = SeqToSeqAtten()
        self.linear_fusion = LinearFusion(encoder_out_size,
                                          dropout_p=dropout_p)
        self.measure_rnn = MyRNNBase(mode=hidden_mode,
                                     input_size=encoder_out_size,
                                     hidden_size=hidden_size,
                                     bidirectional=measure_bidirection,
                                     dropout_p=dropout_p)
        measure_out_size = hidden_size * measure_direction_num
        self.linear_log_softmax = LinearLogSoftmax(measure_out_size)

    def forward(self, context, question, candidate_answer):
        # word-level embedding: (seq_len, batch, embedding_size)
        context_vec, context_mask = self.embedding.forward(context)
        question_vec, question_mask = self.embedding.forward(question)

        # word-level encode: (seq_len, batch, hidden_size)
        context_encode, _ = self.encoder.forward(context_vec, context_mask)
        question_encode, _ = self.encoder.forward(question_vec, question_mask)

        question_length = question_mask.eq(1).long().sum(1)

        batch, k, _ = candidate_answer.shape
        cand_s = []
        for i in range(k):

            qus_ans = []
            qus_ans_len = []
            cand_ans = candidate_answer[:, i, :]    # (batch, ans_len)
            for j in range(batch):
                cur_ans = cand_ans[j, :]
                cur_ans_encode = context_encode[cur_ans[0]:(cur_ans[1]+1), j, :]

                # question de-pad
                cur_qus_length = question_length[j]
                cur_qus_encode = question_encode[:cur_qus_length, j, :]

                cur_qus_ans = torch.cat([cur_ans_encode, cur_qus_encode], dim=0)   # (ans_len+qus_len, hidden_size)
                qus_ans.append(cur_qus_ans)
                qus_ans_len.append(cur_qus_ans.shape[0])

            # padding to same qus_ans length
            max_qus_ans_len = np.max(qus_ans_len)
            qus_ans_mask = []
            for j in range(batch):
                cur_len, hidden_size = qus_ans[j].shape[0], qus_ans[j].shape[1]
                zeros_len = max_qus_ans_len - cur_len
                new_zeros = qus_ans[j].new_zeros((zeros_len, hidden_size))
                qus_ans[j] = torch.cat([qus_ans[j], new_zeros], dim=0)

                cur_mask = torch.cat([torch.ones((cur_len,)), torch.zeros((zeros_len,))], dim=0)
                qus_ans_mask.append(cur_mask)
            qus_ans = torch.stack(qus_ans, dim=1)   # (qus_ans_len, batch, hidden_size)
            qus_ans_mask = torch.stack(qus_ans_mask, dim=0)   # (batch, qus_ans_len)

            # (qus_ans_len, batch, hidden_size)
            qus_ans_aware, alpha = self.seq_to_seq_att(qus_ans, context_encode, context_mask)
            m = self.linear_fusion(qus_ans, qus_ans_aware)

            # (qus_ans_len, batch, hidden_size)
            ans_measure, _ = self.measure_rnn(m, qus_ans_mask)
            ans_measure = qus_ans_mask.transpose(0, 1).unsqueeze(2) * ans_measure

            ans_s, _ = torch.max(ans_measure, dim=0)    # (batch, hidden_size)
            cand_s.append(ans_s)
        cand_s = torch.stack(cand_s, dim=0)  # (k, batch, hidden_size)
        cand_score = self.linear_log_softmax(cand_s)    # (batch, k)

        _, max_cand_k = torch.max(cand_score, dim=-1)   # (batch,)
        out_answer = []
        for i in range(batch):
            tmp_answer = candidate_answer[i, max_cand_k[i], :]
            out_answer.append(tmp_answer)

        out_answer = torch.stack(out_answer, dim=0)     # (batch, ans_len)

        return cand_score, out_answer