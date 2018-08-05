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
        hidden_size = 100
        hidden_mode = 'LSTM'
        emb_dropout_p = 0.2
        dropout_p = 0.2

        self.num_align_hops = 2

        self.embedding = GloveEmbedding(dataset_h5_path=dataset_h5_path)
        self.encoder = MyRNNBase(mode=hidden_mode,
                                 input_size=word_embedding_size,
                                 hidden_size=hidden_size,
                                 bidirectional=True,
                                 dropout_p=emb_dropout_p)

        self.aligner = torch.nn.ModuleList([SeqToSeqAtten() for _ in range(self.num_align_hops)])
        self.aligner_sfu = torch.nn.ModuleList([SFU(input_size=hidden_size * 2,
                                                    fusions_size=hidden_size * 2 * 3) for _ in
                                                range(self.num_align_hops)])
        self.ans_aligner = torch.nn.ModuleList([SeqToSeqAtten() for _ in range(self.num_align_hops)])
        self.ans_aligner_sfu = torch.nn.ModuleList([SFU(input_size=hidden_size * 2,
                                                         fusions_size=hidden_size * 2 * 3)
                                                     for _ in range(self.num_align_hops)])
        self.aggregation = torch.nn.ModuleList([MyRNNBase(mode=hidden_mode,
                                                          input_size=hidden_size * 2,
                                                          hidden_size=hidden_size,
                                                          bidirectional=True,
                                                          dropout_p=dropout_p,
                                                          enable_layer_norm=False)
                                                for _ in range(self.num_align_hops)])

        self.linear_log_softmax = LinearLogSoftmax(hidden_size*2)

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

            ans_encode_list = []
            ans_len = []
            cand_ans = candidate_answer[:, i, :]    # (batch, 2)
            for j in range(batch):
                cur_ans = cand_ans[j, :]
                cur_ans_encode = context_encode[cur_ans[0]:(cur_ans[1]+1), j, :]
                ans_encode_list.append(cur_ans_encode)
                ans_len.append(cur_ans_encode.shape[0])

            # padding to answer encode
            max_ans_len = np.max(ans_len)
            hidden_size = context_encode.shape[-1]
            ans_encode = context_encode.new_zeros((max_ans_len, batch, hidden_size))
            ans_mask = context_encode.new_zeros((batch, max_ans_len))
            for j in range(batch):
                cur_len = ans_encode_list[j].shape[0]
                ans_encode[:cur_len, j, :] = ans_encode_list[j]
                ans_mask[j, :cur_len] = 1

            align_ans = ans_encode
            align_qt = question_encode
            for i in range(self.num_align_hops):
                # question-context align: (seq_len, batch, hidden_size*2)
                ct_align_qt, _ = self.aligner[i](question_encode, context_encode, context_mask)
                align_qt = self.aligner_sfu[i](align_qt, torch.cat([ct_align_qt,
                                                                    align_qt * ct_align_qt,
                                                                    align_qt - ct_align_qt], dim=-1))

                # answer-question align: (seq_len, batch, hidden_size*2)
                qt_align_ans, _ = self.ans_aligner[i](align_ans, align_qt, question_mask)
                hat_ans = self.ans_aligner_sfu[i](align_ans, torch.cat([qt_align_ans,
                                                                        align_ans * qt_align_ans,
                                                                        align_ans - qt_align_ans], dim=-1))

                # aggregation: (seq_len, batch, hidden_size*2)
                align_ans, align_ans_last = self.aggregation[i](hat_ans, ans_mask)

            cand_s.append(align_ans_last)
        cand_s = torch.stack(cand_s, dim=0)  # (k, batch, hidden_size*2)
        cand_score = self.linear_log_softmax(cand_s)    # (batch, k)

        _, max_cand_k = torch.max(cand_score, dim=-1)   # (batch,)
        out_answer = []
        for i in range(batch):
            tmp_answer = candidate_answer[i, max_cand_k[i], :]
            out_answer.append(tmp_answer)

        out_answer = torch.stack(out_answer, dim=0)     # (batch, ans_len)

        return cand_score, out_answer