#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import torch
import torch.nn.functional as F


class MyNLLLoss(torch.nn.modules.loss._Loss):
    """
    a standard negative log likelihood loss. It is useful to train a classification
    problem with `C` classes.

    Shape:
        - y_pred: (batch, answer_len, prob)
        - y_true: (batch, answer_len)
        - output: loss
    """
    def __init__(self):
        super(MyNLLLoss, self).__init__()

    def forward(self, y_pred, y_true):

        y_pred_log = torch.log(y_pred)
        start_loss = F.nll_loss(y_pred_log[:, 0, :], y_true[:, 0])
        end_loss = F.nll_loss(y_pred_log[:, 1, :], y_true[:, 1])
        return start_loss + end_loss
