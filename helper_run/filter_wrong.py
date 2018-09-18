# -*- coding: utf-8 -*-

import os
import json
import random


def filter_dump():
    with open('../data/answer_full/match-lstm+f.json-10', 'r') as json1_f:
        ans_json = json.load(json1_f)

    wrong_json = list(
        filter(lambda x: not x['em'], ans_json)
    )

    random.shuffle(wrong_json)

    print(len(ans_json), len(wrong_json))
    with open('../data/match-lstm_f.json-10-sample', 'w') as json2_f:
        json.dump(wrong_json[:50], json2_f, ensure_ascii=False)


def count_type():
    with open('../data/answer_full/match-lstm_f.json-10-sample', 'r') as f:
        type_json = json.load(f)

    type_cnt = [0 for i in range(10)]
    for ans in type_json:
        type_cnt[ans['type']] += 1

    print(type_cnt, sum(type_cnt))


if __name__ == '__main__':
    count_type()