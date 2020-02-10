from __future__ import print_function
from collections import Counter
import string
import os
import re
import argparse
import json
import sys
import pdb


#def normalize_answer(s):
#    """Lower text and remove punctuation, articles and extra whitespace."""
#    def remove_articles(text):
#        return re.sub(r'\b(a|an|the)\b', ' ', text)
#
#    def white_space_fix(text):
#        return ' '.join(text.split())
#
#    def remove_punc(text):
#        exclude = set(string.punctuation)
#        return ''.join(ch for ch in text if ch not in exclude)
#
#    def lower(text):
#        return text.lower()
#
#    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(predict_list, ground_truths):
    common = Counter(predict_list) & Counter(ground_truths)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(predict_list)
    recall = 1.0 * num_same / len(ground_truths)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1, precision, recall


# 模糊匹配(至少1个真实结果被预测)
def vague_match_score(predict_list, ground_truths):
    if any (g in predict_list for g in ground_truths):
        return 1
    else:
        return 0

# 精确匹配(所有真实结果均被预测)
def exact_match_score(predict_list, ground_truths):
    if all(g in predict_list for g in ground_truths):
        return 1
    else:
        return 0

# 匹配得分
def match_score(predict_list, ground_truths, exact):
    if exact == 'True':
        score = exact_match_score(predict_list, ground_truths)
    else:
        score = vague_match_score(predict_list, ground_truths)
    return score


def prediction_modify(predictions):
    # 去掉重复条目
    predictions_m1 = predictions.copy()
    for key in predictions.keys():
        if int(key.split('_')[2].replace('j','')) != 0:
            del predictions_m1[key]
    # 修正id格式
    predictions_m2 = {}
    for key in predictions_m1.keys():
        id_start = '_'.join([key.split('_')[0],key.split('_')[1]])
        predictions_m2[id_start] = predictions_m1[key]
#    pdb.set_trace()
    return predictions_m2


def evaluate(dataset, predictions, exact):
    f1 = exact_match = precision = recall = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            ground_truths = []
            total += 1
            for qa in paragraph['qas']:
#                if qa['id'] not in predictions:
#                    message = 'Unanswered question ' + qa['id'] + \
#                              ' will receive score 0.'
#                    print(message, file=sys.stderr)
#                    continue
#                
                id_start = '_'.join([qa['id'].split('_')[0],qa['id'].split('_')[1]])
                ground_truths = ground_truths + list(map(lambda x: x['text'], qa['focus'])) #a list
                
#            predict_list = predictions[id_start].split('@@') #a list
            predict_list = predictions[id_start].split('###')[0].split('@@') #a list
            for x in predict_list:
                if x =='':
                    predict_list.remove(x)
            
            exact_match += match_score(predict_list, ground_truths, exact=exact)
            f1 += f1_score(predict_list, ground_truths)[0]
            precision += f1_score(predict_list, ground_truths)[1]
            recall += f1_score(predict_list, ground_truths)[2]
#    pdb.set_trace()

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    precision = 100.0 * precision / total
    recall = 100.0 * recall / total
    
    if exact == 'True':
        return {'exact_match': exact_match, 'f1': f1, 'precision':precision, 'recall':recall}
    else:
        return {'vague_match': exact_match, 'f1': f1, 'precision':precision, 'recall':recall}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_file', help='Dataset file')
    parser.add_argument('prediction_file', help='Prediction File')
    parser.add_argument('exact', help='is exact match or not', default='True', type=str)
    parser.add_argument('modify', help='is prediction modify or not', default='True', type=str)
    
    args = parser.parse_args()
    dataset_file = args.dataset_file
    prediction_file = args.prediction_file
    exact = args.exact
    modify = args.modify

    with open(dataset_file) as f:
        dataset_json = json.load(f)
    dataset = dataset_json['data']
    
    with open(prediction_file) as p:
        predictions = json.load(p)
    predictions = prediction_modify(predictions)
    
    print(json.dumps(evaluate(dataset, predictions, exact)))
    
    if modify == 'True':
        predictions = json.dumps(predictions, ensure_ascii=False, default="utf-8")
        path = os.path.split(prediction_file)[0]
        name = os.path.split(prediction_file)[1].split('.')[0]
        fh = open(path + '/' + name + '_modify.json', 'w')
        fh.write(predictions)
        fh.close()
        print('Modified predict file is done!')

