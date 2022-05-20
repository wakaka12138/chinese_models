import paddle
import numpy as np
from functools import partial
import csv
import paddle.nn as nn
import paddle.nn.functional as F
import paddlenlp as ppnlp
from paddlenlp.data import JiebaTokenizer, Pad, Stack, Tuple, Vocab

from utils import convert_example

from paddlenlp.datasets import load_dataset
def read(data_path):
    flag=1
    with open(data_path, 'r', encoding='utf-8') as f:
        for row in f:
            if flag==1:
                flag=2
                continue
            line=row.strip('\n').split('\t')
            if len(line)!=2:
                print (len(line))
            label,text = row.strip('\n').split('\t')
            yield {'text':text,'label':label}

train_ds = load_dataset(read, data_path='/data/home/scv6134/run/ernie/train.tsv',lazy=False)
dev_ds = load_dataset(read, data_path='/data/home/scv6134/run/ernie/dev.tsv',lazy=False)
test_ds = load_dataset(read, data_path='/data/home/scv6134/run/ernie/test.tsv',lazy=False)
vocab = Vocab.load_vocabulary('/data/home/scv6134/run/ernie/senta_word_dict.txt', unk_token='[UNK]', pad_token='[PAD]')
tokenizer = JiebaTokenizer(vocab)

def create_dataloader(dataset,
                      trans_function=None,
                      mode='train',
                      batch_size=1,
                      pad_token_id=0,
                      batchify_fn=None):
    if trans_function:
        dataset_map = dataset.map(trans_function)

    # return_list 数据是否以list形式返回
    # collate_fn  指定如何将样本列表组合为mini-batch数据。传给它参数需要是一个callable对象，需要实现对组建的batch的处理逻辑，并返回每个batch的数据。在这里传入的是`prepare_input`函数，对产生的数据进行pad操作，并返回实际长度等。
    dataloader = paddle.io.DataLoader(
        dataset_map,
        return_list=True,
        batch_size=batch_size,
        collate_fn=batchify_fn)
        
    return dataloader

# python中的偏函数partial，把一个函数的某些参数固定住（也就是设置默认值），返回一个新的函数，调用这个新函数会更简单。
trans_function = partial(
    convert_example,
    tokenizer=tokenizer,
    is_test=False)

# 将读入的数据batch化处理，便于模型batch化运算。
# batch中的每个句子将会padding到这个batch中的文本最大长度batch_max_seq_len。
# 当文本长度大于batch_max_seq时，将会截断到batch_max_seq_len；当文本长度小于batch_max_seq时，将会padding补齐到batch_max_seq_len.
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=vocab['[PAD]']),  # input_ids
    Stack(dtype="int64"),  # seq len
    Stack(dtype="int64")  # label
): [data for data in fn(samples)]


train_loader = create_dataloader(
    train_ds,
    trans_function=trans_function,
    batch_size=128,
    mode='train',
    batchify_fn=batchify_fn)
dev_loader = create_dataloader(
    dev_ds,
    trans_function=trans_function,
    batch_size=128,
    mode='validation',
    batchify_fn=batchify_fn)
test_loader = create_dataloader(
    test_ds,
    trans_function=trans_function,
    batch_size=128,
    mode='test',
    batchify_fn=batchify_fn)

for i in train_loader:
    print(i)
    break