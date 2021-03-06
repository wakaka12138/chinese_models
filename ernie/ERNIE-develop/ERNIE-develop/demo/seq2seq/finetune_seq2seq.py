#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import sys
import argparse
import logging
import json
import re
import os
import numpy as np

from pathlib import Path
from copy import deepcopy

import paddle as P
from paddle.nn import functional as F

from tqdm import tqdm

from ernie.modeling_ernie import ErnieModel, ErnieModelForPretraining, ErnieModelForGeneration
from ernie.modeling_ernie import _build_linear, _build_ln, append_name
from ernie.tokenizing_ernie import ErnieTokenizer
#from ernie.optimization import AdamW, LinearDecay

from demo.seq2seq.decode import beam_search_infilling, post_process
from demo.utils import create_if_not_exists, get_warmup_and_linear_decay

from propeller import log
log.setLevel(logging.DEBUG)
import propeller.paddle as propeller


@np.vectorize
def rev_lookup(i):
    return rev_dict[i]


def evaluate(model, datasets, step, args):
    predict_output_dir = args.predict_output_dir / ('pred.step%d.%d' %
                                                    (step, env.dev_id))
    with predict_output_dir.open('w') as outf:
        with P.amp.auto_cast(enable=False):
            for step, data in enumerate(
                    P.io.DataLoader(
                        datasets,
                        places=P.CUDAPlace(env.dev_id),
                        batch_size=None)):
                (example_id, src_ids, src_sids, src_pids, _, _, _, _, _, _, _,
                 _) = data  # never use target when infer
                output_ids = beam_search_infilling(
                    model,
                    src_ids,
                    src_sids,
                    eos_id=tokenizer.sep_id,
                    sos_id=tokenizer.cls_id,
                    attn_id=tokenizer.vocab[args.attn_token],
                    max_decode_len=args.max_decode_len,
                    max_encode_len=args.max_encode_len,
                    beam_width=args.beam_width,
                    length_penalty=args.length_penalty,
                    tgt_type_id=args.tgt_type_id, )
                output_str = rev_lookup(output_ids.numpy())
                for eid, ostr in zip(example_id.numpy().tolist(),
                                     output_str.tolist()):
                    if '[SEP]' in ostr:
                        ostr = ostr[:ostr.index('[SEP]')]
                    ostr = ''.join(map(post_process, ostr))
                    print('%d\t%s' % (eid, ostr), file=outf)

    model.train()


def seq2seq(model, tokenizer, args):
    log.info('Training starts with args: %r' % args)
    attn_id = tokenizer.vocab[args.attn_token]

    def gen_mask(batch_ids, mask_type='bidi', query_len=None, pad_value=0):
        if query_len is None:
            query_len = batch_ids.shape[1]
        if mask_type != 'empty':
            mask = (batch_ids != pad_value).astype(np.float32)
            mask = np.tile(np.expand_dims(mask, 1), [1, query_len, 1])
            if mask_type == 'causal':
                assert query_len == batch_ids.shape[1]
                mask = np.tril(mask)
            elif mask_type == 'causal_without_diag':
                assert query_len == batch_ids.shape[1]
                mask = np.tril(mask, -1)
            elif mask_type == 'diag':
                assert query_len == batch_ids.shape[1]
                mask = np.stack([np.diag(np.diag(m)) for m in mask], 0)
        else:
            mask_type == 'empty'
            mask = np.zeros_like(batch_ids).astype(np.float32)
            mask = np.tile(np.expand_dims(mask, 1), [1, query_len, 1])
        return mask

    def make_some_noice(ids):
        if args.use_random_noice:
            noice_ids = np.random.randint(
                1, len(tokenizer.vocab), size=ids.shape)
        else:
            noice_ids = np.ones_like(ids) * tokenizer.vocab['[NOISE]']
        pos, = np.where(np.ones_like(ids))
        np.random.shuffle(pos)
        pos = pos[:int(args.noise_prob * len(pos))]
        ids[pos, ] = noice_ids[pos, ]
        return ids

    def map_fn(example_id, src_ids, tgt_ids):
        src_ids = src_ids[:args.max_encode_len]
        tgt_ids = tgt_ids[:args.max_decode_len]
        src_ids, src_sids = tokenizer.build_for_ernie(src_ids)
        src_pids = np.arange(len(src_ids))

        tgt_ids, tgt_sids = tokenizer.build_for_ernie(tgt_ids)
        tgt_pids = np.arange(len(tgt_ids)) + len(src_ids)  # continues position
        tgt_sids = np.ones_like(tgt_sids) * args.tgt_type_id

        attn_ids = np.ones_like(tgt_ids) * attn_id
        if args.noise_prob > 0.:
            tgt_labels = deepcopy(tgt_ids)
            tgt_ids = make_some_noice(tgt_ids)  #corrupted
        else:
            tgt_labels = tgt_ids

        return (example_id, src_ids, src_pids, src_sids, tgt_ids, tgt_pids,
                tgt_sids, attn_ids, tgt_labels)

    def after_padding(example_id, src_ids, src_pids, src_sids, tgt_ids,
                      tgt_pids, tgt_sids, attn_ids, tgt_labels):
        '''
        attention mask:
        ***  src,  tgt, attn
        src  00,   01,   11
        tgt  10,   11,   12
        attn 20,   21,   22

        ***   s1, s2 | t1 t2 t3| attn1 attn2 attn3
        s1    1,  1  | 0, 0, 0,| 0,    0,    0,
        s2    1,  1  | 0, 0, 0,| 0,    0,    0,
        -
        t1    1,  1, | 1, 0, 0,| 0,    0,    0,
        t2    1,  1, | 1, 1, 0,| 0,    0,    0,
        t3    1,  1, | 1, 1, 1,| 0,    0,    0,
        -
        attn1 1,  1, | 0, 0, 0,| 1,    0,    0,
        attn2 1,  1, | 1, 0, 0,| 0,    1,    0,
        attn3 1,  1, | 1, 1, 0,| 0,    0,    1,

        for details, see Fig3. https://arxiv.org/abs/2001.11314
        '''

        src_len = src_ids.shape[1]
        tgt_len = tgt_ids.shape[1]
        mask_00 = gen_mask(src_ids, 'bidi', query_len=src_len)
        mask_01 = gen_mask(tgt_ids, 'empty', query_len=src_len)
        mask_02 = gen_mask(attn_ids, 'empty', query_len=src_len)

        mask_10 = gen_mask(src_ids, 'bidi', query_len=tgt_len)
        mask_11 = gen_mask(tgt_ids, 'causal', query_len=tgt_len)
        mask_12 = gen_mask(attn_ids, 'empty', query_len=tgt_len)

        mask_20 = gen_mask(src_ids, 'bidi', query_len=tgt_len)
        mask_21 = gen_mask(tgt_ids, 'causal_without_diag', query_len=tgt_len)
        mask_22 = gen_mask(attn_ids, 'diag', query_len=tgt_len)
        '''
        mask = np.concatenate([
            np.concatenate([mask_00, mask_01, mask_02], 2),
            np.concatenate([mask_10, mask_11, mask_12], 2),
            np.concatenate([mask_20, mask_21, mask_22], 2),
        ], 1)

        ids = np.concatenate([src_ids, tgt_ids, attn_ids], 1)
        pids = np.concatenate([src_pids, tgt_pids, tgt_pids], 1)
        sids = np.concatenate([src_sids, tgt_sids, tgt_sids], 1)

        '''

        mask_src_2_src = mask_00
        mask_tgt_2_srctgt = np.concatenate([mask_10, mask_11], 2)
        mask_attn_2_srctgtattn = np.concatenate([mask_20, mask_21, mask_22], 2)

        tgt_labels = tgt_labels[np.where(tgt_labels != 0)]
        return (example_id, src_ids, src_sids, src_pids, tgt_ids, tgt_sids,
                tgt_pids, attn_ids, mask_src_2_src, mask_tgt_2_srctgt,
                mask_attn_2_srctgtattn, tgt_labels)

    bytes_vocab = {k.encode('utf8'): v for k, v in tokenizer.vocab.items()}
    feature_column = propeller.data.FeatureColumns([
        propeller.data.LabelColumn('id'),
        propeller.data.TextColumn(
            'src', unk_id=tokenizer.unk_id, vocab_dict=bytes_vocab),
        propeller.data.TextColumn(
            'tgt', unk_id=tokenizer.unk_id, vocab_dict=bytes_vocab),
    ])

    train_ds = feature_column.build_dataset('train', data_dir=os.path.join(args.data_dir, 'train'), shuffle=True, repeat=True, use_gz=False) \
                                   .map(map_fn) \
                                   .padded_batch(args.bsz) \
                                   .map(after_padding)


    dev_ds = feature_column.build_dataset('dev', data_dir=os.path.join(args.data_dir, 'dev'), shuffle=False, repeat=False, use_gz=False) \
                                   .map(map_fn) \
                                   .padded_batch(args.eval_bsz) \
                                   .map(after_padding) \
                                   .shard(env.nranks, env.dev_id)

    vocab_size, _ = model.word_emb.weight.shape
    model = P.DataParallel(model)
    g_clip = P.nn.ClipGradByGlobalNorm(1.0)
    param_name_to_exclue_from_weight_decay = re.compile(
        r'.*layer_norm_scale|.*layer_norm_bias|.*b_0')
    lr_scheduler = P.optimizer.lr.LambdaDecay(
        args.lr,
        get_warmup_and_linear_decay(
            args.max_steps, int(args.warmup_proportion * args.max_steps)))

    opt = P.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.wd,
        apply_decay_param_fun=lambda n: param_name_to_exclue_from_weight_decay.match(n),
        grad_clip=g_clip)

    scaler = P.amp.GradScaler(enable=args.use_amp)
    attn_id = tokenizer.vocab[args.attn_token]
    create_if_not_exists(args.save_dir)
    if args.predict_output_dir:
        create_if_not_exists(args.predict_output_dir)

    with P.amp.auto_cast(enable=args.use_amp):
        for step, data in enumerate(
                P.io.DataLoader(
                    train_ds, places=P.CUDAPlace(env.dev_id),
                    batch_size=None)):
            (example_id, src_ids, src_sids, src_pids, tgt_ids, tgt_sids,
             tgt_pids, attn_ids, mask_src_2_src, mask_tgt_2_srctgt,
             mask_attn_2_srctgtattn, tgt_labels) = data

            _, __, info = model(
                src_ids,
                sent_ids=src_sids,
                pos_ids=src_pids,
                attn_bias=mask_src_2_src,
                encode_only=True)
            cached_k, cached_v = info['caches']
            _, __, info = model(
                tgt_ids,
                sent_ids=tgt_sids,
                pos_ids=tgt_pids,
                attn_bias=mask_tgt_2_srctgt,
                past_cache=(cached_k, cached_v),
                encode_only=True)
            cached_k2, cached_v2 = info['caches']
            past_cache_k = [
                P.concat([k, k2], 1) for k, k2 in zip(cached_k, cached_k2)
            ]
            past_cache_v = [
                P.concat([v, v2], 1) for v, v2 in zip(cached_v, cached_v2)
            ]
            tgt_labels = F.one_hot(tgt_labels, vocab_size)
            if args.label_smooth > 0.:
                tgt_labels = F.label_smooth(
                    tgt_labels, epsilon=args.label_smooth)
            loss, _, __ = model(
                attn_ids,
                sent_ids=tgt_sids,
                pos_ids=tgt_pids,
                attn_bias=mask_attn_2_srctgtattn,
                past_cache=(past_cache_k, past_cache_v),
                tgt_labels=tgt_labels,
                tgt_pos=P.nonzero(attn_ids == attn_id))

            loss = scaler.scale(loss)
            loss.backward()
            scaler.minimize(opt, loss)
            model.clear_gradients()
            lr_scheduler.step()

            if step % 10 == 0:
                _lr = lr_scheduler.get_lr()
                if args.use_amp:
                    _l = (loss / scaler._scale).numpy()
                    msg = '[rank-%d][step-%d] train loss %.5f lr %.3e scaling %.3e' % (
                        env.dev_id, step, _l, _lr, scaler._scale.numpy())
                else:
                    _l = loss.numpy()
                    msg = '[rank-%d][step-%d] train loss %.5f lr %.3e' % (
                        env.dev_id, step, _l, _lr)
                log.debug(msg)

            if args.save_dir is not None and step % 1000 == 0 and env.dev_id == 0:
                P.save(model.state_dict(), str(args.save_dir / 'ckpt.bin'))

            if args.predict_output_dir is not None and step > args.skip_eval_steps and step % args.eval_steps == 0:
                assert  args.predict_output_dir.exists(), \
                 'predict_output_dir not found: %s' % args.predict_output_dir
                log.debug('doing predict on gpu %d...' % env.dev_id)
                evaluate(model, dev_ds, step, args)
            if step > args.max_steps:
                break
        evaluate(model, dev_ds, step, args)

    if args.save_dir is not None:
        P.save(model.state_dict(),str( args.save_dir / 'ckpt.bin'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('seq2seq model with ERNIE')
    parser.add_argument(
        '--from_pretrained',
        type=Path,
        required=True,
        help='pretrained model directory or tag')
    parser.add_argument('--bsz', type=int, default=8, help='batchsize')
    parser.add_argument('--eval_bsz', type=int, default=20, help='batchsize')
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='data directory includes train / develop data')
    parser.add_argument(
        '--max_steps',
        type=int,
        required=True,
        help='max_train_steps, set this to EPOCH * NUM_SAMPLES / BATCH_SIZE')
    parser.add_argument(
        '--eval_steps', type=int, default=5000, help='evaluation frequency')
    parser.add_argument(
        '--skip_eval_steps',
        type=int,
        default=1,
        help='skip evaluate for first n step')
    parser.add_argument('--max_encode_len', type=int, default=640)
    parser.add_argument('--max_decode_len', type=int, default=120)
    parser.add_argument('--tgt_type_id', type=int, default=3)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--beam_width', type=int, default=5)
    parser.add_argument(
        '--noise_prob',
        type=float,
        default=0.7,
        help='probability of token be repalced')
    parser.add_argument(
        '--use_random_noice',
        action='store_true',
        help='if set, replace target tokens with random token from vocabulary, else replace with `[NOISE]`'
    )
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--label_smooth', type=float, default=0.1)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument(
        '--predict_output_dir',
        type=Path,
        default=None,
        help='predict file output directory')
    parser.add_argument(
        '--attn_token',
        type=str,
        default='[ATTN]',
        help='if [ATTN] not in vocab, you can specified [MAKK] as attn-token')
    parser.add_argument(
        '--inference_model_dir',
        type=str,
        default=None,
        help='inference model output directory')
    parser.add_argument(
        '--init_checkpoint',
        type=str,
        default=None,
        help='checkpoint to warm start from')
    parser.add_argument(
        '--save_dir', type=Path, default=None, help='model output directory')
    parser.add_argument(
        '--wd',
        type=float,
        default=0.01,
        help='weight decay, aka L2 regularizer')
    parser.add_argument(
        '--use_amp',
        action='store_true',
        help='only activate AMP(auto mixed precision accelatoin) on TensorCore compatible devices'
    )

    args = parser.parse_args()

    env = P.distributed.ParallelEnv()
    P.distributed.init_parallel_env()

    ernie = ErnieModelForGeneration.from_pretrained(args.from_pretrained)
    tokenizer = ErnieTokenizer.from_pretrained(
        args.from_pretrained, mask_token=None)
    rev_dict = {v: k for k, v in tokenizer.vocab.items()}
    rev_dict[tokenizer.pad_id] = ''  # replace [PAD]
    rev_dict[tokenizer.unk_id] = ''  # replace [PAD]

    if args.init_checkpoint is not None:
        log.info('loading checkpoint from %s' % args.init_checkpoint)
        sd = P.load(str(args.init_checkpoint))
        ernie.set_state_dict(sd)

    seq2seq(ernie, tokenizer, args)
