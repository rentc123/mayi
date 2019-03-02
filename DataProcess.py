from __future__ import absolute_import, division, print_function
import pickle
import argparse
import csv
import logging
import os
import random
import sys
import jieba
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import torch.optim as optim
import torch.nn as nn

from CLR import CLR
from SentenceMathV3 import SentenceMath

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class MyTokenizer(object):
    # max_seq_length:最大句子长度
    # max_word_length: 一个词的最大长度
    # max_seq_word_length:一个句子中最大词数
    def __init__(self, ch_vacb, max_seq_length=10, max_word_length=10, max_seq_word_length=50, word_vacb=None):

        # [PAD]:0 [UNK]:1
        self.ch_vacb = ch_vacb  # dict --> ch:index
        self.word_vacb = word_vacb  # dict --> word:index
        self.max_seq_length = max_seq_length
        self.max_word_length = max_word_length
        self.max_seq_word_length = max_seq_word_length

    def tokenize(self, sentence, ch=True):
        if (ch):
            return " ".join(sentence).split()
        else:
            return jieba.lcut(sentence)
    def convert_tokens_to_ch_ids(self, tokens):
        idx = [self.ch_vacb.get(w, 1) for w in tokens]
        padding = [0] * (self.max_seq_length - len(idx))
        return idx + padding

    def convert_tokens_to_word_ch_ids(self, tokens):
        idx = []
        for word in tokens:
            idxx = [self.ch_vacb.get(ch, 1) for ch in word]
            if (len(idxx) > self.max_word_length):
                idxx = idxx[0:self.max_word_length]
            padding = [0] * (self.max_word_length - len(idxx))
            idx.append(idxx + padding)

        padding = [[0] * self.max_word_length] * (self.max_seq_word_length - len(idx))

        return idx + padding
    def convert_tokens_to_word_ids(self, tokens):
        idx = [self.word_vacb.get(w, 1) for w in tokens]
        padding = [0] * (self.max_seq_word_length - len(idx))
        return idx + padding

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ch_ids_A, input_word_ch_id_A, input_mask_A,
                 input_ch_ids_B, input_word_ch_id_B, input_mask_B,
                 label_id):
        self.input_ch_ids_A = input_ch_ids_A
        self.input_word_ch_id_A = input_word_ch_id_A
        self.input_mask_A = input_mask_A

        self.input_ch_ids_B = input_ch_ids_B
        self.input_word_ch_id_B = input_word_ch_id_B
        self.input_mask_B = input_mask_B

        self.label_id = label_id

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_csv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        dicts = []
        with open(input_file, 'r', encoding='utf-8') as infs:
            for inf in infs:
                inf = inf.strip().split("\t")
                dicts.append(inf)
            return dicts

class MyProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train_samples.csv")))
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "train_samples.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "dev_samples.csv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            text_b = line[2]
            label = line[3]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

def _truncate_seq_pair(tokens_a, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a)
        if total_length <= max_length:
            break
        tokens_a.pop()

        # max_seq_length:最大句子长度
        # max_word_length: 一个词的最大长度
        # max_seq_word_length:一个句子中最大词数

def convert_examples_to_features(examples, label_list, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a_ch = tokenizer.tokenize(example.text_a)
        # tokens_a_word = ''
        tokens_a_word = tokenizer.tokenize(example.text_a, ch=False)
        _truncate_seq_pair(tokens_a_ch, tokenizer.max_seq_length)
        _truncate_seq_pair(tokens_a_word, tokenizer.max_seq_word_length)

        tokens_b_ch = tokenizer.tokenize(example.text_b)
        # tokens_b_word = ''
        tokens_b_word = tokenizer.tokenize(example.text_b, ch=False)
        _truncate_seq_pair(tokens_b_ch, tokenizer.max_seq_length)
        _truncate_seq_pair(tokens_b_word, tokenizer.max_seq_word_length)

        input_ch_ids_a = tokenizer.convert_tokens_to_ch_ids(tokens_a_ch)
        input_word_ch_ids_a = tokenizer.convert_tokens_to_word_ch_ids(tokens_a_word)

        input_ch_ids_b = tokenizer.convert_tokens_to_ch_ids(tokens_b_ch)
        input_word_ch_ids_b = tokenizer.convert_tokens_to_word_ch_ids(tokens_b_word)
        mask_a = np.ones_like(input_ch_ids_a)
        mask_a[np.array(input_ch_ids_a) == 0] = 0

        mask_b = np.ones_like(input_ch_ids_b)
        mask_b[np.array(input_ch_ids_b) == 0] = 0

        label_id = label_map[example.label]
        if ex_index < 3:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens_a: %s" % " ".join(
                [str(x) for x in tokens_a_ch]))
            logger.info("tokens_b: %s" % " ".join(
                [str(x) for x in tokens_b_ch]))

            logger.info("input_ch_ids_a: %s" % " ".join([str(x) for x in input_ch_ids_a]))
            logger.info("input_word_ch_ids_a: %s" % " ".join([str(x) for x in input_word_ch_ids_a]))
            logger.info("input_ch_ids_b: %s" % " ".join([str(x) for x in input_ch_ids_b]))
            logger.info("mask_a: %s" % " ".join([str(x) for x in mask_a]))
            logger.info("mask_b: %s" % " ".join([str(x) for x in mask_b]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ch_ids_A=input_ch_ids_a,
                          input_word_ch_id_A=input_word_ch_ids_a,
                          input_mask_A=mask_a,
                          input_ch_ids_B=input_ch_ids_b,
                          input_word_ch_id_B=input_word_ch_ids_b,
                          input_mask_B=mask_b,
                          label_id=label_id))
    return features

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def precise(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(((outputs == labels) & (labels == 1)))

def lable_1(labels):
    return np.sum((labels == 1))

def pre_1(out):
    outputs = np.argmax(out, axis=1)
    return np.sum((outputs == 1))

if __name__ == '__main__':
    do_train = True
    do_eval = True
    data_dir = "data"
    train_batch_size = 64
    eval_batch_size = 32
    num_train_epochs = 50
    lr = 1e-4
    processor = MyProcessor()
    num_train_optimization_steps = None
    no_cuda = False
    weight = torch.Tensor([1, 1])

    # ch_vacb = {'[PAD]': 0, '[OOV]': 1, 'A': 2, 'B': 3, '的': 4}
    vacb_path = "data/vacb_ch2index.pkl"
    ch_vacb = pickle.load(open(vacb_path, 'rb'))
    tokenizer = MyTokenizer(ch_vacb, 120, 5, 120)

    if do_train:
        train_examples = processor.get_train_examples(data_dir)
        label_list = processor.get_labels()
        train_features = convert_examples_to_features(train_examples, label_list, tokenizer)

        num_train_optimization_steps = int(len(train_examples) / train_batch_size) * num_train_epochs

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        all_input_ch_ids_A = torch.tensor([f.input_ch_ids_A for f in train_features], dtype=torch.long)
        all_input_word_ch_id_A = torch.tensor([f.input_word_ch_id_A for f in train_features], dtype=torch.long)
        all_input_mask_A = torch.tensor([f.input_mask_A for f in train_features], dtype=torch.long)

        all_input_ch_ids_B = torch.tensor([f.input_ch_ids_B for f in train_features], dtype=torch.long)
        all_input_word_ch_id_B = torch.tensor([f.input_word_ch_id_B for f in train_features], dtype=torch.long)
        all_input_mask_B = torch.tensor([f.input_mask_B for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ch_ids_A, all_input_word_ch_id_A, all_input_mask_A,
                                   all_input_ch_ids_B, all_input_word_ch_id_B, all_input_mask_B,
                                   all_label_ids)
        train_sampler = RandomSampler(train_data)
        # else:
        #     train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)
        device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()

        model = SentenceMath(200, ch_vacb, 100)
        model = model.cuda()
        clr = CLR(len(train_examples), 2, train_batch_size, 1e-5, 1e-4)
        # _______________________________________________________________________________________________________________________
        model.load_state_dict(torch.load(r'ckpt/4_state_dic_V3_lr_submul.ckpt'))
        # _______________________________________________________________________________________________________________________

        optimzer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        lossfuntion = nn.NLLLoss(
            weight=weight.to(device)
        )
        model.train()
        all_steps = 0
        for epoch in trange(int(num_train_epochs), desc="Epoch"):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            lable_1_count = 0
            pre_1_count = 0
            pre_equ_lable = 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ch_ids_A, input_word_ch_id_A, input_mask_A, \
                input_ch_ids_B, input_word_ch_id_B, input_mask_B, \
                label_ids = batch
                v = model(input_ch_ids_A, input_mask_A, input_ch_ids_B, input_mask_B, input_word_ch_id_A,
                          input_word_ch_id_B)

                loss = lossfuntion(v, label_ids)
                all_steps += 1
                # print(loss.grad())
                loss_num = loss.data.cpu().numpy().max()
                if (step % 10 == 0):
                    logger.info("\n")
                    logger.info("loss= %f", loss_num)
                # score = torch.exp(v)
                # out = model(input_ch1, input_pos1, input_ch2, input_pos2)
                # loss = lossfuntion(out, target)
                loss.backward()
                optimzer.step()
                optimzer.zero_grad()
                for param_group in optimzer.param_groups:
                    param_group['lr'] = clr.getlr(all_steps)

            torch.save(model.state_dict(), open(r'ckpt/' + str(epoch) + '_state_dic_V3_lr_submul.ckpt', 'wb'))

            if do_eval:
                eval_examples = processor.get_dev_examples(data_dir)
                eval_features = convert_examples_to_features(
                    eval_examples, label_list, tokenizer)
                logger.info("***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", eval_batch_size)

                # all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
                # all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
                # all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
                # all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
                # eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
                # # Run prediction for full data


                all_input_ch_ids_A = torch.tensor([f.input_ch_ids_A for f in eval_features], dtype=torch.long)
                all_input_word_ch_id_A = torch.tensor([f.input_word_ch_id_A for f in eval_features], dtype=torch.long)
                all_input_mask_A = torch.tensor([f.input_mask_A for f in eval_features], dtype=torch.long)

                all_input_ch_ids_B = torch.tensor([f.input_ch_ids_B for f in eval_features], dtype=torch.long)
                all_input_word_ch_id_B = torch.tensor([f.input_word_ch_id_B for f in eval_features], dtype=torch.long)
                all_input_mask_B = torch.tensor([f.input_mask_B for f in eval_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

                eval_data = TensorDataset(all_input_ch_ids_A, all_input_word_ch_id_A, all_input_mask_A,
                                          all_input_ch_ids_B, all_input_word_ch_id_B, all_input_mask_B,
                                          all_label_ids)
                eval_sampler = SequentialSampler(eval_data)

                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

                model.eval()
                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0

                for batch in tqdm(eval_dataloader, desc="Evaluating"):
                    input_ch_ids_A, input_word_ch_id_A, input_mask_A, \
                    input_ch_ids_B, input_word_ch_id_B, input_mask_B, \
                    label_ids = batch

                    input_ch_ids_A = input_ch_ids_A.to(device)
                    input_word_ch_id_A = input_word_ch_id_A.to(device)
                    input_mask_A = input_mask_A.to(device)

                    input_ch_ids_B = input_ch_ids_B.to(device)
                    input_word_ch_id_B = input_word_ch_id_B.to(device)
                    input_mask_B = input_mask_B.to(device)
                    label_ids = label_ids.to(device)

                    with torch.no_grad():
                        # tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                        logits = model(input_ch_ids_A, input_mask_A, input_ch_ids_B, input_mask_B, input_word_ch_id_A,
                                       input_word_ch_id_B)

                        # score = torch.exp(v)
                        # score = torch.exp(v).data.cpu().numpy()[0][1]

                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    tmp_eval_accuracy = accuracy(logits, label_ids)

                    eval_accuracy += tmp_eval_accuracy

                    nb_eval_examples += input_ch_ids_A.size(0)
                    nb_eval_steps += 1

                    lable_1_count += lable_1(label_ids)
                    pre_1_count += pre_1(logits)
                    pre_equ_lable += precise(logits, label_ids)

                eval_loss = eval_loss / nb_eval_steps
                eval_accuracy = eval_accuracy / nb_eval_examples
                precise_accuracy = pre_equ_lable / pre_1_count
                recall = pre_equ_lable / lable_1_count
                f1 = 2 * precise_accuracy * recall / (precise_accuracy + recall)
                result = {
                    # 'eval_loss': eval_loss,
                    'eval_accuracy': eval_accuracy,
                    'precise_accuracy': precise_accuracy,
                    'recall': recall,
                    "f1": f1
                    # 'global_step': global_step,
                    # 'loss': loss
                }
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    # output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                    # with open(output_eval_file, "w") as writer:
                    #     logger.info("***** Eval results *****")
                    #     for key in sorted(result.keys()):
                    #         logger.info("  %s = %s", key, str(result[key]))
                    #         writer.write("%s = %s\n" % (key, str(result[key])))




# ch_vacb = {'[PAD]': 0, '[UNK]': 1, 'A': 2, 'B': 3, '的': 4}
# tokenizer = MyTokenizer(ch_vacb, 20, 20, 20)
# token = tokenizer.tokenize("ABC噶的我们")
# print(jieba.lcut("ABC噶的我们"))
# token2 = tokenizer.tokenize("ABC噶的我们", ch=False)
# print(tokenizer.convert_tokens_to_ch_ids(token))
# print(tokenizer.convert_tokens_to_word_ch_ids(token2))
