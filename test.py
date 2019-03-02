import torch
import pandas as pd
import pickle
from SentenceMath import SentenceMath
import torch.optim as optim
import torch.nn as nn
import numpy as np

def tranform_input(seq, cuda=True):
    #     cuda=True
    # 每个batch 的seq1们长度必须一样 seq2们长度必须一样
    all_vacb_index = []
    all_index_array = []
    for sen in seq:
        #         print(sen)
        index = 0
        vacb_index = []
        index_array = []
        sen = " ".join(sen).split()
        for ch in sen:
            if (ch in vcab_ch):
                vacb_index.append(vcab_ch[ch])
                index_array.append(index)
            else:
                print(ch, ' not in vacb')
                index_array.append(len(vacb_index) + 1)
            index += 1
        all_vacb_index.append(vacb_index)
        all_index_array.append(index_array)

    if (cuda):
        input_ch = torch.LongTensor(all_vacb_index).cuda()
        input_pos = torch.LongTensor(all_index_array).cuda()
    else:
        input_ch = torch.LongTensor(all_vacb_index)
        input_pos = torch.LongTensor(all_index_array)
    return input_ch, input_pos

def transform_target(lable, cuda=True):
    # if (cuda):
    #     return torch.FloatTensor(lable).cuda()
    # else:
    #     return torch.FloatTensor(lable)


    l = []
    for la in lable:
        if (la == 0):
            l.append(0)
        else:
            l.append(1)
    lable = l

    if (cuda):
        return torch.LongTensor(lable).cuda()
    else:
        return torch.LongTensor(lable)

vcab_ch = pickle.load(open('data/vacb_ch2index.pkl', 'rb'))
print(len(vcab_ch))

# train = pickle.load(open('data/train.pkl', 'rb'))
train = pickle.load(open('data/test.pkl', 'rb'))

pre_array = []
lable_array = []

batch_size = 1
setp = 0
epoch = 0
data_dic = {}
a1 = list(train[1])
a2 = list(train[2])
a3 = list(train[3])

for i in range(len(train)):
    seq1 = a1[i]
    seq2 = a2[i]
    lable = a3[i]
    k = str(len(" ".join(seq1).split())) + ',' + str(len(" ".join(seq2).split()))

    if k not in data_dic:
        data_dic[k] = [[seq1], [seq2], [lable]]
    else:
        data_dic[k][0].append(seq1)
        data_dic[k][1].append(seq2)
        data_dic[k][2].append(lable)

model = SentenceMath(200, vcab_ch, 100, drop_out_flag=False)
model = model.cuda()
# model.load_state_dict(torch.load(r'ckpt/46_state_dic_v2.ckpt'))
model.load_state_dict(torch.load(r'ckpt/0_state_dic_V2.ckpt'))
for k in data_dic:
    # print(k)
    len_data = len(data_dic[k][0])
    for m in range(len_data):
        input_ch1, input_pos1 = tranform_input([data_dic[k][0][m]])
        input_ch2, input_pos2 = tranform_input([data_dic[k][1][m]])
        target = transform_target([data_dic[k][2][m]])

        v = model(input_ch1,  input_ch2)

        score = torch.exp(v)
        score=torch.exp(v).data.cpu().numpy()[0][1]
        # loss = lossfuntion(v1, v2, target)
        print(score, target.data.cpu().numpy().max())
        pre_value = 0
        if (score > 0.35):
            pre_value = 1
        lable_value = target.data.cpu().numpy().max()
        #         print(torch.exp(F.log_softmax(out, dim=1)).data.cpu(), pre_value, lable_value)
        #         # break
        pre_array.append(pre_value)
        lable_array.append(lable_value)
#
a = np.array(pre_array)
b = np.array(lable_array)
p = np.sum((a == 1) & (b == 1)) / np.sum(a == 1)
r = np.sum((a == 1) & (b == 1)) / np.sum(b == 1)
f1 = 2 * p * r / (p + r)
print(p, r, f1)
