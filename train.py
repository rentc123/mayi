import torch
import pandas as pd
import pickle

from CLR import CLR
from SentenceMath import SentenceMath
import torch.optim as optim
import torch.nn as nn

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

train = pickle.load(open('data/train.pkl', 'rb'))
test = pickle.load(open('data/test.pkl', 'rb'))

batch_size = 64
setp = 0
epoch = 0

model = SentenceMath(200, vcab_ch, 100)
model = model.cuda()
# model.load_state_dict(torch.load(r'ckpt/13_state_dic_v2.ckpt'))

optimzer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
lossfuntion = nn.NLLLoss()
# lossfuntion = nn.CrossEntropyLoss()
data_dic = {}

print('prepare data')
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
print('finish prepare data')

clr = CLR(100000, 2, batch_size, 0.001, 0.05)

for epoch in range(500):
    for k in data_dic:
        # print(k)
        len_data = len(data_dic[k][0])
        if (len_data <= batch_size):
            input_ch1, input_pos1 = tranform_input(data_dic[k][0])
            input_ch2, input_pos2 = tranform_input(data_dic[k][1])
            target = transform_target(data_dic[k][2])
            optimzer.zero_grad()
            v = model(input_ch1, input_ch2)
            # v1 = v[0]
            # v2 = v[1]
            loss = lossfuntion(v, target)
            score = torch.exp(v)
            # out = model(input_ch1, input_pos1, input_ch2, input_pos2)
            # loss = lossfuntion(out, target)
            loss.backward()
            optimzer.step()
            loss_num = loss.data.cpu().numpy().max()

            setp += 1
            print('epoch:', epoch, 'setp:', setp, 'loss:', loss_num)
        else:
            for m in range(0, len_data, batch_size):
                input_ch1, input_pos1 = tranform_input(data_dic[k][0][m:m + batch_size])
                input_ch2, input_pos2 = tranform_input(data_dic[k][1][m:m + batch_size])
                target = transform_target(data_dic[k][2][m:m + batch_size])
                optimzer.zero_grad()
                v = model(input_ch1, input_ch2)
                # v1 = v[0]
                # v2 = v[1]
                loss = lossfuntion(v, target)

                # out = model(input_ch1, input_pos1, input_ch2, input_pos2)
                # loss = lossfuntion(out, target)
                loss.backward()
                optimzer.step()
                loss_num = loss.data.cpu().numpy().max()

                setp += 1
                print('epoch:', epoch, 'setp:', setp, 'loss:', loss_num, clr.getlr(setp))
                # if (setp % 10000 == 0):
                # torch.save(model.state_dict(), open(r'ckpt/' + str(epoch) + "_" + setp + '_state_dic.ckpt', 'wb'))
        for param_group in optimzer.param_groups:
            param_group['lr'] = clr.getlr(setp)
    torch.save(model.state_dict(), open(r'ckpt/' + str(epoch) + '_state_dic_self.ckpt', 'wb'))
    # break
