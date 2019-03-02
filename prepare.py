import pandas as pd

all = pd.read_csv("data/train.csv", header=None,sep="\t")
maxLen=0
vacb={}
for i in range (len(all)):
    q1=all[1][i]
    q2=all[2][i]
    m=max(len(q1),len(q2))
    if(maxLen<m):
        maxLen=m
    for ch in (q1+q2):
        if(ch not in vacb):
            vacb[ch]=1
        else:
            vacb[ch]+=1
#     break
vacb_ch2index=dict(zip(vacb.keys(),range(0,len(vacb))))
vacb_index2ch=dict(zip(range(0,len(vacb)),vacb.keys()))

import pickle
pickle.dump(vacb_ch2index,open('data/vacb_ch2index.pkl','wb'))
pickle.dump(vacb_index2ch,open('data/vacb_index2ch.pkl','wb'))


all = all.sample(len(all))
test = all[0:10000]
train = all[10000:-1]

print(len(all))
