import sys
import codecs
import os
import numpy as np

inputdir=sys.argv[1]

files=os.listdir(inputdir)
index2dir={}
for file in files:
    ss=file.split('-')
    if len(ss) == 2 and ss[0]=='checkpoint':
        index2dir[ss[1]]=os.path.join(inputdir,file)

sorted_dirs=sorted(index2dir.items(),key=lambda x:x[0],reverse=True)


N=int(sys.argv[2])

label_num=-1

def np_softmax(x):
    x=np.exp(x)
    x_sum=np.sum(x)
    return x/x_sum

def read_one_file(filename):
    global  label_num
    labels=[]
    with codecs.open(filename,'r','utf-8') as f:
        for line in f:
            ss=line.strip().split()
            if label_num<0:
                label_num=len(ss)
            elif label_num!=len(ss):
                print('diff label num {}!={}'.format(label_num,len(ss)))
            probs=np.array(list(map(float,ss)))
            probs=np_softmax(probs)
            labels.append(probs)


    return labels

number=-1
label_list=[]
for n in range(N):
    labels=read_one_file(os.path.join(sorted_dirs[n][1],'dev.probs.res'))
    if number>0 and len(labels) != number:
        print('ERROR different label number')
        exit(-1)
    else:
        number=len(labels)

    label_list.append(labels)

final_labels=[]
for i in range(len(label_list[0])):
    sum_probs=np.zeros([label_num])
    for j in range(len(label_list)):
        sum_probs+=label_list[j][i]
    sum_probs/=len(label_list)
    max_label=np.argmax(sum_probs)
    final_labels.append(max_label)

if len(final_labels)!=number:
    print('ERROR, final is not equal to the number')
with codecs.open(sys.argv[3],'w','utf-8') as fout:
    for label in final_labels:
        fout.write('{}\n'.format(label))

