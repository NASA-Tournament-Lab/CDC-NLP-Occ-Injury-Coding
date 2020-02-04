import sys
import codecs
import os

inputdir=sys.argv[1]

files=os.listdir(inputdir)
index2dir={}
for file in files:
    ss=file.split('-')
    if len(ss) == 2 and ss[0]=='checkpoint':
        index2dir[ss[1]]=os.path.join(inputdir,file)

sorted_dirs=sorted(index2dir.items(),key=lambda x:x[0],reverse=True)


N=int(sys.argv[2])

def read_one_file(filename):
    labels=[]
    with codecs.open(filename,'r','utf-8') as f:
        for line in f:
            ss=line.strip()
            labels.append(int(ss))

    return labels

number=-1
label_list=[]
for n in range(N):
    labels=read_one_file(os.path.join(sorted_dirs[n][1],'dev.res'))
    if number>0 and len(labels) != number:
        print('ERROR different label number')
        exit(-1)
    else:
        number=len(labels)

    label_list.append(labels)

final_labels=[]
for i in range(len(label_list[0])):
    label2count={}
    for j in range(len(label_list)):
        if not label_list[j][i] in label2count:
            label2count[label_list[j][i]]=0
        label2count[label_list[j][i]]+=1
    sorted_labels=sorted(label2count.items(),key=lambda x:x[1],reverse=True)
    final_labels.append(sorted_labels[0][0])

if len(final_labels)!=number:
    print('ERROR, final is not equal to the number')
with codecs.open(sys.argv[3],'w','utf-8') as fout:
    for label in final_labels:
        fout.write('{}\n'.format(label))

