import sys
import codecs
import numpy as np

def read_label_index(filename):
    event2id={}
    id2event=[]
    with codecs.open(filename,'r','utf-8') as f:
        for line in f:
            ss=line.strip().split('\t')
            if len(ss) == 2:
                event2id[ss[0]]=len(event2id)
                id2event.append(ss[0])
    return event2id,id2event

event2id,id2event=read_label_index(sys.argv[1])

with codecs.open(sys.argv[2],'r','utf-8') as f:
    with codecs.open(sys.argv[3],'w','utf-8') as fout:
        fout.write('event\n')
        for line in f:
            ss=line.strip()
            if ss!="":
                ans_index=int(ss)
                ans=id2event[ans_index]
                fout.write(str(ans)+'\n')
