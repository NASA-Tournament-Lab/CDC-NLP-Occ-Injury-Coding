import sys
import codecs
import uuid


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
        header=None
        for line in f:
            ss=line.strip().split(',')
            if header == None:
                header=ss
                fout.write('header\n')
                continue

            if len(ss) >=3:
                text=ss[0].lower()
                other=ss[1]+' '+ss[2]
                label=0
                if len(ss) == 4:
                    label=event2id[ss[3]]

                #output for tf bert format
                fout.write('\t'.join([str(label),text,other,str(uuid.uuid1())])+'\n')



