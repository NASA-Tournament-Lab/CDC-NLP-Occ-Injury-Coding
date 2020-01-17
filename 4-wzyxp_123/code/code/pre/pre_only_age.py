import sys
import codecs
import uuid

whether_remove_age=True

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


def remove_age_from_text(text,age):
    text=age+' '+text.replace(age,'').strip()
    return text

s_map={1:'male',2:'female'}

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
                sex=ss[1]
                age=ss[2]

                if whether_remove_age:
                    text = remove_age_from_text(text, age)
                    sex=s_map[int(sex)]

                other=sex+' '+age
                label=0
                if len(ss) == 4:
                    label=event2id[ss[3]]

                #output for tf bert format
                #fout.write('\t'.join([str(label),text,other,str(uuid.uuid1())])+'\n')
                fout.write('\t'.join([str(label),text,str(uuid.uuid1())])+'\n')




