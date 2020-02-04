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
    text=text.replace(age,'').strip()
    return text

prefix_map={}
def collect_YOFM(text,sex):
    global prefix_map
    ss=text.strip().split()
    buffer=""
    for i in range(len(ss)):
        buffer+=ss[i]
        if len(buffer) >= 3:
            tmp=' '.join(ss[:i+1])
            if not tmp in prefix_map:
                prefix_map[tmp]=0
            prefix_map[tmp]+=1
            break

def modify_YOFM(text,sex):
    flag=True
    #yom
    if 'yom ' in text and sex == 'male':
        text=text.replace('yom ','y o male ')
    elif 'yo m ' in text and sex == 'male':
        text = text.replace('yo m ', 'y o male ')
    elif 'yo male ' in text and sex == 'male':
        text = text.replace('yo male ', 'y o male ')
    elif 'y o m ' in text and sex == 'male':
        text = text.replace('y o m ', 'y o male')
    elif 'y o male ' in text:
        pass
    elif 'y om ' in text and sex == 'male':
        text = text.replace('y om ', 'y o male ')
    #yof
    elif 'yof ' in text and sex == 'female':
        text = text.replace('yof ', 'y o female ')
    elif 'yo f ' in text and sex == 'female':
        text = text.replace('yo f ', 'y o female ')
    elif 'yo female ' in text and sex == 'female':
        text = text.replace('yo female ', 'y o female ')
    elif 'y o f ' in text and sex == 'female':
        text = text.replace('y o f ', 'y o female ')
    elif 'y o female ' in text:
        pass
    elif 'y of ' in text and sex == 'female':
        text = text.replace('y of ', 'y o female ')
    #ym
    elif 'ym ' in text and sex == 'male':
        text = text.replace('ym ', 'y male ')
    elif 'y m ' in text and sex == 'male':
        text = text.replace('y m ', 'y male ')
    elif 'y male ' in text and sex == 'male':
        pass
    #yf
    elif 'yf ' in text and sex == 'female':
        text = text.replace('yf ', 'y female ')
    elif 'y f ' in text and sex == 'female':
        text = text.replace('y f ', 'y female ')
    elif 'y female ' in text:
        pass
    elif 'yo ' in text:
        if sex == 'male':
            text = text.replace('yo ', 'y o male')
        else:
            text = text.replace('yo ', 'y o female')
    elif 'y o ' in text:
        if sex == 'male':
            text = text.replace('y o ', 'y o male')
        else:
            text = text.replace('y o ', 'y o female')
    else:
        flag=False

    return text



s_map={1:'male',2:'female'}

all_count=0
remaind_count=0

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
                    text=modify_YOFM(text,sex)


                other=sex+' '+age
                label=0
                if len(ss) == 4:
                    label=event2id[ss[3]]

                #output for tf bert format
                fout.write('\t'.join([str(label),text,other,str(uuid.uuid1())])+'\n')


prefixs=sorted(prefix_map.items(),key=lambda x:x[1],reverse=True)
for pre in prefixs:
    print('{}\t{}'.format(pre[0],pre[1]))

print('all={}, reminder={}'.format(all_count,remaind_count))
