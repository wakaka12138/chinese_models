from cgitb import text
import csv
import numpy as np
def read(data_path):
    label=[]
    text=[]
    with open(data_path, 'r', encoding='utf-8') as f:
        for row in f:
            qid,label_t,text_t = row.strip('\n').split('\t')
            label.append(label_t)
            text.append(text_t)
    return label,text

label_1,text_1=read('G:/postgraduate/First_grade/ernie/ChnSentiCorp/ChnSentiCorp/dev.tsv')
print(label_1[0])
path='G:/postgraduate/First_grade/ernie/ChnSentiCorp/ChnSentiCorp/dev_new.tsv'
with open(path, 'wt',encoding='utf-8') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t',lineterminator='\n')
    line=[]
    line.append(label_1[0])
    line.append(text_1[0])
    tsv_writer.writerow(line)
    i=1
    while i<len(label_1):
        line[0]=label_1[i]
        line[1]=text_1[i]
        tsv_writer.writerow(line)
        i=i+1