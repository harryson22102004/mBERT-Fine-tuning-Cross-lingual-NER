import torch, torch.nn as nn
from transformers import BertTokenizerFast, BertForTokenClassification
 
LABELS={"O":0,"B-PER":1,"I-PER":2,"B-ORG":3,"I-ORG":4,"B-LOC":5,"I-LOC":6}
 
class NERModel(nn.Module):
    def __init__(self, n_labels=len(LABELS)):
        super().__init__()
        self.bert=BertForTokenClassification.from_pretrained(
            'bert-base-multilingual-cased', num_labels=n_labels)
    def forward(self, ids, mask, labels=None):
        return self.bert(ids, attention_mask=mask, labels=labels)
 
def tokenize_and_align(tokenizer, words, tags, max_len=128):
    enc=tokenizer(words, is_split_into_words=True, truncation=True,
                  padding='max_length', max_length=max_len)
    label_ids=[]
    for wid in enc.word_ids():
        if wid is None: label_ids.append(-100)
        else: label_ids.append(LABELS.get(tags[wid] if wid<len(tags) else "O",0))
    return enc, label_ids
 
tokenizer=BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
words=["Barack","Obama","was","born","in","Hawaii"]; tags=["B-PER","I-PER","O","O","O","B-LOC"]
enc,lab=tokenize_and_align(tokenizer,words,tags)
model=NERModel()
ids=torch.tensor([enc['input_ids']]); mask=torch.tensor([enc['attention_mask']])
labels=torch.tensor([lab])
with torch.no_grad(): out=model(ids,mask,labels)
print(f"NER loss: {out.loss.item():.3f}"); pred=out.logits.argmax(-1)[0]
id2l={v:k for k,v in LABELS.items()}
for w,t,p in zip(words,tags,pred[1:len(words)+1]):
    print(f"  {w:10s}  true:{t:6s}  pred:{id2l.get(p.item(),'O')}")
