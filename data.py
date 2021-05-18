from torch.utils.data import DataLoader, Dataset
import pathlib
import spacy
import torch
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import numpy as np

import json
import numpy
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import math



def ann2dict(file):
    e = {}
    for line in open(file):
        if line.strip() != '' and line[0] =="T":
            label = line.split("\t")[1].strip().split(" ")[0]
            remainder = " ".join(line.split("\t")[1].strip().split(" ")[1:])
            for span in remainder.split(";"):
                start, end = span.split(" ")
                for character in range(int(start), int(end)):
                    if character == int(start):
                        e[character] = "B-"+label
                    else:
                        e[character] = "I-"+label
    return e


class NERDataset(Dataset):
    def __init__(self, tokenizer):
        self.instances = []
        self.scispacy_model = spacy.load('en_core_sci_md')
        self.NER_ID = 0
        self.tokenizer = tokenizer
        self.predictions = {}
    def file2conll(self, raw, ann,):
        adict = ann2dict(ann)
        r = open(raw).read()
        self.predictions[ann] = []
        for line in open(ann):
            if line[0]=="T":
                scores = line.split("\t")[1].split(" ")
                text = line.split("\t")[2].strip()
                self.predictions[ann].append((scores[0], scores[1], scores[2], text))
        ssm = self.scispacy_model(r)
        last_start =0
        last_end = 0
        for each_sentence in ssm.sents:            
            # Currently model one sentence at aa time
            # todo : larger context (there's a paper about this)
            window_start = each_sentence.start_char 
            window_end = each_sentence.end_char
            mini = {x-window_start:adict[x] for x in adict if x >= window_start and x <=window_end}
            textwindow = r[window_start:window_end]
            self.instances.append({"sentence":str(textwindow),"filename":ann, "offset":window_start, "raw_ner":mini, "id":str(self.NER_ID)})
            self.NER_ID +=1

                
    def __getitem__(self, index):
        return self.instances[index]
    def __len__(self):
        return len(self.instances)
    def add_folder(self, filename, reducedata=1.0, seed=0):
        ## add dataset, optionally sampling a subset of the dataa
        all_paths = [f for f in list(pathlib.Path(filename).iterdir()) if str(f).endswith('txt')]

        # randomly (but deteerministically) sample a number of file id once for that dataset
        file_ids = np.array(list(range(len(all_paths))))
        np.random.seed(seed+len(all_paths))
        np.random.shuffle(file_ids)
        if reducedata >1:
            trainset =  int(reducedata)
            if trainset > len(all_paths):
                logging.error(f"not enough files to get {str(reducedata)} files in {str(filename)}!")
                trainset = len(all_paths)
        else:
            trainset = int(float(len(all_paths)) * reducedata)
        subset = file_ids[:trainset]

        # convert those file IDs to a list of files
        all_paths = [file for file_id, file in enumerate(all_paths) if file_id in subset]

        # add each file to the data
        for each_path in all_paths:
            if str(each_path).endswith(".txt"):
                q = str(each_path)[:-4]+".ann"
                if pathlib.Path(q).exists():
                    self.file2conll(str(each_path), q)
                else:
                    logging.error(f"doesn't exist: {q}")


class DataCollator:
    def __init__(self, tokenizer, name, tagdict, seed):
        self.token = tokenizer
        self.tagdict = tagdict
        self.datasets = {}
        self.dataset_name =name
        self.predictions = {}
        self.seed =seed
    def add_dataset(self, split_name, folder, batch_size,reducedata=1.0):
        ## add dataset, optionally sampling a subset of the dataa
        self.datasets[split_name], predictions= self.load_dataset(folder, self.token, batch_size=batch_size,reducedata=reducedata)
        self.predictions[split_name] = self.predictions.get(split_name, {})
        self.predictions[split_name].update(predictions)
    def data_collator(self, batch):
        ### tokenize, collate, convert raw data to tokenized labels
        output = []
        attention = []
        b = self.token([x['sentence'] for x in batch], max_length=256, return_offsets_mapping=True, padding=True, truncation=True)
        raw_words = []
        labels = []
        targets = []
        offset_box = []
            
        for sid, st in enumerate(b['input_ids']):
            ner_mapping = batch[sid]['raw_ner']
            offset = batch[sid]['offset']
            ner_mapping = {int(x):ner_mapping[x] for x in ner_mapping}
            om = b['offset_mapping'][sid]
            pd = []
            offsets = []
            yyy = []
            for mid, mapping in enumerate(om):

                ## This is all ugly code to handle "First" subword pooling
                if not mapping[1] == 0:
                    items = [ner_mapping.get(x,"O") for x in range(mapping[0], mapping[1])]
                    label = sorted(items)[0]
                    if (label.startswith("I-") or label.startswith("B-")) and "O" in items:
                        pass
                    if label.startswith("B-") and self.token.convert_ids_to_tokens(st)[mid].startswith("#"):
                        pass
                    elif self.token.convert_ids_to_tokens(st)[mid].startswith("#"):
                        label ="X"
                    if label != "X":
                        yyy.append(mid)
                elif mid ==0:
                    label = "O"
                    yyy.append(mid)
                else:
                    label = "B-PAD"
                    yyy.append(mid)
                locs = (batch[sid]['filename'], mapping[0]+offset, mapping[1]+offset)

                self.tagdict[label] = self.tagdict.get(label, len(self.tagdict) )
                offsets.append(locs)
                pd.append(label)
            attention.append(b['attention_mask'][sid])
            output.append(st)
            offset_box.append(offsets)
            labels.append([self.tagdict[x] for x in pd])
            raw_words.append(self.token.convert_ids_to_tokens(st))
        return {'input_ids':torch.LongTensor(output),'raw':raw_words,'offsetbox':offset_box,'attention_mask':torch.LongTensor(attention),"labels":torch.LongTensor(labels)}

    def load_dataset(self, folder, tokenizer, batch_size=16, reducedata=1.0):
        nd = NERDataset(tokenizer)
        nd.add_folder(folder, reducedata, seed=self.seed)
        
        conll_matsci_loader = DataLoader(nd, batch_size=batch_size, shuffle=True, collate_fn=self.data_collator)
        return conll_matsci_loader, nd.predictions

    def heatmap(self, gold_list, pred_list, labelsubset, cmap, heatmap_loc):
        ovoc = ["NULL"] + sorted(list(set(labelsubset)))
        vv = {x:ovoc.index(x) for x in ovoc}
        counts = {}
        for qid, query in enumerate(gold_list):
            candidate = pred_list[qid]
            if query in labelsubset or candidate in labelsubset:
                
                sco = (vv.get(query, 0),vv.get(candidate, 0))
                counts[sco] = counts.get(sco, 0) +1
        nz = numpy.zeros((len(ovoc), len(ovoc)))
        for p in counts:
            nz[p[0], p[1]] = counts[p]

        align = numpy.sum(nz, 1) #+0.001
        nz = nz /align[:, numpy.newaxis]
        nz = numpy.round(nz, decimals=2)
        nz_mod = (0.9 * nz)+1.00001
        xlabs = [x.replace("Nonrecipe-operation","Nonrecipe-op.") for x in ovoc]
        ylabs = [x.replace("Nonrecipe-operation","Nonrec.-op.") for x in ovoc]
        ax = sns.heatmap(nz_mod,annot=nz,  xticklabels=xlabs, yticklabels=ylabs,  cmap=cmap, norm=LogNorm(vmin=1, vmax=numpy.max(nz)+0.2)   , linewidth=0.5 , annot_kws={'fontsize':"x-large"}, fmt='.2g', cbar=False)     
        ax.invert_yaxis()
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, horizontalalignment='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right')
        plt.tight_layout()
        plt.savefig(heatmap_loc)
        plt.clf()

    def evaluate(self, split_name, prediction_dictionary, heatmap_loc=None):
        assert(split_name in self.predictions)
        gold_locations = {}
        predicted_locations = {}
        for file in self.predictions[split_name]:
            for location in self.predictions[split_name][file]:
                loc = (file, int(location[1]), int(location[2]))
                label = location[0]
                gold_locations[loc] = label
        for file in prediction_dictionary:
            for location in prediction_dictionary[file]:
                loc = (file, int(location[1]), int(location[2]))
                label = location[0]
                predicted_locations[loc] = label
        all_items = list(set(list(gold_locations)+list(predicted_locations)))
        gold = []
        pred = []
        for item in all_items:
            gold.append(gold_locations.get(item, "NULL"))
            pred.append(predicted_locations.get(item, "NULL"))
        labels = ["NULL"] + [x for x in list(set(gold+pred)) if not x == "NULL"]
        f1= f1_score([labels.index(x) for x in gold],[labels.index(x) for x in pred], labels=[labels.index(x) for x in labels if not x =="NULL"], average='micro')
        report = list(confusion_matrix([labels.index(x) for x in gold],[labels.index(x) for x in pred]))
        scores = {"f1":f1,'labels':labels}
        #ops = ["Operation","Nonrecipe-operation", "Meta"]
        #mats = ["Material","Nonrecipe-Material","Sample","Target","Unspecified-Material"]
        #try:
        #    self.heatmap(gold, pred, labelsubset=mats, cmap='Reds', heatmap_loc=heatmap_loc+"/"+"material.png")
        #    self.heatmap(gold, pred, labelsubset=ops, cmap='Reds', heatmap_loc=heatmap_loc+"/"+"operations.png")
        #except:
        #    print("ISSUES DOING HEATMAP!!!")
        #for interesting_type in []


        #eee = {"mat":["Material", "Nonrecipe-Material","Unspecified-Material","Sample", "Target"],
        #"opr":["Operation","Nonrecipe-operation","Meta"],
        #"other":["Amount-Unit", "Property-Unit", "Condition-Unit", "Number","Property-Unit", "Synthesis-Appratus"]}#, "Apparatus-Descriptor", "Synthesis-Apparatus", "Property-Misc", "Brand", "Characterization-Apparatus", "Meta", "Apparatus-Unit", "Amount-Misc", "Number", "Condition-Type", "Apparatus-Property-Type", "Operation", "Condition-Misc", "Property-Type", "Reference", "Material-Descriptor", "Condition-Unit"]
        #specials = {}
        #for mentiontype in eee:
        ##if "Material" in labels:
        #    f1mat= f1_score([labels.index(x) for x in gold],[labels.index(x) for x in pred], labels=[labels.index(x) for x in labels if x in eee[mentiontype]], average='weighted')
        #    specials[mentiontype] = f1mat
        
        #scores["types"] = specials
        return scores