

import torch
import random
import spacy
import json
import logging

from datasets import ClassLabel, load_dataset, load_metric
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader, Dataset
import pathlib
import os
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
import numpy as np
from transformers import DistilBertForSequenceClassification, AdamW

from tqdm import tqdm
import argparse
from data import *
from model import *
import json
import logging


logging.basicConfig(level=logging.INFO, format='%(message)s')



def save_folder(saveloc, multitask_model, collators):
    if not pathlib.Path(saveloc).exists():
        pathlib.Path(saveloc).mkdir()
    print("SAVING MODEL")

    torch.save(multitask_model, saveloc+"/"+"model.pt")
    vocabs = {col:collators[col].tagdict for col in collators}
    json.dump(vocabs, open(saveloc+"/"+"vocab.json",'w'))

def main(args):
    torch.manual_seed(args.seed)
    pretrain_models = args.pretrain.split(",")
    USECUDA =torch.cuda.is_available()
    print(USECUDA)

    #DLOC = '../cmumtl/cmumtl/data/'
    DLOC = args.dloc
    #DLOC = '../cmumtl/data/'
    #sDLOC = '../../cmumtl/data/'
    
    #model_name_or_path ='allenai/scibert_scivocab_uncased'
    model_name = args.model
    #model_name = 'bert-base-uncased'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    metric = load_metric("seqeval")

    all_data = {}#"sofc":sofc_loader, "mspt":conll_matsci_loader,"matsci":conll_matsci2_loader}
    all_dev_data = {}#"sofc":sofc_dev_loader, "mspt":conll_matsci_dev_loader,"matsci":conll_matsci2_dev_loader}
    all_test_data = {}
    collators = {}#"sofc":sofc_col, "mspt":matsci2019col,"matsci":matsci2020col}

    #if 'sofc' in pretrain_models+[args.main]:
    #    sofc_col = DataCollator(tokenizer)
    #
    #    sofc_loader  = sofc_col.load_dataset('brat/', tokenizer, batch_size=args.batchsize)
    #    sofc_dev_loader  = sofc_col.load_dataset('bratdev/', tokenizer,batch_size=args.batchsize)
    #    
    #    all_data['sofc'] = sofc_loader
    #    all_dev_data['sofc'] = sofc_dev_loader
    #    collators['sofc'] =sofc_col
    if args.loadmodel is not None:
        vocabs = json.load(open(args.loadmodel+"/"+"vocab.json"))
    else:
        vocabs = {}
    dat = {'mspt':DLOC+'matsci2019/brat/','matsci':DLOC+'matsci2020/brat/','matsci2019':DLOC+'matsci2019/brat/','sofc':DLOC+'sofc/brat/','wlp':DLOC+'wlp/brat/'}    
    dat['ms20random'] = DLOC+"/random/"
    dat['ms20domain'] = DLOC+"/Li-Battery/"
    dat['ms20d2g'] = DLOC+"/domain2general/"

    dat['ms20g2d'] = DLOC+"/general2domain/"
    split_list = ['train','dev']
    if args.usetest:
        split_list.append('test')
    
    for corp in dat:

        if corp in pretrain_models+[args.main]:
            tags = vocabs.get(corp, {"O":0, "B-PAD":1, "X":2})
            collators[corp] =DataCollator(tokenizer,corp, tagdict=tags, seed=args.seed)
            for split_name in split_list:
                if corp == args.main and split_name =='train':
                    train_size =args.trainsize
                else:
                    train_size =1.0             
                collators[corp].add_dataset(split_name, dat[corp]+split_name+"/", args.batchsize, reducedata=train_size)


    #if 'wlp' in pretrain_models+[args.main]:
    #    tags = vocabs.get('wlp', {"O":0, "B-PAD":1, "X":2})
    #    collators['wlp'] =DataCollator(tokenizer,'wlp', tagdict=tags)
    #    for split_name in ['train','dev','test']:
    #        collators['wlp'].add_dataset(split_name, DLOC+'../../../wnut/data//'+split_name+"_data/Standoff_Format/", args.batchsize)
    for collator in collators:
        for split in split_list:
            if split in collators[collator].datasets:
                for batch in collators[collator].datasets[split]:
                    pass
    
    #model_type_dict = {}
    #m#odel_config_dict = {}
    
    max_vocab_size_dict = {task:max(collators[task].tagdict.values())+1 for task in collators}
    use_crf=bool(args.crf.lower()=='true')
    multitask_model = MultitaskModel.create(
        model_name=model_name,
        model_dim_list=max_vocab_size_dict,
        use_crf=use_crf)
    multitask_model.seed = args.seed

    #input(">>>>")
    #for task in collators:

    #    model_type_dict[task] = transformers.AutoModelForTokenClassification
    #    model_config_dict[task] = transformers.AutoConfig.from_pretrained(model_name, num_labels=max(collators[task].tagdict.values())+1)


    #multitask_model = MultitaskModel.create(
    #    model_name=model_name,
    #    model_type_dict=model_type_dict,
    #    model_config_dict=model_config_dict,
    

    if args.bertmodel is not None:
        mod = AutoModelForMaskedLM.from_pretrained(args.bertmodel)
        for task in multitask_model.taskmodels_dict:
            multitask_model.taskmodels_dict[task].bert = mod.bert

    if USECUDA:
        device = torch.cuda.current_device()

        multitask_model =multitask_model.to(device)
        multitask_model.to_cuda()  
    if args.crf.lower() =='true':
        for task in collators:
            num_labels = max(collators[task].tagdict.values())+1
            multitask_model.add_crf(task, num_labels, 1)
    if args.loadmodel is not None:
        multitask_model = torch.load(args.loadmodel+"/"+"model.pt")
    optim = AdamW(multitask_model.parameters(), lr=args.lr)
    all_data = {collators[dataset].dataset_name:collators[dataset].datasets['train'] for dataset in collators if 'train' in collators[dataset].datasets}    
    all_dev_data = {collators[dataset].dataset_name:collators[dataset].datasets['dev'] for dataset in collators if 'dev' in collators[dataset].datasets}    
    if args.usetest:
    
        all_test_data = {collators[dataset].dataset_name:collators[dataset].datasets['test'] for dataset in collators if 'test' in collators[dataset].datasets}    
    
    if args.preepochs > 0:
        for epoch in range(args.preepochs):
            multitask_model.train({x:all_data[x] for x in all_data if x in pretrain_models}, optim,  args.gradacc, alpha=0.8)
            for task in pretrain_models:
                f1, prediction_dict= multitask_model.eval(all_dev_data[task], task, collators[task].tagdict)
                print(task, f1)
    
    if args.savemodel:
        save_folder(args.savemodel, multitask_model, collators)
        lfile = open(args.savemodel+"/"+"log.jsonl",'w')
        image_loc = args.savemodel+"/"
    else:
        lfile = open("temp.log.txt",'w')
        image_loc = None
    predictions = []
    last_improvement = 0
    epoch=0
    for epoch in range(args.finaltaskepochs):
        print(epoch)
        multitask_model.train({args.main:all_data[args.main]} , optim, 1, alpha=1.0)
        f1, prediction_dict = multitask_model.eval(all_dev_data[args.main], args.main, collators[args.main].tagdict)
        log = collators[args.main].evaluate('dev', prediction_dict, heatmap_loc=image_loc)
        
        lfile.write(json.dumps({"log":log, 'epoch':epoch,'split':'dev'})+"\n")
        print(json.dumps({"log":log, 'epoch':epoch,'split':'dev'})+"\n")
        if f1 > max(predictions+[0]):
            last_improvement = epoch
            if args.savemodel:
                save_folder(args.savemodel, multitask_model, collators)
        if epoch - last_improvement > 10:
            break
        predictions.append(f1)
    if args.savemodel is not None:
        print("LOADING MODEL")

        multitask_model = torch.load(args.savemodel+"/"+"model.pt")
    f1, dev_prediction_dict = multitask_model.eval(all_dev_data[args.main], args.main, collators[args.main].tagdict)
    print("DEV", f1)            
    log = collators[args.main].evaluate('dev', dev_prediction_dict, heatmap_loc=image_loc)
    lfile.write(json.dumps({"log":log, 'epoch':epoch,'split':'dev'})+"\n")
    if args.usetest:
        f1, test_prediction_dict = multitask_model.eval(all_test_data[args.main], args.main, collators[args.main].tagdict)
        print("TEST", f1)        
        log = collators[args.main].evaluate('test', test_prediction_dict)
        lfile.write(json.dumps({"log":log, 'epoch':epoch,'split':'test'})+"\n")
    lfile.close()
parser = argparse.ArgumentParser(description='he')

parser.add_argument('pretrain', type=str,  help='pretrain on all data for how many epochs')
parser.add_argument('main', help="main task")
parser.add_argument('--gradacc', type=int, help="pretraining batches per update", default=0)
parser.add_argument('--bertmodel', type=str, help="model", default=None)
parser.add_argument('--loadmodel', type=str, help="finalmodel", default=None)
parser.add_argument('--savemodel', type=str, help="finalmodel", default=None)
parser.add_argument('--model', type=str, help="coree model", default='allenai/scibert_scivocab_uncased')
parser.add_argument('--crf', type=str, default="False" , help="CRF me?")
parser.add_argument('--batchsize', type=int, help="model", default=8)
parser.add_argument('--finaltaskepochs', type=int, help="final tasak", default=100)
parser.add_argument('--usetest', default=False, action='store_true')
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--trainsize', type=float, help="size of training set", default=1.0)
parser.add_argument('--seed', type=int, help="seed", default=1)
parser.add_argument('--preepochs', type=int,  default=0, help='pretrain on all data for how many epochs')
parser.add_argument('--dloc',  default='../../cmumtl/data/', help='pretrain on all data for how many epochs')

args = parser.parse_args()

main(args)