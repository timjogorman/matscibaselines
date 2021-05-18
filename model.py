import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForTokenClassification,
    ElectraForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    AutoModelForMaskedLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)

import torchcrf
import random
import numpy as np
from tqdm import tqdm
import math
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2


class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict, use_crf):
        """
        Just a encoder with a moduledict of classifiers;
        Would rquir meaningful edits to mak more generally MTL (i.e. for DAPT)
        """
        super().__init__(transformers.PretrainedConfig())

        self.encoder = encoder
        self.taskmodels_dict = torch.nn.ModuleDict(taskmodels_dict)
        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=1)
        self.crf_dict ={}
        self.use_crf = use_crf
        self.dropout = torch.nn.Dropout(0.1)
        self.do_usecuda = False
        self.seed= 0
    def to_cuda(self):
        self.do_usecuda = True
        self.my_device_id = torch.cuda.current_device()

    @classmethod
    def create(cls, model_name, model_dim_list, use_crf=False):
        """
        Create a linear layer for each task
        """
        shared_encoder = AutoModel.from_pretrained(model_name)
        taskmodels_dict = {}
        for task_name, num_labels in model_dim_list.items():
            taskmodels_dict[task_name]= torch.nn.Linear(shared_encoder.config.hidden_size, num_labels)

        return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict, use_crf=use_crf)
    
    def add_crf(self, task, num_labels, pad_idx):
        self.crf_dict[task] = torchcrf.CRF(num_labels=num_labels, pad_idx=pad_idx, use_gpu=self.do_usecuda) 
        self.crf_dict[task].to(self.my_device_id)

    def forward(self, task_name, **kwargs):
        prediction = self.taskmodels_dict[task_name](self.encoder(**kwargs))


    def shuffle(self, list_of_tasks):
        # this should be in a separat traineer class
        # 
        np.random.seed(self.seed)
        lrl = list(range(len(list_of_tasks)))
        np.random.shuffle(lrl)
        self.seed +=1
        return [list_of_tasks[x] for x in lrl]

    def prep_data(self, data_loader_dict, alpha):
        # this should be in a separat traineer class
        # load datasets together, sample subset of each in proportion to ^alpha
        # Used so that you can pre-train with multiple datasts without large sets dominating

        task_list = []
        for task in data_loader_dict:
            list_of_tasks = list(data_loader_dict[task])

            list_of_tasks = self.shuffle(list_of_tasks)
            sq = max(1, int(math.pow(len(list_of_tasks), alpha)))
            for item in list_of_tasks[:sq]:
                task_list.append([task, item])
        task_list = self.shuffle(task_list)
            
        return task_list

    def run(self, batch, task):
        # this should be in a separat traineer class
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        raw = batch['raw']
        if self.do_usecuda:
            input_ids = input_ids.to(self.my_device_id)
            attention_mask = attention_mask.to(self.my_device_id)
            labels= labels.to(self.my_device_id)
        amb = attention_mask.byte()#.to(self.my_device_id)
        inp = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.taskmodels_dict[task](inp.last_hidden_state)
        return logits, labels, raw, amb

    def train(self, data_loader_dict, optim, gradacc, alpha):
        # this should be in a separat traineer class
        progress_bar = tqdm(self.prep_data(data_loader_dict, alpha))
        loss_list= []
        checkpoints = 0
        for batch_pair in progress_bar:
            task = batch_pair[0]
            self.taskmodels_dict[task].train()
            batch = batch_pair[1]
            logits, labels,raw, amb = self.run(batch, task)
            loss = self.loss_function(logits.flatten(0, 1), labels.flatten(0, 1))
            if self.use_crf:
                loglikelihood = self.crf_dict[task].forward(logits, labels, mask=amb)
                loss =   -torch.mean(loglikelihood)
            
            loss_list.append(loss.cpu().data.numpy())
            progress_bar.set_description(f"{np.average(loss_list)}")
            loss.backward()
            if checkpoints > gradacc:
                optim.step()
                optim.zero_grad()
                checkpoints =0
            checkpoints +=1
        if checkpoints > 0:
            optim.step()
            optim.zero_grad()


    def eval(self, data_loader, task, tag_dict):
        # this should be in a separat traineer class
        predictions = {}
        predbox, goldbox =[], []
        rev_dict = {tag_dict[x]:x for x in tag_dict}
        progress_bar = tqdm(data_loader)
        self.taskmodels_dict[task].eval()
        o = []
        o2 = []
        for batch in progress_bar:
            logits, labels,raw, amb = self.run(batch, task)
            if self.use_crf:    
                pred = self.crf_dict[task].viterbi_decode(outputs.logits, amb)
            else:
                pred = torch.argmax(logits, 2).cpu().data.numpy().tolist()
            lab=labels.cpu().data.numpy()
            for row_idx in range(batch['input_ids'].shape[0]):
                gd, pd = [] , [] 
                # Converts raw predictions into start-end offsets with labels
                for line_idx in range(batch['input_ids'].shape[1]):
                    if line_idx < len(pred[row_idx]):
                        gold_label =  rev_dict[lab[row_idx, line_idx]]
                        pred_label = rev_dict[pred[row_idx][line_idx]]
                        if pred_label == "X" and len(pd) > 0 and (pd[-1].startswith("B-") or pd[-1].startswith("I-")):
                            pred_label = "I-"+pd[-1][2:]
                        elif pred_label == "X":
                            pred_label = "O"
                        if gold_label == "X" and len(gd) > 0 and (gd[-1][:2] in ["B-","I-"]):
                            gold_label = "I-"+gd[-1][2:]
                        elif gold_label == "X":
                            gold_label = "O"

                        if gold_label != "B-PAD":
                            gd.append(gold_label)
                            pd.append(pred_label)
                sent_offsets = batch['offsetbox'][row_idx]
                goldbox.append(gd)  
                predbox.append(pd)
                stack = []
                file = sent_offsets[0][0]
                predictions[file]= predictions.get(file, [])
                for jid, j in enumerate(pd):
                    if j =="O" or j.startswith("B-"):
                        if len(stack) > 0:
                            file = stack[0][2][0]
                            start = stack[0][2][1]
                            end = stack[-1][2][2]
                            label = stack[0][0]
                            text_span = [x[1] for x in stack]
                            if label[:2]=="B-":
                                label = label[2:]
                            predictions[file]= predictions.get(file, []) + [(label, start, end, " ".join(text_span))]
                        stack = []
                    if j.startswith("B-") or j.startswith("I-"):
                        stack.append([j, raw[row_idx][jid], sent_offsets[jid]])
                goldtext= [(word, gd[word_id]) for word_id, word in enumerate(raw[row_idx]) if word_id < len(gd)]
                predtext= [(word, pd[word_id]) for word_id, word in enumerate(raw[row_idx]) if word_id < len(pd)]
        report =str(classification_report(predbox, goldbox, mode='strict', scheme=IOB2))    
        return f1_score(predbox, goldbox ), predictions

