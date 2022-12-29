#importing the libraries
seed_val = seed_trn = 42
import torch
from torch import nn
import torch.nn.functional as F
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from features import foi, ic, num, obj, ct, triplets, cl_features, x_n
import numpy as np
import os
import re
import time
import random
import datetime
from rouge import Rouge
#import dependencies
from itertools import chain
#import required libraries
import os #operating system utils
import pandas as pd #data manipulation package
import numpy as np #numerical operation package
#import matplotlib.pyplot as plt #plotting function
import nltk
import pickle
import re
import glob
import langid
import shutil
from tqdm import tqdm, trange
from sklearn.utils.class_weight import compute_sample_weight #for using sample weight...
from os.path import join
from collections import Counter
from multiprocessing.dummy import Pool as ThreadPool
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from Similarity import similarity as sm
from collections import Counter
from sklearn.preprocessing import StandardScaler
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import FrenchStemmer, PorterStemmer, ItalianStemmer
#--------transformer utils -------------------------------------------------------
import gc
from torch.utils.tensorboard import SummaryWriter
from pytorchtools import EarlyStopping
from torch.utils.data import Dataset, random_split
from transformers import (WEIGHTS_NAME, CONFIG_NAME,
                            GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, #GPT Model
                            BertTokenizer, BertForMaskedLM, BertConfig, #Bert Model
                            RobertaTokenizer, RobertaForMaskedLM, RobertaConfig, #Roberta Molde
                            DistilBertTokenizer, DistilBertForMaskedLM, DistilBertConfig, #DistilBert Model
                            XLNetTokenizer, XLNetLMHeadModel, XLNetConfig, #XLNET Model
                            XLMTokenizer, XLMWithLMHeadModel, XLMConfig, #XLM Model
                            TransfoXLTokenizer, TransfoXLLMHeadModel, TransfoXLConfig, #TransfoXL Model
                            OpenAIGPTTokenizer, OpenAIGPTLMHeadModel, OpenAIGPTConfig, #OpenAIGPTT Model
                            CTRLTokenizer, CTRLLMHeadModel, CTRLConfig, #CTRL Model
                            )
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler, DistributedSampler
from transformers import pipeline, set_seed
torch.manual_seed(seed_trn)
#---bleu evaluation------------------------------------
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate import ribes_score, meteor_score
#--LESE eveluation 
from LESE import LESE
#--- logging
import logging
import argparse
logging.basicConfig(format="", level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

#--
path = os.getcwd()

#%% Preprocessing...


class Mergefeatures(object):
    def __init__(self, string):
        super(Mergefeatures, self).__init__()
        self.string = string
        return
    
    def concat(self):
        '''Concatenate along the horizontal axis
        '''
        z = ','.join(y.strip('[]') for y in self.string)
        z = [x.strip().strip("''") for x in z.split(',')]
        #return ' '.join(x for x in z if not x == 'nan')
        z = ' '.join(x for x in z if not x == 'nan' if not x == ' ' if not x == '')
        z = [x for x in z.split(' ')]
        return z
    
def prepretreat(x, stopword = None, threshold = None):
    '''Docstring

    Parameters
    ----------
    x : string type
        word/sentence string.
    threshold : TYPE, optional
        threshold for cutting words. The default is None.

    Returns
    -------
    list
        list of pretreated words.

    '''
    if not threshold:
        threshold = 3
    else:
        threshold = threshold
    if not stopword:
        with open(join(path, 'stopwords.txt'), 'r+', encoding="utf8") as st: #note that you need to define path in the function
            stopwords = set([x for x in st.read().split()])
    else:
        stopwords = stopword
    txt = ','.join(list(set([re.sub(r'[^\w+]', '', x.lower()) for x in set(''.join(str([str(ii).strip() for ii in x])).split())])))
    txt = ' '.join(x for x in txt.split(',') if x not in stopwords if not len(x) < threshold if not any(z.isdigit() for z in x)) #remove stowords etc
    return ' '.join(re.sub('\[^a-zA-Z0-9\n\.]', ' ', x) for x in txt.split(' ') if not len(x) < threshold if not any(z.isdigit() for z in x)) #remove special characters from string


#%% Evaluation function

def evaluate(args, eval_dataset, model, tokenizer, prefix=""):
    eval_output_dir = args.eval_dir

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler = eval_sampler, batch_size = args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Evaluation!
    logger.info("   ***** Running evaluation {} *****".format(prefix))
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Batch size = {args.eval_batch_size}")
    eval_loss, perplexity = 0.0, 0.0
    nb_eval_steps = 0
    model.eval()
    for batch in tqdm(eval_dataloader, desc = "Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)
        #No optimization during evaluation. i.e mini-batch gradient descent not needed.
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'labels':         batch[0],
                      'attention_mask': batch[1],
                      }
            outputs  = model(inputs['input_ids'], 
                             attention_mask = inputs['attention_mask'],
                             labels = inputs['labels']
                             )
            tmp_eval_loss = outputs['loss']
            avg_tmp_eval_loss = tmp_eval_loss.mean().item() #average batch evaluation loss
            avg_templ_ppl = np.exp(avg_tmp_eval_loss) #average batch perplexity
            eval_loss += avg_tmp_eval_loss #total inreamental loss
            perplexity += avg_templ_ppl #
            # avg_eval_loss.append(avg_tmp_eval_loss)
            # ppl.append(avg_templ_ppl)
        nb_eval_steps += 1
    
    eval_loss = eval_loss / nb_eval_steps
    # np.save(join(eval_output_dir, f'avg_evaluation_loss_{eval_loss}.npy'), avg_eval_loss)
    # np.save(join(eval_output_dir, f'avg_ppl_{perplexity}.npy'), ppl)
    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("   ***** Eval results *****")
        logger.info(f"  Evaluation loss: {eval_loss} PPL: {perplexity}")
        writer.write(f"Evaluation loss: {eval_loss} PPL: {perplexity}")
    return eval_loss, perplexity


#%% Training function

#KL loss

def loss_fn_kl(output, target, model, args, labels, num_tokens, device):
    lprobs = F.log_softmax(output, dim=-1)
    loss = F.nll_loss(lprobs.view(-1, lprobs.size(-1)), target.view(-1), reduction='sum')
    loss = loss / num_tokens
    # KL Divergence
    kl_loss = 0
    for p in model.parameters():
        if p.requires_grad:
            kl_loss += 0.5 * torch.sum(torch.pow(p, 2))
    kl_loss = kl_loss * args.kl_lambda
    loss = loss + kl_loss
    return loss

def _rotate_checkpoints(args, checkpoint_prefix = 'checkpoint', use_mtime = False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    output_dir = os.path.abspath(args.output_dir)
    checkpoints = [output_dir]
    if os.path.isdir(output_dir):
        checkpoints = list(os.path.join(output_dir, n) for n in os.listdir(output_dir))
        if args.local_rank not in [-1, 0]:
            checkpoints = [checkpoint for checkpoint in checkpoints if torch.distributed.get_rank() == int(checkpoint.split('-')[-1])]
        checkpoints.sort(key=lambda x: int(x.split('-')[-1]) if len(x.split('-')) > 1 else 0)
        if len(checkpoints) > args.save_total_limit:
            logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoints[0]))
            shutil.rmtree(checkpoints[0])
            
#Defining the training loop
def train(args, train_dataset, eval_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler = train_sampler, batch_size = args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr = args.learning_rate, eps = args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = args.warmup_steps, num_training_steps = t_total)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, 'optimizer.pt')) and os.path.isfile(os.path.join(args.model_name_or_path, 'scheduler.pt')):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'optimizer.pt')))
            scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'scheduler.pt')))    

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level = args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization) set local_rank = -1 for Non-distributed training
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids = [args.local_rank],
                                                            output_device = args.local_rank,
                                                            find_unused_parameters = True)

    # Train!
    logger.info("  ***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per GPU = {args.per_gpu_train_batch_size}")
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),)
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {t_total}")

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split('-')[-1].split('/')[0])
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            global_step = 0
            logger.info("  Start fine-tuning...")

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc = "Epoch", disable = args.local_rank not in [-1, 0])
    set_seed(args.seed)  # Added here for reproducibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable = args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            
            #begin training
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                      'input_ids':      batch[0],
                      'labels':         batch[0],
                      'attention_mask': batch[1],
                      }
            #revisit if the result is unsatisfactory
            # if args.model_type != 'distilbert':
            #     inputs['token_type_ids'] = None if args.model_type == 'xlm' else batch[3]
            outputs  = model(inputs['input_ids'], 
                             attention_mask = inputs['attention_mask'],
                             labels = inputs['labels']
                             )
            
            loss = outputs['loss']  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                
        #---model evaluation
        if args.local_rank in [-1, 0]:
            # Log metrics
            if args.local_rank == -1 and args.evaluate_during_training:
                eval_loss, ppl = evaluate(args, eval_dataset, model, tokenizer) #evaluation here
                tb_writer.add_scalar(f'Eval loss: {eval_loss} Perplexity: {ppl}', global_step)
            tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
            tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
            logging_loss = tr_loss
        
    #-----save model
    if args.local_rank in [-1, 0]:
        # Save model checkpoint
        output_dir = os.path.join(args.output_dir, f'checkpoint-{global_step}')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        logger.info(f"  Saving model checkpoint to {output_dir}")

        _rotate_checkpoints(args, checkpoint_prefix = 'checkpoint')
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        logger.info("  Saving optimizer and scheduler states to {output_dir}")

        # if args.max_steps > 0 and global_step > args.max_steps:
        #     epoch_iterator.close()
        #     break
    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


#%% Main

def main():
    #Model config, Model and their respective tokenizers
    MODEL_CLASSES = {
                    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
                    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
                    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
                    'xlnet': (XLNetConfig, XLNetLMHeadModel, XLNetTokenizer),
                    'xlm': (XLMConfig, XLMWithLMHeadModel, XLMTokenizer),
                    'transfo-xl': (TransfoXLConfig, TransfoXLLMHeadModel, TransfoXLTokenizer),
                    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
                    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
                    'gpt2-medium': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
                    'gpt2-large': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
                    'ctrl': (CTRLConfig, CTRLLMHeadModel, CTRLTokenizer),
                    }
    
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_type",
                        default = None,
                        type = str,
                        required = True,
                        help = "Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path to pre-trained model or shortcut name selected in the list")
    parser.add_argument("--output_dir",
                        default = None,
                        type = str,
                        required = True,
                        help = "The output directory where the model results and checkpoints will be written.")
    parser.add_argument("--eval_dir",
                        default = None,
                        type = str,
                        required = True,
                        help = "The output directory where the evaluation metrics are stored.")
    ## Other parameters
    parser.add_argument("--config_name",
                        default = "",
                        type = str,
                        help = "Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name",
                        default = "",
                        type = str,
                        help = "Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--year",
                        default = 2019,
                        type = int,
                        help = "Year reference for failure analysis dataset")
    parser.add_argument("--max_seq_length",
                        default = 128,
                        type = int,
                        help = "The maximum total input sequence length after tokenization. Sequences longer "
                               "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--seed",
                        type = int,
                        default = 42,
                        help = "random seed for initialization")
    parser.add_argument("--bos_token",
                        type = str,
                        default = '<|startoftext|>',
                        help = "random seed for initialization")
    parser.add_argument("--eos_token",
                        type = str,
                        default = '<|endoftext|>',
                        help = "random seed for initialization")
    parser.add_argument("--pad_token",
                        type = str,
                        default = '<|pad|>',
                        help = "random seed for initialization")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training",
                        action = 'store_true',
                        help = "Rule evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case",
                        action = 'store_true',
                        help = "Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size",
                        default = 1,
                        type = int,
                        help = "Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size",
                        default = 1,  
                        type = int,
                        help = "Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps",
                        type = int,
                        default = 1,
                        help = "Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--save_total_limit",
                        type = int,
                        default = None,
                        help = "Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default")
    parser.add_argument("--learning_rate", 
                        default = 5e-5,
                        type = float,
                        help  ="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default = 0.0,
                        type = float,
                        help = "Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon",
                        default = 1e-8,
                        type = float,
                        help = "Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm",
                        default = 1.0,
                        type = float,
                        help = "Max gradient norm.")
    parser.add_argument("--num_train_epochs",
                        default = 3.0,
                        type = float,
                        help = "Total number of training epochs to perform.")
    parser.add_argument("--max_steps",
                        default = -1,
                        type = int,
                        help = "If > 0: set total number of training steps to perform. Override num_train_epochs.")  
    parser.add_argument("--warmup_steps",
                        default = 10,
                        type = int,
                        help = "Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps",
                        type = int,
                        default = 500,
                        help="Log every N-updates steps.")
    parser.add_argument("--save_steps",
                        type = int,
                        default = 500,
                        help = "Save checkpoint every N-updates steps.")
    parser.add_argument("--eval_all_checkpoints", #checck this to know if it is worth evaluating all checkpoints and the significance
                        action = 'store_true',
                        help = "Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--overwrite_output_dir",
                        action = 'store_true',
                        help = "Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache",
                        action = 'store_true',
                        help = "Overwrite the cached training and evaluation sets")
    parser.add_argument("--fp16",
                        action = 'store_true',
                        help = "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level",
                        type = str,
                        default = "O1",
                        help = "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                                "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank",
                        type = int,
                        default = -1,
                        help = "For distributed training: local_rank is 0 and -1 for unit gpu")
    args = parser.parse_args()
    
    args.output_dir = join(args.model_name_or_path, args.output_dir)
    args.eval_dir = join(args.model_name_or_path, args.eval_dir)
    #--------------------------------- main
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
        
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
                        format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
                        )
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s", args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,) #Config class
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case = args.do_lower_case,
                                                bos_token = args.bos_token, eos_token = args.eos_token, pad_token = args.pad_token) #Tokenization
    model = model_class.from_pretrained(args.model_name_or_path, from_tf = bool('.ckpt' in args.model_name_or_path), config = config) #LM class
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    logging.info(f'  Embedding size: {embedding_size}')
    
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info(f"  Training/evaluation parameters: {args}")

    # Dataset
    class FailureAnalysisDataset(Dataset):
        def __init__(self, args, txt_list, tokenizer, max_length, wts = None, use_weights = False):
            self.input_ids = []
            self.attn_masks = []
            self.labels = []
            self.wts = wts       #weights GCVAE+GMM...Fixing failure analysis yearly imbalance in dataset
            self.use_weights = use_weights     #probabilistic weights from GCVAE + GMM
            for txt in txt_list:
                encodings_dict = tokenizer(args.bos_token + txt + args.eos_token, truncation = True,
                                            max_length = max_length, padding = "max_length")
                self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
                self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    
        def __len__(self):
            return len(self.input_ids)
    
        def __getitem__(self, idx):
            if not self.use_weights:
                return self.input_ids[idx], self.attn_masks[idx]
            else:
                return self.input_ids[idx], self.attn_masks[idx], self.wts[idx]
    
    TRAIN_DATA_FILE = join(path, f'combine_corpus_{args.year}.csv')
    df_df = pd.read_csv(TRAIN_DATA_FILE, sep = ',')['text']
    max_length = max([len(tokenizer.encode(fa)) for fa in df_df])
    
    dataset = FailureAnalysisDataset(args, df_df, tokenizer, max_length = max_length)
    train_size = int(0.7 * len(dataset)) #split data into 70/30 train-test proportion
    train_dataset, eval_dataset = random_split(dataset, [train_size, len(dataset) - train_size]) #validationsize = len(dataset) - train_size
    
    # Training
    if args.do_train:
        global_step, tr_loss = train(args, train_dataset, eval_dataset, model, tokenizer)
        logger.info(f" global_step = {global_step}, average loss = {tr_loss}")

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            logger.info("Evaluate the following checkpoints: %s", checkpoints)
            for checkpoint in checkpoints:
                global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
                model = model_class.from_pretrained(checkpoint)
                model.to(args.device)
                eval_loss, ppl = evaluate(args, model, tokenizer, prefix=global_step)
                logger.info(f"  Evaluation loss: {eval_loss} PPL: {ppl}")

    #---- Generation evaluation
    x_nns = ['REFERENCE',
              'Subject',
              'Site',
              'Requested activity',
              'Priority level',
              'High confidentiality',
              'Context',
              'Objectives / Work description',
              'Source of failure / request',
              'Source of failure (Detailled)',
              ]
    
    df_xn_lda = pd.read_csv(join(path, f'XN_LAMBDA_BF_PREF_{args.year}.csv'), sep = ',')
    df_xn_lda_xx = pd.read_csv(join(path, 'XN_STDATA_EN.csv'), sep = ',')
    df_xn_lda_xx['LASTTRANSITIONDATE'] = df_xn_lda_xx['LASTTRANSITIONDATE'].apply(pd.to_datetime)
    df_xn_lda_xx = df_xn_lda_xx.iloc[:df_xn_lda.shape[0], :]
    
    lambdas = [x for x in df_xn_lda.columns if 'lambda' in x] #for extracting triplets \lambda = {step type, substep technique; equipment}
    x_df = df_xn_lda_xx[x_nns].astype(str).apply(lambda x: Mergefeatures(x).concat(), axis = 1)
    x_df = x_df.apply(lambda x: prepretreat(x),)
    predictor_corpus = list(x_df) 
    #targets
    target = df_xn_lda[lambdas].astype(str).apply(lambda x: Mergefeatures(x).concat(), axis = 1)
    target = target.apply(lambda x: ' '.join(x))
    target = list(target)
    
    # Generative metric evaluation
    bluescore = []
    bluescore_3 = []
    meteor_score_s = []
    model_generated_fas = []
    lev_d_1, prec_lev_1, rec_lev_1, fs_lev_1 = [], [], [], []
    lev_d, prec_lev, rec_lev, fs_lev = [], [], [], []
    for ii, ij in zip(predictor_corpus, target):
        logger.info(f'  --> Failure description: {ii}\n')
        logger.info(f'  Expert FA: {ij}\n')
        start_prompt = ii
        # start_prompt = 'data castelletto customer manufacturing limit axis analysis failure complaint failed thd abnormal'
        start_tokens = tokenizer(start_prompt, return_tensors="pt").input_ids.to(device)
        sampled_generated_outputs_token = model.generate(start_tokens, do_sample = True, top_k = 10, max_length = max_length, top_p = 0.95, 
                                                         temperature = 1.9, num_return_sequences = 1, pad_token_id = tokenizer.eos_token_id
                                                         )
        generated_text_is = []
        for _, s_output in enumerate(sampled_generated_outputs_token):
            logger.info(f"A  I generated FA: {tokenizer.decode(s_output[len(start_tokens[0]):], skip_special_tokens = True)}")
            generated_text_is.append(tokenizer.decode(s_output[len(start_tokens[0]):], skip_special_tokens = True))
        prediction = " ".join(generated_text_is)
        model_generated_fas.append(prediction) #to be used for scoring ROUGE and BLEU
        #bleu score
        chencherry = SmoothingFunction()
        bluescore.append(sentence_bleu([ij.split(' ')], prediction.split(' '), weights=(1, 0, 0, 0), smoothing_function = chencherry.method2))
        bluescore_3.append(sentence_bleu([ij.split(' ')], prediction.split(' '), weights=(0.33, 0.33, 0.33, 0), smoothing_function = chencherry.method2))
        #meteor scores
        meteor_score_s.append(nltk.translate.meteor_score.meteor_score([ij.split(' ')], prediction.split(' ')))
        #lese score
        lese_1 = LESE(ij.split(' '), prediction.split(' '), 1, False)
        lese = LESE(ij.split(' '), prediction.split(' '), 3, False)
        #---------
        lev_d_1.append(lese_1.levenshstein_distance)
        prec_lev_1.append(lese_1.precision_)
        rec_lev_1.append(lese_1.recall_)
        fs_lev_1.append(lese_1.f_score_)
        #-------
        lev_d.append(lese.levenshstein_distance)
        prec_lev.append(lese.precision_)
        rec_lev.append(lese.recall_)
        fs_lev.append(lese.f_score_)
        logger.info('*'*50)
    less_scores_1 = {'lev_d': lev_d_1, 'prec_lev': prec_lev_1, 'rec_lev': rec_lev_1, 'fs_lev': fs_lev_1}
    less_scores = {'lev_d': lev_d, 'prec_lev': prec_lev, 'rec_lev': rec_lev, 'fs_lev': fs_lev}
    logger.info(f"  Average blue score: {np.mean(bluescore)}")
    np.save(os.path.join(args.eval_dir, f"pt_{args.model_name_or_path}_bleuscore_{len(x_n)}_{args.num_train_epochs}_{args.year}.npy"), bluescore)
    logger.info('  *********************************Done computing self-BELU score*********************************')
    logger.info(f"  Average blue-3 score: {np.mean(bluescore_3)}")
    np.save(os.path.join(args.eval_dir, f"pt_{args.model_name_or_path}_bleuscore3_{len(x_n)}_{args.num_train_epochs}_{args.year}.npy"), bluescore_3)
    logger.info('  *********************************Done computing self-BELU score**********************************************')
    np.save(os.path.join(args.eval_dir, f"pt_{args.model_name_or_path}_lese1_{len(x_n)}_{args.num_train_epochs}_{args.year}.npy"), less_scores_1)
    logger.info(f'  LESE Precision: {np.mean(prec_lev_1)}\nLESE Recall: {np.mean(rec_lev_1)}\nLESE F1-score: {np.mean(fs_lev_1)}')
    logger.info('****************************************Done computing LESE score**********************************************')
    np.save(os.path.join(args.eval_dir, f"pt_{args.model_name_or_path}_lese3_{len(x_n)}_{args.num_train_epochs}_{args.year}.npy"), less_scores)
    logger.info(f'  LESE Precision: {np.mean(prec_lev)}\nLESE Recall: {np.mean(rec_lev)}\nLESE F1-score: {np.mean(fs_lev)}')
    logger.info('*************************************Done computing LESE score***********************************************')
    #------------------ Remove empty hypothesis before computing ROUGE and METEOR scores -------------------------
    hyps_and_refs = zip(model_generated_fas, target)
    hyps_and_refs = [ x for x in hyps_and_refs if len(x[0]) > 0]
    model_generated_fas, target = zip(*hyps_and_refs)
    #-------------------------------------------------------------------------------------------------------------
    rouge = Rouge()
    rouge_score = rouge.get_scores(model_generated_fas, target, avg = True)
    logger.info(f'  ROUGE SCORES: {rouge_score}')
    np.save(os.path.join(args.eval_dir, f"pt_{args.model_name_or_path}_rouge_{len(x_nns)}_{args.num_train_epochs}_{args.year}.npy"), rouge_score)
    logger.info('  ************************************Done computing ROUGE score*****************************************')
    logger.info(f"  Average metoer score: {np.mean(meteor_score_s)}")
    np.save(os.path.join(args.eval_dir, f"pt_{args.model_name_or_path}_meteor_{len(x_n)}_{args.num_train_epochs}_{args.year}.npy"), meteor_score_s)
    logger.info('  **************************************Done computing METEOR score**************************************')


if __name__ == "__main__":
    main()
    
    
    
