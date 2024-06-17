#region ###################################### Imports ######################################
import sys
import os
import gc
import re
from datetime import datetime
import logging

import en_core_web_lg # This model is leveraged for every spaCy usage (https://spacy.io/models/en#en_core_web_lg)
from tqdm import tqdm
import numpy as np
import pandas as pd
import json

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel
from transformers import Trainer, TrainingArguments, get_constant_schedule
from transformers import AutoModelForMaskedLM, AutoModelForSequenceClassification
from transformers import DataCollatorForLanguageModeling
from accelerate import Accelerator

#endregion


#region ###################################### Configuration ######################################

#region ################### Arguments parsing ###################
def argument_parsing():
    if (args_count := len(sys.argv)) > 2:
        logging.exception(Exception(f"One argument expected, got {args_count - 1}"))
    elif args_count < 2:
        logging.exception(Exception("You must specify the JSON configuration filepath as first argument"))

    target_dir = sys.argv[1]
    return target_dir

#endregion

#region ################### Configuration file ###################
def get_config_from_file(target_dir):
    if not target_dir.endswith(".json"):
        logging.exception(f"The configuration file {target_dir} needs to have json format (end with .json)")
    elif not os.path.isfile(target_dir):
        logging.exception(f"The JSON configuration file {target_dir} doesn't exist")

    with open(target_dir, "r") as f:
        config = json.load(f)
    return config
#endregion

#endregion


#region ###################################### Data ######################################

#region ################### Load pretreatment ###################
def load_pretreatment(PRETREATED_DATA_PATH):
    with open(PRETREATED_DATA_PATH, "r") as f:
        (train_df_json_str, eval_dfs_jsons) = json.load(f)        
    train_df = pd.read_json(train_df_json_str)
    eval_dfs = {name:pd.read_json(df_json) for name, df_json in eval_dfs_jsons.items()}
    return train_df, eval_dfs

#endregion

#region ################### Data reading ###################
def read_data(DATA_FILEPATH, INDIVIDUAL_NAME_COLUMN, BACKGROUND_KNOWLEDGE_COLUMN):
    if DATA_FILEPATH.endswith(".json"):
        data_df = pd.read_json(DATA_FILEPATH)
    elif DATA_FILEPATH.endswith(".csv"):
        data_df = pd.read_csv(DATA_FILEPATH)
    else:
        logging.exception(f"Unrecognized file extension for data file [{DATA_FILEPATH}]. Compatible formats are JSON and CSV.")    
    assert INDIVIDUAL_NAME_COLUMN in data_df.columns
    assert BACKGROUND_KNOWLEDGE_COLUMN in data_df.columns

    return data_df


def split_data(INDIVIDUAL_NAME_COLUMN, BACKGROUND_KNOWLEDGE_COLUMN, data_df):
    data_df.replace('', np.nan, inplace=True)   # Replace empty texts by NaN

    train_cols = [INDIVIDUAL_NAME_COLUMN, BACKGROUND_KNOWLEDGE_COLUMN]
    train_df = data_df[train_cols].dropna()
    train_df.reset_index(drop=True, inplace=True)

    eval_columns = [col for col in data_df.columns if col not in train_cols]
    eval_dfs = {col:data_df[[INDIVIDUAL_NAME_COLUMN, col]].dropna().reset_index(drop=True) for col in eval_columns}
    
    return train_df, eval_dfs

#endregion

#region ################### Data statistics ###################
def get_individuals(train_df, eval_dfs, individual_name_column):
    train_individuals = set(train_df[individual_name_column])
    eval_individuals = set()
    for eval_df in eval_dfs.values():
        eval_individuals.update(set(eval_df[individual_name_column]))
    all_individuals = train_individuals.union(eval_individuals)
    no_train_individuals = eval_individuals - train_individuals
    no_eval_individuals = train_individuals - eval_individuals

    return train_individuals, eval_individuals, all_individuals, no_train_individuals, no_eval_individuals

def get_individuals_labels(all_individuals):
    sorted_indvidiuals = sorted(list(all_individuals)) # Sort individuals for ensuring same order every time (required for automatic loading)
    label_to_name = {idx:name for idx, name in enumerate(sorted_indvidiuals)}
    name_to_label = {name:idx for idx, name in label_to_name.items()}

    return label_to_name, name_to_label

def show_data_stats(train_df, eval_dfs, no_eval_individuals, no_train_individuals, eval_individuals):
    logging.info(f"Number of background knowledge documents for training: {len(train_df)}")

    eval_n_dict = {name:len(df) for name, df in eval_dfs.items()}
    logging.info(f"Number of protected documents for evaluation: {eval_n_dict}")

    if len(no_eval_individuals) > 0:
        logging.info(f"No protected documents found for {len(no_eval_individuals)} individuals.")
    
    if len(no_train_individuals) > 0:
        max_risk = (1 - len(no_train_individuals) / len(eval_individuals)) * 100
        logging.info(f"No background knowledge documents found for {len(no_train_individuals)} individuals. Re-identification risk limited to {max_risk:.3f}%.")

#endregion

#region ################### Data pretreatment ###################

#region ########## Anonymize background knowledge ##########
def anonymize_bk(train_df, spacy_nlp, ONLY_USE_ANONYMIZED_BACKGROUND_KNOWLEDGE):
    train_anon_df = anonymize_df(train_df, spacy_nlp) # Perform anonymization

    if ONLY_USE_ANONYMIZED_BACKGROUND_KNOWLEDGE:
        train_df = train_anon_df # Overwrite train dataframe with the anonymized version
    else:
        train_df = pd.concat([train_df, train_anon_df], ignore_index=True, copy=False) # Concatenate to train dataframe

    return train_df

def anonymize_df(df, spacy_nlp, gc_freq=5):
    assert len(df.columns) == 2 # Columns expected: name and text

    # Copy
    anonymized_df = df.copy(deep=True)

    # Process the text column
    column_name = anonymized_df.columns[1]
    texts = anonymized_df[column_name]
    for i, text in enumerate(tqdm(texts, desc=f"Anonymizing {column_name} documents")):
        new_text = text

        # Anonymize by NER
        doc = spacy_nlp(text) # Usage of spaCy NER (https://spacy.io/api/entityrecognizer)
        for e in reversed(doc.ents): # Reversed to not modify the offsets of other entities when substituting
            start = e.start_char
            end = start + len(e.text)
            new_text = new_text[:start] + e.label_ + new_text[end:]

        # Remove doc and (periodically) use GarbageCollector to reduce memory consumption
        del doc
        if i % gc_freq == 0:
            gc.collect()

        # Assign new text
        texts[i] = new_text

    return anonymized_df

#endregion

#region ########## Document curation ##########
def document_curation(train_df, eval_dfs, spacy_nlp):
    # Perform preprocessing for both training and evaluation
    df_curation(train_df, spacy_nlp)
    for eval_df in eval_dfs.values():
        df_curation(eval_df, spacy_nlp)

def df_curation(df, spacy_nlp, gc_freq=5):
    assert len(df.columns) == 2 # Columns expected: name and text

    # Predefined patterns
    special_characters_pattern = re.compile(r"[^ \nA-Za-z0-9À-ÖØ-öø-ÿЀ-ӿ./]+")
    stopwords = spacy_nlp.Defaults.stop_words

    # Process the text column (discarting the first one, that is the name column)
    column_name = df.columns[1]
    texts = df[column_name]
    for i, text in enumerate(tqdm(texts, desc=f"Preprocessing {column_name} documents")):
        doc = spacy_nlp(text) # Usage of spaCy (https://spacy.io/)
        new_text = ""   # Start text string
        for token in doc:
            if token.text not in stopwords:
                # Lemmatize
                token_text = token.lemma_ if token.lemma_ != "" else token.text
                # Remove special characters
                token_text = re.sub(special_characters_pattern, '', token_text)
                # Add to new text (without space if dot)
                new_text += ("" if token_text == "." else " ") + token_text

        # Remove doc and (periodically) use force GarbageCollector to reduce memory consumption
        del doc
        if i % gc_freq == 0:
            gc.collect()

        # Store result
        texts[i] = new_text

#endregion

#region ########## Save pretreatment ##########
def save_pretreatment(train_df, eval_dfs, PRETREATED_DATA_PATH):
    logging.info("Saving pretreated data")
    with open(PRETREATED_DATA_PATH, "w") as f:
        f.write(json.dumps((train_df.to_json(),
                            {name:df.to_json() for name, df in eval_dfs.items()})))
    logging.info("Pretreated data saved")

#endregion

#endregion

#endregion


#region ###################################### Build classifier ######################################
# Implementation grounded on HuggingFace's Transformers (https://huggingface.co/docs/transformers/index)

#region ################### Load already trained TRI model ###################
def load_trained_TRI_model(TRI_PIPE_PATH, name_to_label):
    num_labels = len(name_to_label)
    model = AutoModelForSequenceClassification.from_pretrained(TRI_PIPE_PATH, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(TRI_PIPE_PATH)

    return model, tokenizer

#endregion

#region ################### Create base language model ###################
def create_base_model(base_model_name):
    base_model = AutoModel.from_pretrained(base_model_name)
    logging.info(f"Model size = {sum([np.prod(p.size()) for p in base_model.parameters()])}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name) 
    return base_model, tokenizer

#endregion

#region ################### Additional pretraining ###################
def additional_pretraining(base_model, tokenizer, dataset, BASE_MODEL_NAME, DEVICE, PRETRAINING_MLM_PROBABILITY, pretraining_config):
    # Create MLM model
    model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL_NAME)
    model = ini_extended_model(model, base_model, BASE_MODEL_NAME, DEVICE, link_instead_of_copy_base_model=True)

    # Create data collator for training
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm_probability=PRETRAINING_MLM_PROBABILITY)
    
    # Perform further pretraining
    trainer = get_trainer(model, pretraining_config, dataset, None, None, data_collator=data_collator)
    trainer.train()

    # Move base_model to CPU to free GPU memory
    base_model = base_model.cpu()
    
    # Clean memory
    del model # Remove header from MaskedLM
    del dataset
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    return base_model

#endregion

#region ################### Finetuning ###################
def finetuning(base_model, BASE_MODEL_NAME, DEVICE, train_dataset, eval_datasets_dict, finetuning_config, RESULTS_PATH):
    # Create classifier
    num_labels = len(train_dataset.name_to_label)
    tri_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME, num_labels=num_labels)

    # Initialize model
    tri_model = ini_extended_model(tri_model, base_model, BASE_MODEL_NAME, DEVICE, link_instead_of_copy_base_model=False)

    # Create trainer and train
    trainer = get_trainer(tri_model, finetuning_config, train_dataset, eval_datasets_dict, RESULTS_PATH)
    results = trainer.train()

    # Clean memory
    gc.collect()
    #torch.cuda.empty_cache()

    return tri_model, results, trainer

#endregion

#region ################### Common ###################

#region ######### Datasets #########
class TextDataset(Dataset):
    def __init__(self, df, tokenizer, name_to_label, return_labels, sliding_window_config, tokenization_block_size):
        # Dataframe must have two columns: name and text
        assert len(df.columns) == 2
        self.df = df

        # Set general attributes
        self.tokenizer = tokenizer
        self.name_to_label = name_to_label
        self.return_labels = return_labels

        # Set sliding window
        self.sliding_window_config = sliding_window_config
        try:
            sw_elems = [int(x) for x in sliding_window_config.split("-")]
            self.sliding_window_length = sw_elems[0]
            self.sliding_window_overlap = sw_elems[1]
            self.use_sliding_window = True
        except:
            self.use_sliding_window = False # If no sliding window (e.g., "No"), use sentence splitting

        if self.use_sliding_window and self.sliding_window_length > self.tokenizer.model_max_length:
            logging.exception(f"Sliding window length ({self.sliding_window_length}) must be lower than the maximum sequence length ({self.tokenizer.model_max_length})")     

        self.tokenization_block_size = tokenization_block_size

        # Compute inputs and labels
        self.generate()

    def generate(self, gc_freq=5):
        texts_column = list(self.df[self.df.columns[1]])
        names_column = list(self.df[self.df.columns[0]])
        labels_idxs = list(map(lambda x: self.name_to_label[x], names_column))   # Compute labels, translated to the identity index
        
        # Sliding window
        if self.use_sliding_window:
            texts = texts_column            
            labels = labels_idxs
        # Sentence splitting
        else:
            texts = []
            labels = []

            # Load spacy model for sentence splitting
            # Create spaCy model. Compontents = tok2vec, tagger, parser, senter, attribute_ruler, lemmatizer, ner
            # disable = ["tok2vec", "tagger", "attribute_ruler", "lemmatizer", "ner"]) # Required components: "senter" and "parser"
            spacy_nlp = en_core_web_lg.load(disable = ["tok2vec", "tagger", "parser", "senter", "attribute_ruler", "lemmatizer", "ner"])
            spacy_nlp.add_pipe('sentencizer')

            # Get texts and labels per sentence
            for idx, (text, label) in tqdm(enumerate(zip(texts_column, labels_idxs)), total=len(texts_column),
                                                    desc="Processing sentence splitting"):
                for paragraph in text.split("\n"):
                    if len(paragraph.strip()) > 0:
                        doc = spacy_nlp(paragraph)
                        for sentence in doc.sents:
                            # Parse sentence to text
                            sentence_txt = ""
                            for token in sentence:
                                sentence_txt += " " + token.text
                            sentence_txt = sentence_txt[1:] # Remove initial space
                            # Ensure length is less than the maximum
                            sent_token_count = len(self.tokenizer.encode(sentence_txt, add_special_tokens=True))
                            if sent_token_count > self.tokenizer.model_max_length:
                                logging.exception(f"ERROR: Sentence with length {sent_token_count} > {self.tokenizer.model_max_length} at index {idx} with label {label} not included because is too long | {sentence_txt}")
                            else:
                                # Store sample
                                texts.append(sentence_txt)
                                labels.append(label)
                    
                        # Delete document for reducing memory consumption
                        del doc
                    
                # Periodically force GarbageCollector for reducing memory consumption
                if idx % gc_freq == 0:
                    gc.collect()
                
        # Tokenize texts
        self.inputs, self.labels = self.tokenize_data(texts, labels)        

    def tokenize_data(self, texts, labels):
        # Sliding window
        if self.use_sliding_window:
            input_length = self.sliding_window_length
            padding_strategy = "longest"
        # Sentence splitting
        else:
            input_length = self.tokenizer.model_max_length            
            padding_strategy = "max_length"

        all_input_ids = torch.zeros((0, input_length), dtype=torch.int)
        all_attention_masks = torch.zeros((0, input_length), dtype=torch.int)
        all_labels = []

        # For each block of data
        with tqdm(total=len(texts)) as pbar:
            for ini in range(0, len(texts), self.tokenization_block_size):
                end = min(ini+self.tokenization_block_size, len(texts))
                pbar.set_description("Tokenizing (progress bar frozen)")
                block_inputs = self.tokenizer(texts[ini:end],
                                            add_special_tokens=not self.use_sliding_window,
                                            padding=padding_strategy,  # Warning: If an text is longer than tokenizer.model_max_length, an error will raise on prediction
                                            truncation=False,
                                            max_length=self.tokenizer.model_max_length,
                                            return_tensors="pt")
                
                # Force GarbageCollector after tokenization
                gc.collect()

                # Sliding window
                if self.use_sliding_window:                    
                    all_input_ids, all_attention_masks, all_labels = self.do_sliding_window(labels[ini:end], input_length, all_input_ids, all_attention_masks, all_labels, pbar, block_inputs)
                # Sentence splitting
                else:
                    # Concatenate to all data            
                    all_input_ids = torch.cat((all_input_ids, block_inputs["input_ids"]))
                    all_attention_masks = torch.cat((all_attention_masks, block_inputs["attention_mask"]))
                    all_labels = labels
                    pbar.update(len(block_inputs))

        # Get inputs
        inputs = {"input_ids": all_input_ids, "attention_mask": all_attention_masks}

        # Transform labels to tensor
        labels = torch.tensor(all_labels)

        return inputs, labels

    def do_sliding_window(self, block_labels, input_length, all_input_ids, all_attention_masks, all_labels, pbar, block_inputs):
        # Predict number of windows
        n_windows = 0
        old_seq_length = block_inputs["input_ids"].size()[1]
        window_increment = self.sliding_window_length - self.sliding_window_overlap - 2 # Minus 2 because of the CLS and SEP tokens
        for old_attention_mask in block_inputs["attention_mask"]:
            is_sequence_finished = False
            is_padding_required = False
            ini = 0
            end = ini + self.sliding_window_length - 2
            while not is_sequence_finished:
                # Get the corresponding window's ids and mask
                if end > old_seq_length:
                    end = old_seq_length
                    is_padding_required = True
                window_mask = old_attention_mask[ini:end]
                            
                # Check end of sequence
                is_sequence_finished = end == old_seq_length or is_padding_required or window_mask[-1] == 0

                # Increment indexes
                ini += window_increment
                end = ini + self.sliding_window_length - 2 # Minus 2 because of the CLS and SEP tokens

                n_windows += 1
                    
        # Allocate memory for ids and masks
        all_sequences_windows_ids = torch.empty((n_windows, input_length), dtype=torch.int)
        all_sequences_windows_masks = torch.empty((n_windows, input_length), dtype=torch.int)                                   

        # Sliding window for block sequences' splitting
        window_idx = 0
        old_seq_length = block_inputs["input_ids"].size()[1]
        pbar.set_description("Processing sliding window")
        for block_seq_idx, (old_input_ids, old_attention_mask) in enumerate(zip(block_inputs["input_ids"], block_inputs["attention_mask"])):
            ini = 0
            end = ini + self.sliding_window_length - 2 # Minus 2 because of the CLS and SEP tokens
            is_sequence_finished = False
            is_padding_required = False
            n_windows_in_seq = 0
            while not is_sequence_finished:
                # Get the corresponding window's ids and mask
                if end > old_seq_length:
                    end = old_seq_length
                    is_padding_required = True
                window_ids = old_input_ids[ini:end]
                window_mask = old_attention_mask[ini:end]

                # Check end of sequence
                is_sequence_finished = end == old_seq_length or is_padding_required or window_mask[-1] == 0

                # Add CLS and SEP tokens
                num_attention_tokens = torch.count_nonzero(window_mask)
                if num_attention_tokens == window_mask.size()[0]:  # If window is full
                    window_ids = torch.cat(( torch.tensor([self.tokenizer.cls_token_id]), window_ids, torch.tensor([self.tokenizer.sep_token_id]) ))
                    window_mask = torch.cat(( torch.tensor([1]), window_mask, torch.tensor([1]) )) # Attention to CLS and SEP
                else: # If window has empty space (to be padded later)
                    window_ids[num_attention_tokens] = torch.tensor(self.tokenizer.sep_token_id) # SEP at last position
                    window_mask[num_attention_tokens] = 1 # Attention to SEP
                    window_ids = torch.cat(( torch.tensor([self.tokenizer.cls_token_id]), window_ids, torch.tensor([self.tokenizer.pad_token_id]) )) # PAD at the end of sentence
                    window_mask = torch.cat(( torch.tensor([1]), window_mask, torch.tensor([0]) )) # No attention to PAD

                # Padding if it is required
                if is_padding_required:
                    padding_length = self.sliding_window_length - window_ids.size()[0]
                    padding = torch.zeros((padding_length), dtype=window_ids.dtype)
                    window_ids = torch.cat((window_ids, padding))
                    window_mask = torch.cat((window_mask, padding))

                # Store ids and mask
                all_sequences_windows_ids[window_idx] = window_ids
                all_sequences_windows_masks[window_idx] = window_mask

                # Increment indexes
                ini += self.sliding_window_length - self.sliding_window_overlap - 2 # Minus 2 because of the CLS and SEP tokens
                end = ini + self.sliding_window_length - 2 # Minus 2 because of the CLS and SEP tokens
                n_windows_in_seq += 1
                window_idx += 1
                        
            # Stack lists and concatenate with new data
            all_labels += [block_labels[block_seq_idx]] * n_windows_in_seq
            pbar.update(1)
                    
        # Concat the block data        
        all_input_ids = torch.cat((all_input_ids, all_sequences_windows_ids))
        all_attention_masks = torch.cat((all_attention_masks, all_sequences_windows_masks))

        # Force GarbageCollector after sliding window
        gc.collect()

        return all_input_ids, all_attention_masks, all_labels
    
    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, index):
        # Get each value (tokens, attention...) of the item
        input = {key: value[index] for key, value in self.inputs.items()}

        # Get label if is required
        if self.return_labels:
            label = self.labels[index]
            input["labels"] = label
        
        return input

def create_datasets(train_df, eval_dfs, tokenizer, name_to_label, task_config, TOKENIZATION_BLOCK_SIZE):
    train_dataset = TextDataset(train_df, tokenizer, name_to_label, task_config.uses_labels, task_config.sliding_window, TOKENIZATION_BLOCK_SIZE)
    eval_datasets_dict = {name:TextDataset(eval_df, tokenizer, name_to_label, task_config.uses_labels, task_config.sliding_window, TOKENIZATION_BLOCK_SIZE) for name, eval_df in eval_dfs.items()}
    return train_dataset, eval_datasets_dict

#endregion

#region ######### Model initialization #########
def ini_extended_model(extended_model, base_model, base_model_name, device, link_instead_of_copy_base_model=True):
    # Link: Use base_model in extended model
    if link_instead_of_copy_base_model:
        if "distilbert" in base_model_name:
            old_base_model = extended_model.distilbert
            extended_model.distilbert = base_model
        elif "roberta" in base_model_name:
            old_base_model = extended_model.roberta
            extended_model.roberta = base_model
        elif "bert" in base_model_name:
            old_base_model = extended_model.bert
            extended_model.bert = base_model
        else:
            logging.exception(f"Not code available for base model [{base_model_name}]")
        
        # Remove old base model for memory saving
        del old_base_model
        gc.collect()

    # Copy: Clone the weights of base_model into extended model
    else:
        if "distilbert" in base_model_name:
            extended_model.distilbert.load_state_dict(base_model.state_dict())
        elif "roberta" in base_model_name:
            base_model_dict = base_model.state_dict()
            base_model_dict = dict(base_model_dict) # Copy
            base_model_dict.pop("pooler.dense.weight")  # Specific for transformers version 4.20.1
            base_model_dict.pop("pooler.dense.bias")
            extended_model.roberta.load_state_dict(base_model_dict)
        elif "bert" in base_model_name:
            extended_model.bert.load_state_dict(base_model.state_dict())
        else:
            logging.exception(f"No code available for base model [{base_model_name}]")

    # Model to device, and show size
    extended_model.to(device)
    logging.info(f"Extended model size = {sum([np.prod(p.size()) for p in extended_model.parameters()])}")

    return extended_model

#endregion

#region ######### Trainer #########
class MyTrainer(Trainer):
    def __init__(self, results_filepath:str = None, **kwargs):
        self.results_filepath = results_filepath
        self.eval_datasets_dict = kwargs["eval_dataset"]        
        self.do_custom_eval = results_filepath is not None and type(self.eval_datasets_dict) is dict
        if self.do_custom_eval:
            kwargs["eval_dataset"] = None # Substitue for avoiding bug from https://github.com/huggingface/transformers/pull/19158#issuecomment-1429486221
        
        Trainer.__init__(self, **kwargs)
        
        if self.do_custom_eval:
            self.all_results = []
            self.evaluation_epoch = 1   # Start epoch counter
            self.initialize_results_file()
    
    def current_time_str(self):
        return datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        
    def initialize_results_file(self):
        text = f"{self.current_time_str()}\n"
        text += "Time,Epoch"
        for dataset_name in self.eval_datasets_dict.keys():
            text+=f",{dataset_name}"
        text += "\n"
        self.write_results(text)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # If custom evaluation
        if self.do_custom_eval:
            custom_results = {}
            avg_loss = 0
            loss_key = f"{metric_key_prefix}_loss"
            avg_acc = 0
            acc_key = f"{metric_key_prefix}_Accuracy"

            # Get results
            for dataset_name, dataset in self.eval_datasets_dict.items():       
                res = Trainer.evaluate(self, eval_dataset=dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
                avg_loss += res[loss_key] / len(self.eval_datasets_dict)
                avg_acc += res[acc_key] / len(self.eval_datasets_dict)
                custom_results[dataset_name] = res
            
            # Save results intro list and file
            self.store_results(custom_results)
            self.all_results.append(custom_results)

            # Increment evaluation epoch
            self.evaluation_epoch += 1

            return {loss_key: avg_loss, acc_key: avg_acc}
        # Otherwise, standard evaluation with eval_dataset
        else:
            return Trainer.evaluate(self, eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)        
    
    def store_results(self, eval_results:dict):
        current_time = self.current_time_str()
        try:
            results_text = f"{current_time},{self.evaluation_epoch}"
            for data in eval_results.values():
                key = list(filter(lambda k: "_Accuracy" in k, data.keys()))[0]
                accuracy = data[key]
                accuracy_str = "{:.3f}".format(accuracy)
                results_text += f",{accuracy_str}"
            results_text += "\n"
            self.write_results(results_text)
        except Exception as e:
            self.write_results(f"{current_time}, Error writing the results of epoch {self.evaluation_epoch} ({e})")
            logging.info(f"ERROR writing the results: {e}")

    def write_results(self, text:str):
        with open(self.results_filepath, "a+") as f:
            f.write(text)

def compute_metrics(results):
    logits, labels = results

    # Get predictions sum
    logits = torch.from_numpy(logits)    
    logits_dict = {}
    for logit, label in zip(logits, labels):
        current_logits = logits_dict.get(label, torch.zeros_like(logit))
        logits_dict[label] = current_logits.add_(logit)
    
    # Cumpute final predictions
    num_preds = len(logits_dict)
    all_preds = torch.zeros(num_preds, device="cpu")
    all_labels = torch.zeros(num_preds, device="cpu")
    for idx, item in enumerate(logits_dict.items()):
        label, logits = item
        all_labels[idx] = label
        probs = F.softmax(logits, dim=-1)
        all_preds[idx] = torch.argmax(probs)
    
    correct_preds = torch.sum(all_preds == all_labels)
    accuracy = (float(correct_preds)/num_preds)*100
    return {"Accuracy": accuracy}

def get_trainer(model, model_config, train_dataset, eval_datasets_dict, results_filepath, data_collator=None):
    is_for_mlm = model_config.is_for_mlm

    # Variable settings
    evaluation_strategy = "no" if is_for_mlm else "epoch"
    save_strategy = "no" if is_for_mlm else "epoch"
    load_best_model_at_end = not is_for_mlm
    eval_datasets_dict = None if is_for_mlm else eval_datasets_dict
    results_filepath = None if is_for_mlm else results_filepath

    # Define TrainingArguments    
    args = TrainingArguments(
        output_dir=model_config.trainer_folder_path,
        overwrite_output_dir=True,        
        load_best_model_at_end=load_best_model_at_end,
        save_strategy=save_strategy,
        save_total_limit=1,
        num_train_epochs=model_config.epochs,
        per_device_train_batch_size=model_config.batch_size,
        per_device_eval_batch_size=model_config.batch_size,        
        logging_strategy="epoch",
        logging_steps=500,        
        evaluation_strategy=evaluation_strategy,
        log_level="error",
        disable_tqdm=False,
        eval_accumulation_steps=5,  # Number of eval steps before move preds from GPU to RAM        
        dataloader_num_workers=0,
        metric_for_best_model="eval_Accuracy",
        dataloader_persistent_workers=False,
        dataloader_prefetch_factor=None,
    )

    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=model_config.learning_rate, betas=(0.9, 0.999), eps=1e-06, weight_decay=0.0)
    scheduler = get_constant_schedule(optimizer)

    # Use Accelerate
    accelerator = Accelerator()
    (model, optimizer, scheduler, train_dataset) = accelerator.prepare(model, optimizer, scheduler, train_dataset)

    # Define trainer    
    trainer = MyTrainer(results_filepath,
                        model=model,
                        args=args,
                        train_dataset=train_dataset,
                        eval_dataset=eval_datasets_dict,
                        optimizers=[optimizer, scheduler],
                        compute_metrics=compute_metrics,
                        data_collator=data_collator
                    )
    
    return trainer

#endregion

#endregion

#endregion


#region ###################################### Predict TRIR ######################################
def predict_TRIR(trainer):
    trainer.evaluate()
    
    # Show results from the last (just already done) evaluate
    results = trainer.all_results[-1]
    
    # Show results
    for dataset_name, res in trainer.all_results[-1].items():
        logging.info(f"TRIR {dataset_name} = {res['eval_Accuracy']}%")
    
    return results

#endregion 
