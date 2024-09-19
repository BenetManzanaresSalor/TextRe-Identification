#region ###################################### Imports ######################################
import sys
import os
import json
import gc
import re
from datetime import datetime
import logging
from collections import OrderedDict

from argparse import Namespace
from tqdm.autonotebook import tqdm
import numpy as np
import pandas as pd
from io import StringIO
import en_core_web_lg # This model is leveraged for every spaCy usage (https://spacy.io/models/en#en_core_web_lg)

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel
from transformers import Trainer, TrainingArguments, get_constant_schedule
from transformers import AutoModelForMaskedLM, AutoModelForSequenceClassification
from transformers import DataCollatorForLanguageModeling, pipeline, Pipeline
from accelerate import Accelerator

import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO) # Configure logging

#endregion

#region ###################################### Configuration file argument ######################################

#region ################### Arguments parsing ###################
def argument_parsing():
    if (args_count := len(sys.argv)) > 2:
        logging.exception(Exception(f"One argument expected, got {args_count - 1}"))
    elif args_count < 2:
        logging.exception(Exception("You must specify the JSON configuration filepath as first argument"))

    target_dir = sys.argv[1]
    return target_dir

#endregion

#region ################### JSON file loading ###################
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

#region ###################################### TRI class ######################################

class TRI():
    #region ################### Properties ###################

    #region ########## Mandatory configs ##########

    mandatory_configs_names = ["output_folder_path", "data_file_path",
        "individual_name_column", "background_knowledge_column"]
    output_folder_path = None
    data_file_path = None
    individual_name_column = None
    background_knowledge_column = None

    #endregion

    #region ########## Optional configs with default values ##########

    optional_configs_names = ["load_saved_pretreatment", "add_non_saved_anonymizations",
        "anonymize_background_knowledge", "only_use_anonymized_background_knowledge", 
        "use_document_curation", "save_pretreatment", "load_saved_finetuning", "base_model_name", 
        "tokenization_block_size", "use_additional_pretraining", "save_additional_pretraining",
        "load_saved_pretraining", "pretraining_epochs", "pretraining_batch_size",
        "pretraining_learning_rate", "pretraining_mlm_probability", "pretraining_sliding_window",
        "save_finetuning", "load_saved_finetuning", "finetuning_epochs", "finetuning_batch_size",
        "finetuning_learning_rate", "finetuning_sliding_window", "dev_set_column_name"]
    load_saved_pretreatment = True
    add_non_saved_anonymizations = True
    anonymize_background_knowledge = True
    only_use_anonymized_background_knowledge = True
    use_document_curation = True
    save_pretreatment = True
    base_model_name = "distilbert-base-uncased"
    tokenization_block_size = 250
    use_additional_pretraining = True
    save_additional_pretraining = True
    load_saved_pretraining = True
    pretraining_epochs = 3
    pretraining_batch_size = 8
    pretraining_learning_rate = 5e-05
    pretraining_mlm_probability = 0.15
    pretraining_sliding_window = "512-128"
    save_finetuning = True
    load_saved_finetuning = True
    finetuning_epochs = 15
    finetuning_batch_size = 16
    finetuning_learning_rate = 5e-05
    finetuning_sliding_window = "100-25"
    dev_set_column_name = False

    #endregion

    #region ########## Derived configs ##########

    pretreated_data_path:str = None
    pretrained_model_path:str = None
    results_path:str = None
    tri_pipe_path:str = None
    pretraining_config = Namespace()
    finetuning_config = Namespace()

    #endregion

    #region ########## Functional properties ##########

    # Data
    data_df:pd.DataFrame = None
    train_df:pd.DataFrame = None
    eval_dfs:dict = None
    train_individuals:set = None
    eval_individuals:set = None
    all_individuals:set = None
    no_train_individuals:set = None
    no_eval_individuals:set = None
    label_to_name:dict = None
    name_to_label:dict = None
    spacy_nlp = None
    pretreated_data_loaded:bool = None

    # Build classifier
    device = None
    tri_model = None
    tokenizer = None
    pretraining_dataset:Dataset = None # Removed after usage
    finetuning_dataset:Dataset = None      
    finetuning_trainer:Trainer = None
    eval_datasets_dict:dict = None
    pipe:Pipeline = None

    # Predict
    trir_results:dict = None

    #endregion

    #endregion

    #region ################### Constructor and configurations ###################

    def __init__(self, **kwargs):
        self.set_configs(**kwargs, are_mandatory_configs_required=True)

    def set_configs(self, are_mandatory_configs_required=False, **kwargs):
        arguments = kwargs.copy()

        # Mandatory configs
        for setting_name in self.mandatory_configs_names:
            value = arguments.get(setting_name, None)
            if isinstance(value, str):
                self.__dict__[setting_name] = arguments[setting_name]
                del arguments[setting_name]
            elif are_mandatory_configs_required:
                raise AttributeError(f"Mandatory argument {setting_name} is not defined or it is not a string")
        
        # Store remaining optional configs
        for (opt_setting_name, opt_setting_value) in arguments.items():
            if opt_setting_name in self.optional_configs_names:                
                if isinstance(opt_setting_value, str) or isinstance(opt_setting_value, int) or \
                isinstance(opt_setting_value, float) or isinstance(opt_setting_value, bool):
                    self.__dict__[opt_setting_name] = opt_setting_value
                else:
                    raise AttributeError(f"Optional argument {opt_setting_name} is not a string, integer, float or boolean.")
            else:
                logging.warning(f"Unrecognized setting name {opt_setting_name}")

        # Generate derived configs
        self.pretreated_data_path = os.path.join(self.output_folder_path, "Pretreated_Data.json")
        self.pretrained_model_path = os.path.join(self.output_folder_path, "Pretrained_Model.pt")
        self.results_file_path = os.path.join(self.output_folder_path, "Results.csv")
        self.tri_pipe_path = os.path.join(self.output_folder_path, "TRI_Pipeline")

        self.pretraining_config.is_for_mlm = True
        self.pretraining_config.uses_labels = False
        self.pretraining_config.epochs = self.pretraining_epochs
        self.pretraining_config.batch_size = self.pretraining_batch_size
        self.pretraining_config.learning_rate = self.pretraining_learning_rate
        self.pretraining_config.sliding_window = self.pretraining_sliding_window
        self.pretraining_config.trainer_folder_path = os.path.join(self.output_folder_path, f"Pretraining")

        self.finetuning_config.is_for_mlm = False
        self.finetuning_config.uses_labels = True
        self.finetuning_config.epochs = self.finetuning_epochs
        self.finetuning_config.batch_size = self.finetuning_batch_size
        self.finetuning_config.learning_rate = self.finetuning_learning_rate
        self.finetuning_config.sliding_window = self.finetuning_sliding_window
        self.finetuning_config.trainer_folder_path = os.path.join(self.output_folder_path, f"Finetuning")

        # Check for GPU with CUDA
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        else:
            self.device = torch.device("cpu")

    #endregion

    #region ################### Run all blocks ###################
    
    def run(self, verbose=True):
        self.run_data(verbose=verbose)
        self.run_build_classifier(verbose=verbose)
        results = self.run_predict_trir(verbose=verbose)
        return results

    #endregion

    #region ################### Data ###################

    def run_data(self, verbose=True):
        if verbose: logging.info("######### START: DATA #########")
        self.pretreated_data_loaded = False
        self.pretreatment_done = False

        # Create output directory if it does not exist
        if not os.path.isdir(self.output_folder_path):
            os.makedirs(self.output_folder_path, exist_ok=True)

        # Read pretreated data if it exists        
        if self.load_saved_pretreatment and os.path.isfile(self.pretreated_data_path):
            if verbose: logging.info("######### START: LOADING SAVED PRETREATED DATA #########")
            self.train_df, self.eval_dfs = self.load_pretreatment()            
            self.pretreated_data_loaded = True

            # If curate non-saved anonymizations and document curation are required
            if self.add_non_saved_anonymizations:
                # Pretreated saved anonymizations
                self.saved_anons = set(self.eval_dfs.keys())

                # Non-pretreated anonymizations from raw file
                new_data_df = self.read_data()
                _, new_eval_dfs = self.split_data(new_data_df)
                self.non_pretreated_anons = set(new_eval_dfs.keys())

                # Find non-pretreated anonymizations not present in pretreated saved anonymizations
                self.non_saved_anons = []
                for anon_name in self.non_pretreated_anons:
                    if not anon_name in self.saved_anons:
                        self.non_saved_anons.append(anon_name)

                # If there are non-pretreated anonymizations not present in saved anonymizations, add them
                if len(self.non_saved_anons) > 0:
                    if verbose: logging.info("######### START: ADDING NON-SAVED ANONYMIZATIONS #########")
                    if verbose: logging.info(f"The following non-saved anonymizations will be added: {self.non_saved_anons}")
                    for anon_name in self.non_saved_anons:
                        # Curate anonymizations if needed
                        if self.use_document_curation:
                            self.curate_df(new_eval_dfs[anon_name], self.load_spacy_nlp())
                        # Add to eval_dfs
                        self.eval_dfs[anon_name] = new_eval_dfs[anon_name]
                    self.pretreatment_done = True
                    if verbose: logging.info("Non-saved anonymizations added")
                    if verbose: logging.info("######### END: ADDING NON-SAVED ANONYMIZATIONS #########")
                else:
                    if verbose: logging.info("There are not non-saved anonymizations to add")
                    if verbose: logging.info("######### SKIPPING: ADDING NON-SAVED ANONYMIZATIONS #########")
            else:
                if verbose: logging.info("######### SKIPPING: ADDING NON-SAVED ANONYMIZATIONS #########")

            if verbose: logging.info("######### END: LOADING SAVED PRETREATED DATA #########")

        # Otherwise, read raw data
        else:
            if self.load_saved_pretreatment:
                if verbose: logging.info(f"Impossible to load saved pretreated data, file {self.pretreated_data_path} not found.")

            if verbose: logging.info("######### START: READ RAW DATA FROM FILE #########")

            if verbose: logging.info("Reading data")
            self.data_df = self.read_data()
            if verbose: logging.info("Data reading complete")

            # Split into train and evaluation (dropping rows where no documents are available)
            if verbose: logging.info("Splitting into train (background knowledge) and evaluation (anonymized) sets")
            self.train_df, self.eval_dfs = self.split_data(self.data_df)
            del self.data_df # Remove general dataframe for saving memory
            if verbose: logging.info("Train and evaluation splitting complete")
            
            if verbose: logging.info("######### END: READ RAW DATA FROM FILE #########")

        if verbose: logging.info("######### START: DATA STATISTICS #########")

        # Get individuals found in each set
        res = self.get_individuals(self.train_df, self.eval_dfs)
        self.train_individuals, self.eval_individuals, self.all_individuals, self.no_train_individuals, self.no_eval_individuals = res

        # Generat Label->Name and Name->Label dictionaries
        self.label_to_name, self.name_to_label, self.num_labels = self.get_individuals_labels(self.all_individuals)

        # Show relevant information
        if verbose:
            self.show_data_stats(self.train_df, self.eval_dfs, self.no_eval_individuals, self.no_train_individuals, self.eval_individuals)        

        if verbose: logging.info("######### END: DATA STATISTICS #########")

        # Pretreat data if required and not pretreatment loaded
        if (self.anonymize_background_knowledge or self.use_document_curation) and not self.pretreated_data_loaded:
            if verbose: logging.info("######### START: DATA PRETREATMENT #########")
            
            if self.anonymize_background_knowledge:
                if verbose: logging.info("######### START: BACKGROUND KNOWLEDGE ANONYMIZATION #########")        
                self.train_df = self.anonymize_bk(self.train_df)
                if verbose: logging.info("######### END: BACKGROUND KNOWLEDGE ANONYMIZATION #########")
            else:
                if verbose: logging.info("######### SKIPPING: BACKGROUND KNOWLEDGE ANONYMIZATION #########")

            if self.use_document_curation:
                if verbose: logging.info("######### START: DOCUMENT CURATION #########")
                self.document_curation(self.train_df, self.eval_dfs)
                if verbose: logging.info("######### END: DOCUMENT CURATION #########")
            else:
                if verbose: logging.info("######### SKIPPING: DOCUMENT CURATION #########")            

            self.pretreatment_done = True

            if verbose: logging.info("######### END: DATA PRETREATMENT #########")
        else:
            if verbose: logging.info("######### SKIPPING: DATA PRETREATMENT #########")

        # If save pretreatment is required and there is any pretreatment modification to save
        if self.save_pretreatment and self.pretreatment_done:
            if verbose: logging.info("######### START: SAVE PRETREATMENT #########")
            self.save_pretreatment_dfs(self.train_df, self.eval_dfs)
            if verbose: logging.info("######### END: SAVE PRETREATMENT #########")
        else:
            if verbose: logging.info("######### SKIPPING: SAVE PRETREATMENT #########")
        
        if verbose: logging.info("######### END: DATA #########")

    #region ########## Load pretreatment ##########

    def load_pretreatment(self):
        with open(self.pretreated_data_path, "r") as f:
            (train_df_json_str, eval_dfs_jsons) = json.load(f)        
        
        train_df = pd.read_json(StringIO(train_df_json_str))
        eval_dfs = OrderedDict([(name, pd.read_json(StringIO(df_json))) for name, df_json in eval_dfs_jsons.items()])

        return train_df, eval_dfs
    
    #endregion

    #region ########## Data reading ##########

    def read_data(self) -> pd.DataFrame:
        if self.data_file_path.endswith(".json"):
            data_df = pd.read_json(self.data_file_path)
        elif self.data_file_path.endswith(".csv"):
            data_df = pd.read_csv(self.data_file_path)
        else:
            raise Exception(f"Unrecognized file extension for data file [{self.data_file_path}]. Compatible formats are JSON and CSV.")
        
        # Check required columns exist
        if not self.individual_name_column in data_df.columns:
            raise Exception(f"Dataframe does not contain the individual name column {self.individual_name_column}")
        if not self.background_knowledge_column in data_df.columns:
            raise Exception(f"Dataframe does not contain the background knowledge column {self.background_knowledge_column}")
        if self.dev_set_column_name is not False and not self.dev_set_column_name in data_df.columns:
            raise Exception(f"Dataframe does not contain the dev set column {self.dev_set_column_name}")
        
        # Check there are additional columns providing texts to re-identify
        anon_cols = [col_name for col_name in data_df.columns if not col_name in [self.individual_name_column, self.background_knowledge_column]]        
        if len(anon_cols) == 0:
            raise Exception(f"Dataframe does not contain columns with texts to re-identify, only individual name and background knowledge columns")
        
        # Sort by individual name
        data_df.sort_values(self.individual_name_column).reset_index(drop=True, inplace=True)

        return data_df

    def split_data(self, data_df:pd.DataFrame):
        data_df.replace('', np.nan, inplace=True)   # Replace empty texts by NaN

        # Training data formed by labeled background knowledge
        train_cols = [self.individual_name_column, self.background_knowledge_column]
        train_df = data_df[train_cols].dropna().reset_index(drop=True)

        # Evaluation data formed by texts to re-identify
        eval_columns = [col for col in data_df.columns if col not in train_cols]
        eval_dfs = {col:data_df[[self.individual_name_column, col]].dropna().reset_index(drop=True) for col in eval_columns}

        return train_df, eval_dfs

    #endregion

    #region ########## Data statistics ##########

    def get_individuals(self, train_df:pd.DataFrame, eval_dfs:dict):
        train_individuals = set(train_df[self.individual_name_column])
        eval_individuals = set()
        for name, eval_df in eval_dfs.items():
            if name != self.dev_set_column_name: # Exclude dev_set from these statistics
                eval_individuals.update(set(eval_df[self.individual_name_column]))
        all_individuals = train_individuals.union(eval_individuals)
        no_train_individuals = eval_individuals - train_individuals
        no_eval_individuals = train_individuals - eval_individuals

        return train_individuals, eval_individuals, all_individuals, no_train_individuals, no_eval_individuals

    def get_individuals_labels(self, all_individuals:set):
        sorted_indvidiuals = sorted(list(all_individuals)) # Sort individuals for ensuring same order every time (required for automatic loading)
        label_to_name = {idx:name for idx, name in enumerate(sorted_indvidiuals)}
        name_to_label = {name:idx for idx, name in label_to_name.items()}
        num_labels = len(name_to_label)

        return label_to_name, name_to_label, num_labels

    def show_data_stats(self, train_df:pd.DataFrame, eval_dfs:dict, no_eval_individuals:set, no_train_individuals:set, eval_individuals:set):
        logging.info(f"Number of background knowledge documents for training: {len(train_df)}")

        eval_n_dict = {name:len(df) for name, df in eval_dfs.items()}
        logging.info(f"Number of protected documents for evaluation: {eval_n_dict}")

        if len(no_eval_individuals) > 0:
            logging.info(f"No protected documents found for {len(no_eval_individuals)} individuals.")
        
        if len(no_train_individuals) > 0:
            max_risk = (1 - len(no_train_individuals) / len(eval_individuals)) * 100
            logging.info(f"No background knowledge documents found for {len(no_train_individuals)} individuals. Re-identification risk limited to {max_risk:.3f}% (excluding dev set).")

    #endregion

    #region ########## Data pretreatment ##########

    def load_spacy_nlp(self):
        # Load if it is not already loaded
        if self.spacy_nlp is None:
            self.spacy_nlp = en_core_web_lg.load()
        return self.spacy_nlp

    #region ##### Anonymize background knowledge #####
    
    def anonymize_bk(self, train_df:pd.DataFrame) -> pd.DataFrame:
        # Perform anonymization
        spacy_nlp = self.load_spacy_nlp()        
        train_anon_df = self.anonymize_df(train_df, spacy_nlp)

        if self.only_use_anonymized_background_knowledge:
            train_df = train_anon_df # Overwrite train dataframe with the anonymized version
        else:
            train_df = pd.concat([train_df, train_anon_df], ignore_index=True, copy=False) # Concatenate to train dataframe

        return train_df

    def anonymize_df(self, df, spacy_nlp, gc_freq=5) -> pd.DataFrame:
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

    #region ##### Document curation #####

    def document_curation(self, train_df:pd.DataFrame, eval_dfs:dict):
        spacy_nlp = self.load_spacy_nlp()

        # Perform preprocessing for both training and evaluation
        self.curate_df(train_df, spacy_nlp)
        for eval_df in eval_dfs.values():
            self.curate_df(eval_df, spacy_nlp)

    def curate_df(self, df, spacy_nlp, gc_freq=5):
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

    #region ##### Save pretreatment #####

    def save_pretreatment_dfs(self, train_df:pd.DataFrame, eval_dfs:dict):
        with open(self.pretreated_data_path, "w") as f:
            f.write(json.dumps((train_df.to_json(orient="records"),
                                {name:df.to_json(orient="records") for name, df in eval_dfs.items()})))        

    #endregion

    #endregion

    #endregion

    #region ################### Build classifier ###################
    # Implementation grounded on HuggingFace's Transformers (https://huggingface.co/docs/transformers/index)

    def run_build_classifier(self, verbose=True):
        if verbose: logging.info("######### START: BUILD CLASSIFIER #########")

        if self.load_saved_finetuning and os.path.exists(self.tri_pipe_path):
            if verbose: logging.info("######### START: LOAD ALREADY TRAINED TRI MODEL #########")

            # Get TRI classifier and tokenizer
            self.tri_model, self.tokenizer = self.load_trained_TRI_model()

            # Datasets for TRI
            res = self.create_datasets(self.train_df, self.eval_dfs, self.tokenizer, self.name_to_label, self.finetuning_config)
            self.finetuning_dataset, self.eval_datasets_dict = res

            # Create trainer for TRI
            self.finetuning_trainer = self.get_trainer(self.tri_model, self.finetuning_config,
                                                        self.finetuning_dataset, eval_datasets_dict=self.eval_datasets_dict)
            
            if verbose: logging.info("######### END: LOAD ALREADY TRAINED TRI MODEL #########")

        # Otherwise, pretrain (if required) and finetune a TRI model
        else:
            if self.load_saved_finetuning:
                if verbose: logging.info(f"Fail loading saved TRI pipeline: Folder {self.tri_pipe_path} not found.")

            if verbose: logging.info("######### START: CREATE BASE LANGUAGE MODEL #########")
            self.base_model, self.tokenizer = self.create_base_model(verbose=verbose)
            if verbose: logging.info("######### END: CREATE BASE LANGUAGE MODEL #########")

            if self.use_additional_pretraining:
                if verbose: logging.info("######### START: ADDITIONAL PRETRAINING #########")

                if self.load_saved_pretraining and os.path.exists(self.pretrained_model_path):
                    if verbose: logging.info("Loading additionally pretrained base model")
                    self.load_pretrained_base_model(self.base_model)
                    if verbose: logging.info("Additionally pretrained base model loaded")
                else:
                    if self.load_saved_pretraining:
                        if verbose: logging.info(f"Fail loading saved pretrained base model: File {self.pretrained_model_path} not found.")

                    # Datasets for additional pretraining
                    self.pretraining_dataset, _ = self.create_datasets(self.train_df, self.eval_dfs, self.tokenizer, 
                                                                       self.name_to_label, self.pretraining_config)

                    # Additionally pretrain the base language model
                    self.base_model = self.additional_pretraining(self.base_model, self.tokenizer, self.pretraining_config, 
                                                                  self.pretraining_dataset, verbose=verbose)
                    
                    if self.save_additional_pretraining:
                        if verbose: logging.info("Saving additionally pretrained base model")
                        self.save_pretrained_base_model(self.base_model)
                        if verbose: logging.info("Additionally pretrained base model saved")
                
                if verbose: logging.info("######### END: ADDITIONAL PRETRAINING #########")
            else:
                if verbose: logging.info("######### SKIPPING: ADDITIONAL PRETRAINING #########")            

            if verbose: logging.info("######### START: FINETUNING #########")

            # Datasets for finetuning
            self.finetuning_dataset, self.eval_datasets_dict = self.create_datasets(self.train_df, self.eval_dfs, 
                                                                                    self.tokenizer, self.name_to_label,
                                                                                      self.finetuning_config)

            # Finetuning for text re-identification
            self.tri_model, self.finetuning_trainer, _ = self.finetuning(self.base_model, self.num_labels,
                                                                     self.finetuning_config, self.finetuning_dataset,
                                                                     self.eval_datasets_dict, verbose=verbose)

            if self.save_finetuning:
                if verbose: logging.info("Saving finetuned TRI pipeline")
                self.pipe, self.tri_model = self.save_finetuned_tri_pipeline(self.tri_model, self.tokenizer)
                if verbose: logging.info("Finetuned TRI model pipeline saved")

            if verbose: logging.info("######### END: FINETUNING #########")

        if verbose: logging.info("######### END: BUILD CLASSIFIER #########")
    
    #region ########## Load already trained TRI model ##########

    def load_trained_TRI_model(self):
        tri_model = AutoModelForSequenceClassification.from_pretrained(self.tri_pipe_path)
        tokenizer = AutoTokenizer.from_pretrained(self.tri_pipe_path)
        return tri_model, tokenizer

    #endregion

    #region ########## Create base language model ##########

    def create_base_model(self, verbose=True):
        base_model = AutoModel.from_pretrained(self.base_model_name)
        if verbose: logging.info(f"Model size = {sum([np.prod(p.size()) for p in base_model.parameters()])}")
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        return base_model, tokenizer

    #endregion
    
    #region ########## Load additional pretraining ##########

    def load_pretrained_base_model(self, base_model):
        base_model.load_state_dict(torch.load(self.pretrained_model_path))

    #endregion  

    #region ########## Additional pretraining ##########

    def additional_pretraining(self, base_model, tokenizer, pretraining_config:Namespace, pretraining_dataset:Dataset, verbose=True):
        # Create MLM model
        pretraining_model = AutoModelForMaskedLM.from_pretrained(self.base_model_name)
        pretraining_model = self.ini_extended_model(base_model, pretraining_model, link_instead_of_copy_base_model=True, verbose=verbose)

        # Create data collator for training
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm_probability=self.pretraining_mlm_probability)
        
        # Perform further pretraining
        pretraining_trainer = self.get_trainer(pretraining_model, pretraining_config, pretraining_dataset, data_collator=data_collator)
        pretraining_trainer.train()

        # Move base_model to CPU to free GPU memory
        base_model = base_model.cpu()
        
        # Clean memory
        del pretraining_model # Remove header from MaskedLM
        del pretraining_dataset
        del pretraining_trainer
        gc.collect()
        torch.cuda.empty_cache()

        return base_model

    #endregion

    #region ########## Save additional pretraining ##########

    def save_pretrained_base_model(self, base_model):
        torch.save(base_model.state_dict(), self.pretrained_model_path)

    #endregion

    #region ########## Finetuning ##########

    def finetuning(self, base_model, num_labels, finetuning_config, finetuning_dataset, eval_datasets_dict, verbose=True):
        # Create classifier
        tri_model = AutoModelForSequenceClassification.from_pretrained(self.base_model_name, num_labels=num_labels)

        # Initialize model
        tri_model = self.ini_extended_model(base_model, tri_model, link_instead_of_copy_base_model=False, verbose=verbose)

        # Create trainer and train
        finetuning_trainer = self.get_trainer(tri_model, finetuning_config, finetuning_dataset,
                                                    eval_datasets_dict=eval_datasets_dict)
        training_results = finetuning_trainer.train()

        return tri_model, finetuning_trainer, training_results

    #endregion

    #region ########## Save finetuned TRI model ##########

    def save_finetuned_tri_pipeline(self, tri_model, tokenizer):
        pipe = pipeline("text-classification", model=tri_model, tokenizer=tokenizer)
        pipe.save_pretrained(self.tri_pipe_path)
        tri_model = tri_model.to(self.device) # Saving moves model to CPU, return it to defined DEVICE
        return pipe, tri_model

    #endregion

    #region ########## Common ##########

    def create_datasets(self, train_df, eval_dfs, tokenizer, name_to_label, task_config):
        train_dataset = TRIDataset(train_df, tokenizer, name_to_label, task_config.uses_labels, task_config.sliding_window, self.tokenization_block_size)
        eval_datasets_dict = OrderedDict([(name, TRIDataset(eval_df, tokenizer, name_to_label, task_config.uses_labels, task_config.sliding_window, self.tokenization_block_size)) for name, eval_df in eval_dfs.items()])
        return train_dataset, eval_datasets_dict

    def ini_extended_model(self, base_model, extended_model, link_instead_of_copy_base_model, verbose=True):
        # Link: Use base_model in extended model
        if link_instead_of_copy_base_model:
            if "distilbert" in self.base_model_name:
                old_base_model = extended_model.distilbert
                extended_model.distilbert = base_model
            elif "roberta" in self.base_model_name:
                old_base_model = extended_model.roberta
                extended_model.roberta = base_model
            elif "bert" in self.base_model_name:
                old_base_model = extended_model.bert
                extended_model.bert = base_model
            else:
                raise Exception(f"Not code available for base model [{self.base_model_name}]")
            
            # Remove old base model for memory saving
            del old_base_model
            gc.collect()

        # Copy: Clone the weights of base_model into extended model
        else:
            if "distilbert" in self.base_model_name:
                extended_model.distilbert.load_state_dict(base_model.state_dict())
            elif "roberta" in self.base_model_name:
                base_model_dict = base_model.state_dict()
                base_model_dict = dict(base_model_dict) # Copy
                base_model_dict.pop("pooler.dense.weight")  # Specific for transformers version 4.20.1
                base_model_dict.pop("pooler.dense.bias")
                extended_model.roberta.load_state_dict(base_model_dict)
            elif "bert" in self.base_model_name:
                extended_model.bert.load_state_dict(base_model.state_dict())
            else:
                raise Exception(f"No code available for base model [{self.base_model_name}]")

        # Model to device, and show size
        extended_model.to(self.device)
        if verbose: 
            logging.info(f"Extended model size = {sum([np.prod(p.size()) for p in extended_model.parameters()])}")

        return extended_model

    def get_trainer(self, model, task_config, train_dataset, eval_datasets_dict=None, data_collator=None):
        is_for_mlm = task_config.is_for_mlm

        # Settings for additional pretraining (Masked Language Modeling)
        if is_for_mlm:
            eval_strategy = "no"
            save_strategy = "no"
            load_best_model_at_end = False
            metric_for_best_model = None
            eval_datasets_dict = None
            results_filepath = None
        # Settings for finetuning
        else:
            eval_strategy = "epoch"
            save_strategy = "epoch"
            load_best_model_at_end = True
            if self.dev_set_column_name:
                metric_for_best_model = self.dev_set_column_name+"_eval_Accuracy" # Prefix (e.g., "eval_") will be added by the Trainer
            else:
                metric_for_best_model = "avg_Accuracy" # Prefix (e.g., "eval_") will be added later will be added by the Trainer
            eval_datasets_dict = self.eval_datasets_dict
            results_filepath = self.results_file_path

        # Define TrainingArguments
        args = TrainingArguments(
            output_dir=task_config.trainer_folder_path,
            overwrite_output_dir=True,
            load_best_model_at_end=load_best_model_at_end,
            save_strategy=save_strategy,
            save_total_limit=1,
            num_train_epochs=task_config.epochs,
            per_device_train_batch_size=task_config.batch_size,
            per_device_eval_batch_size=task_config.batch_size,
            logging_strategy="epoch",
            logging_steps=500,
            eval_strategy=eval_strategy,
            disable_tqdm=False,
            eval_accumulation_steps=5,  # Number of eval steps before move preds are moved from GPU to RAM        
            dataloader_num_workers=0,
            metric_for_best_model=metric_for_best_model,
            dataloader_persistent_workers=False,
            dataloader_prefetch_factor=None,
        )

        # Define optimizer
        optimizer = AdamW(model.parameters(), lr=task_config.learning_rate, betas=(0.9, 0.999), eps=1e-06, weight_decay=0.0)
        scheduler = get_constant_schedule(optimizer)

        # Use Accelerate
        accelerator = Accelerator()
        (model, optimizer, scheduler, train_dataset) = accelerator.prepare(model, optimizer, scheduler, train_dataset)

        # Define trainer    
        trainer = TRITrainer(results_filepath,
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

    #region ################### Predict TRIR ###################
    
    def run_predict_trir(self, verbose=True):
        if verbose: logging.info("######### START: PREDICT TRIR #########")

        # Predict
        self.finetuning_trainer.evaluate()
        
        # Show results from the last epoch (i.e., just already done evaluation)
        self.trir_results = self.finetuning_trainer.all_results[-1]
        
        # Show results
        if verbose:
            for dataset_name, res in self.trir_results.items():
                #res_key = list(filter(lambda x:x.endswith("_Accuracy"), res.keys()))[0]
                logging.info(f"TRIR {dataset_name} = {res['eval_Accuracy']}%")
        
        if verbose: logging.info("######### END: PREDICT TRIR #########")
        
        return self.trir_results

    #endregion

#endregion

#region ###################################### TRI dataset ######################################

class TRIDataset(Dataset):
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

#endregion

#region ###################################### TRI trainer ######################################

class TRITrainer(Trainer):
    def __init__(self, results_filepath:str = None, **kwargs):
        Trainer.__init__(self, **kwargs)
        self.results_filepath = results_filepath
        
        if self.results_filepath is not None and "eval_dataset" in self.__dict__ and isinstance(self.eval_dataset, dict):
            self.do_custom_eval = True
            self.eval_dataset_dict = self.eval_dataset
        else:
            self.do_custom_eval = False
        
        if self.do_custom_eval:
            self.eval_datasets_dict = self.eval_dataset
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
        text+=",Average"
        text += "\n"
        self.write_results(text)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        if self.do_custom_eval:
            metrics = OrderedDict()
            structured_results = OrderedDict()
            avg_loss = 0
            loss_key = f"{metric_key_prefix}_loss"
            avg_acc = 0
            acc_key = f"{metric_key_prefix}_Accuracy"

            # Get results
            for dataset_name, dataset in self.eval_datasets_dict.items():
                dataset_metrics = Trainer.evaluate(self, eval_dataset=dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)                
                avg_loss += dataset_metrics[loss_key] / len(self.eval_datasets_dict)
                avg_acc += dataset_metrics[acc_key] / len(self.eval_datasets_dict)
                structured_results[dataset_name] = dataset_metrics
                dataset_metrics = {f"{metric_key_prefix}_{dataset_name}_{key}":val for key, val in dataset_metrics.items()} # Add dataset name to results keys
                metrics.update(dataset_metrics)
                
            
            # Add average metrics to results
            metrics.update({f"{metric_key_prefix}_avg_loss": avg_loss, f"{metric_key_prefix}_avg_Accuracy": avg_acc})
            
            # Save results into file and list
            self.store_results(metrics)
            self.all_results.append(structured_results)

            # Increment evaluation epoch
            self.evaluation_epoch += 1

            # Add average metrics with the prefix, for compatibility with super class
            metrics.update({loss_key: avg_loss, acc_key: avg_acc})

            return metrics
        # Otherwise, standard evaluation with eval_dataset
        else:
            return Trainer.evaluate(self, eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)        
    
    def store_results(self, eval_results:dict):
        current_time = self.current_time_str()
        try:
            results_text = f"{current_time},{self.evaluation_epoch}"
            for key, value in eval_results.items():
                if key.endswith("_Accuracy"):
                    results_text += f",{value:.3f}"
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

#endregion

#region ###################################### Main CLI ######################################
if __name__ == "__main__":
    # Load configuration
    logging.info("######### START: CONFIGURATION #########")
    target_dir = argument_parsing()
    config = get_config_from_file(target_dir)
    tri = TRI(**config)
    logging.info("######### END: CONFIGURATION #########")
    
    # Run all sections
    tri.run(verbose=True)
#endregion
