#region ###################################### Imports and initialization ######################################
from argparse import Namespace
import os
import logging

import en_core_web_lg
import torch
from transformers import pipeline

from functions import *

# Check for GPU with CUDA
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
else:
    DEVICE = torch.device("cpu")

# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO)

#endregion

if __name__ == "__main__":
    #region ###################################### Configuration ######################################
    logging.info("######### START: CONFIGURATION #########")

    #region ################### Arguments parsing ###################
    target_dir = argument_parsing()

    #endregion

    #region ################### Configuration file ###################
    config = get_config_from_file(target_dir)

    #endregion

    #region ################### Set configurations ###################
    # Mandatory configurations assigned to None
    OUTPUT_FOLDERPATH = None
    DATA_FILEPATH = None
    INDIVIDUAL_NAME_COLUMN = None
    BACKGROUND_KNOWLEDGE_COLUMN = None
    mandatory_configs_names = [var_name for var_name in locals() if var_name.isupper() and "_" in var_name]

    # Optional configurations with default values
    ANONIMIZE_BACKGROUND_KNOWLEDGE = True
    ONLY_USE_ANONYMIZED_BACKGROUND_KNOWLEDGE = False
    USE_DOCUMENT_CURATION = True
    SAVE_PRETREATMENT = True
    LOAD_SAVED_PRETREATMENT = True
    BASE_MODEL_NAME = "distilbert-base-uncased"
    TOKENIZATION_BLOCK_SIZE =250
    USE_ADDITIONAL_PRETRAINING = True
    SAVE_ADDITIONAL_PRETRAINING = True
    LOAD_SAVED_PRETRAINING = True
    PRETRAINING_EPOCHS = 3
    PRETRAINING_BATCH_SIZE = 8
    PRETRAINING_LEARNING_RATE = 5e-05
    PRETRAINING_MLM_PROBABILITY = 0.15
    PRETRAINING_SLIDING_WINDOW = "512-128"
    SAVE_FINETUNING = True
    LOAD_SAVED_FINETUNING = True
    FINETUNING_EPOCHS = 15
    FINETUNING_BATCH_SIZE = 16
    FINETUNING_LEARNING_RATE = 5e-05
    FINETUNING_SLIDING_WINDOW = "100-25"
    optional_configs_names = [var_name for var_name in locals() if var_name.isupper() and "_" in var_name and not var_name in mandatory_configs_names]

    # Get configurations into local variables
    for config_name, value in config.items():
        if config_name in mandatory_configs_names:
            locals().update({config_name:value})
            mandatory_configs_names.remove(config_name)
        elif config_name in optional_configs_names:
            locals().update({config_name:value})
            optional_configs_names.remove(config_name)
        else:
            logging.warning(f"Unrecognized configuration {config_name}")

    # Error checking
    if len(optional_configs_names) > 0:
        logging.info(f"Default values used for the following optional configuration/s: {optional_configs_names}")
    if len(mandatory_configs_names) > 0:
        raise Exception(f"No value given for the following mandatory configuration/s: {mandatory_configs_names}")

    # Paths
    if not os.path.exists(OUTPUT_FOLDERPATH):
        os.makedirs(OUTPUT_FOLDERPATH)
    logging.info(f"Results folder [{OUTPUT_FOLDERPATH}] created")
    PRETREATED_DATA_PATH = os.path.join(OUTPUT_FOLDERPATH, "Pretreated_Data.json")
    PRETRAINED_MODEL_PATH = os.path.join(OUTPUT_FOLDERPATH, "Pretrained_Model.pt")
    RESULTS_PATH = os.path.join(OUTPUT_FOLDERPATH, "Results.csv")
    TRI_PIPE_PATH = os.path.join(OUTPUT_FOLDERPATH, "TRI_Pipeline")
    TRI_MODEL_PATH = os.path.abspath(os.path.join(TRI_PIPE_PATH, "model.safetensors"))

    # Additional pretraining and finetuning configurations
    pretraining_config = Namespace()
    pretraining_config.is_for_mlm = True
    pretraining_config.uses_labels = False
    pretraining_config.epochs = PRETRAINING_EPOCHS
    pretraining_config.batch_size = PRETRAINING_BATCH_SIZE
    pretraining_config.learning_rate = PRETRAINING_LEARNING_RATE
    pretraining_config.sliding_window = PRETRAINING_SLIDING_WINDOW
    pretraining_config.trainer_folder_path = os.path.join(OUTPUT_FOLDERPATH, f"Pretraining")

    finetuning_config = Namespace()
    finetuning_config.is_for_mlm = False
    finetuning_config.uses_labels = True
    finetuning_config.epochs = FINETUNING_EPOCHS
    finetuning_config.batch_size = FINETUNING_BATCH_SIZE
    finetuning_config.learning_rate = FINETUNING_LEARNING_RATE
    finetuning_config.sliding_window = FINETUNING_SLIDING_WINDOW
    finetuning_config.trainer_folder_path = os.path.join(OUTPUT_FOLDERPATH, f"Finetuning")

    #endregion

    logging.info("######### END: CONFIGURATION #########")

    #endregion


    #region ###################################### Data ######################################
    logging.info("######### START: DATA #########")

    pretreated_data_loaded = False

    if LOAD_SAVED_PRETREATMENT and os.path.isfile(PRETREATED_DATA_PATH):
        #region ################### Load pretreatment ###################
        logging.info("######### START: LOADING SAVED PRETREATED DATA #########")
        train_df, eval_dfs = load_pretreatment(PRETREATED_DATA_PATH)
        pretreated_data_loaded = True
        logging.info("######### END: LOADING SAVED PRETREATED DATA #########")

        #endregion

    # Otherwise, read data from file
    else:
        if LOAD_SAVED_PRETREATMENT:
            logging.info(f"Impossible to load saved pretreated data, file {PRETREATED_DATA_PATH} not found.")

        #region ################### Data reading ###################
        logging.info("######### START: READ RAW DATA FROM FILE #########")

        logging.info("Reading data...")
        data_df = read_data(DATA_FILEPATH, INDIVIDUAL_NAME_COLUMN, BACKGROUND_KNOWLEDGE_COLUMN)
        logging.info("Data reading complete")

        # Split into train and evaluation (dropping rows where no documents are available)
        logging.info("Splitting into train (background knowledge) and evaluation (anonymized) sets...")
        train_df, eval_dfs = split_data(INDIVIDUAL_NAME_COLUMN, BACKGROUND_KNOWLEDGE_COLUMN, data_df)
        del data_df # Remove general dataframe for saving memory
        logging.info("Train and evaluation splitting complete")
        
        logging.info("######### END: READ RAW DATA FROM FILE #########")

        #endregion

    #region ################### Data statistics ###################
    logging.info("######### START: DATA STATISTICS #########")

    # Get individuals found in each set
    train_individuals, eval_individuals, all_individuals, no_train_individuals, no_eval_individuals = get_individuals(train_df, eval_dfs, INDIVIDUAL_NAME_COLUMN)

    # Label->Name and Name->Label dictionaries
    label_to_name, name_to_label = get_individuals_labels(all_individuals)

    # Show relevant information
    show_data_stats(train_df, eval_dfs, no_eval_individuals, no_train_individuals, eval_individuals)

    logging.info("######### END: DATA STATISTICS #########")

    #endregion

    #region ################### Data pretreatment ###################
    # Pretreat data if required and not already loaded
    if (ANONIMIZE_BACKGROUND_KNOWLEDGE or USE_DOCUMENT_CURATION) and not pretreated_data_loaded:
        logging.info("######### START: DATA PRETREATMENT #########")

        spacy_nlp = en_core_web_lg.load() # Load spaCy model

        ########## Anonymize background knowledge ##########
        if ANONIMIZE_BACKGROUND_KNOWLEDGE:
            logging.info("######### START: BACKGROUND KNOWLEDGE ANONYMIZATION #########")        
            train_df = anonymize_bk(train_df, spacy_nlp, ONLY_USE_ANONYMIZED_BACKGROUND_KNOWLEDGE)        
            logging.info("######### END: BACKGROUND KNOWLEDGE ANONYMIZATION #########")
        else:
            logging.info("######### SKIPPING: BACKGROUND KNOWLEDGE ANONYMIZATION #########")

        ########## Document curation ##########
        if USE_DOCUMENT_CURATION:
            logging.info("######### START: DOCUMENT CURATION #########")
            document_curation(train_df, eval_dfs, spacy_nlp)
            logging.info("######### END: DOCUMENT CURATION #########")
        else:
            logging.info("######### SKIPPING: DOCUMENT CURATION #########")

        ########## Save pretreatment ##########
        if SAVE_PRETREATMENT:
            save_pretreatment(train_df, eval_dfs, PRETREATED_DATA_PATH)

        logging.info("######### END: DATA PRETREATMENT #########")
    else: 
        logging.info("######### SKIPPING: DATA PRETREATMENT #########")

    #endregion

    #endregion


    #region ###################################### Build classifier ######################################
    logging.info("######### START: BUILD CLASSIFIER #########")

    if LOAD_SAVED_FINETUNING and os.path.exists(TRI_PIPE_PATH):
        #region ################### Load already trained TRI model ###################
        logging.info("######### START: LOAD ALREADY TRAINED TRI MODEL #########")

        # Get TRI classifier and tokenizer
        model, tokenizer = load_trained_TRI_model(TRI_PIPE_PATH, name_to_label)

        # Datasets for TRI
        train_dataset, eval_datasets_dict = create_datasets(train_df, eval_dfs, tokenizer, name_to_label, finetuning_config, TOKENIZATION_BLOCK_SIZE)

        # Create trainer for TRI
        trainer = get_trainer(model, finetuning_config, train_dataset, eval_datasets_dict, RESULTS_PATH)

        logging.info("######### END: LOAD ALREADY TRAINED TRI MODEL #########")

        #endregion

    # Otherwise, pretraing (if required) and finetune a TRI model
    else:
        if LOAD_SAVED_FINETUNING:
            logging.info(f"Impossible to load saved TRI pipeline, folder {TRI_PIPE_PATH} not found.")

        #region ################### Create base language model ###################
        logging.info("######### START: CREATE BASE LANGUAGE MODEL #########")
        base_model, tokenizer = create_base_model(BASE_MODEL_NAME)
        logging.info("######### END: CREATE BASE LANGUAGE MODEL #########")

        #endregion

        #region ################### Additional pretraining ###################
        if USE_ADDITIONAL_PRETRAINING:
            logging.info("######### START: ADDITIONAL PRETRAINING #########")

            if LOAD_SAVED_PRETRAINING and os.path.exists(PRETRAINED_MODEL_PATH):
                logging.info("Loading additionally pretrained base model")
                base_model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))
                logging.info("Additionally pretrained base model loaded")
            else:
                if LOAD_SAVED_PRETRAINING:
                    logging.info(f"Impossible to load saved pretrained base model, file {PRETRAINED_MODEL_PATH} not found.")

                # Datasets for additional pretraining
                train_dataset, _ = create_datasets(train_df, eval_dfs, tokenizer, name_to_label, pretraining_config, TOKENIZATION_BLOCK_SIZE)

                # Additionally pretrain the base language model
                base_model = additional_pretraining(base_model, tokenizer, train_dataset, BASE_MODEL_NAME, DEVICE, PRETRAINING_MLM_PROBABILITY, pretraining_config)
                
                if SAVE_ADDITIONAL_PRETRAINING:
                    logging.info("Saving additionally pretrained base model")
                    torch.save(base_model.state_dict(), PRETRAINED_MODEL_PATH)
                    logging.info("Additionally pretrained base model saved")
            
            logging.info("######### END: ADDITIONAL PRETRAINING #########")
        else:
            logging.info("######### SKIPPING: ADDITIONAL PRETRAINING #########")
        
        #endregion

        #region ################### Finetuning ###################
        logging.info("######### START: FINETUNING #########")

        # Datasets for finetuning
        train_dataset, eval_datasets_dict = create_datasets(train_df, eval_dfs, tokenizer, name_to_label, finetuning_config, TOKENIZATION_BLOCK_SIZE)

        # Finetuning for text re-identification
        tri_model, results, trainer = finetuning(base_model, BASE_MODEL_NAME, DEVICE, train_dataset, eval_datasets_dict, finetuning_config, RESULTS_PATH)

        if SAVE_FINETUNING:
            logging.info("Saving finetuned TRI pipeline...")
            pipe = pipeline("text-classification", model=tri_model, tokenizer=tokenizer)
            pipe.save_pretrained(TRI_PIPE_PATH)
            logging.info("Finetuned TRI model pipeline saved")

        logging.info("######### END: FINETUNING #########")

        #endregion

    logging.info("######### END: BUILD CLASSIFIER #########")

    #endregion


    #region ###################################### Predict TRIR ######################################
    logging.info("######### START: PREDICT TRIR #########")
    results = predict_TRIR(trainer)
    logging.info("######### END: PREDICT TRIR #########")

    #endregion
