# %%
from transformers import pipeline, set_seed
from datasets import load_dataset
import pandas as pd
import json
import nltk
from nltk.tokenize import sent_tokenize
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import ast
import torch
from huggingface_hub import scan_cache_dir
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig
import ast
device = "cuda"
import os
from datasets import DatasetDict, concatenate_datasets
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

#watch -n0.1 nvidia-smi


config = AutoConfig.from_pretrained("mistralai/Mistral-7B-v0.1")
max_input_size =  config.max_position_embeddings  
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", device=device, padding_side="left", )
#tokenizer = AutoTokenizer.from_pretrained("TheBloke/mistral-7b-v0.1.Q6_K.gguf", device=device, padding_side="left", )

tokenizer.pad_token = tokenizer.eos_token
#model = AutoModel.from_pretrained("mistralai/Mistral-7B-v0.1",device_map = "auto" )#.to(device)  #,device_map = "auto"
model = AutoModel.from_pretrained("mistralai/Mistral-7B-v0.1", use_flash_attention_2=True, torch_dtype= torch.bfloat16 ).to(device)  #,device_map = "auto"
#parallel_model = torch.nn.DataParallel(model)

# %%

def getIdsType(brief_type):    
    def getIds(briefs):

        briefs = briefs[brief_type]
        briefs_ids = []
        briefs = ast.literal_eval(briefs)
        for brief in briefs:
            iD = tokenizer(brief, max_length = max_input_size, padding='max_length', truncation= True ,return_tensors="pt") #.to(device).input_ids 
            briefs_ids.append(iD)
        return { f"ids_{brief_type}": briefs_ids }

    return getIds



def supportEmbeddings(briefs):
    brief_type = "support"
    briefs = ast.literal_eval(briefs)
    # Place model inputs on the GPU
    embeddings = []
    for brief in briefs:
        support = tokenizer(brief, max_length = max_input_size , padding="max_length" ,truncation= True ,return_tensors="pt").to(device) 
        # Extract last hidden states
        model.eval()
        with torch.no_grad():
            support = support.to(device)
            output = model(**support)
            support = support.to(device)
            last_hidden_state = output.last_hidden_state
        #print(last_hidden_state)
        inputs = last_hidden_state.cpu().to(torch.float64).numpy()
        del last_hidden_state
        del output

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
                
        embeddings.append(inputs)
    # Return vector for [CLS] token
    return inputs


def oppositionEmbeddings(briefs):
    brief_type = "opposition"
    briefs = ast.literal_eval(briefs)
    # Place model inputs on the GPU
    embeddings = []
    for brief in briefs:
        support = tokenizer(brief, max_length = max_input_size , padding="max_length" ,truncation= True ,return_tensors="pt").to(device) 
        # Extract last hidden states
        model.eval()
        with torch.no_grad():
            support = support.to(device)
            output = model(**support)
            support = support.to(device)
            last_hidden_state = output.last_hidden_state
        #print(last_hidden_state)
        inputs = last_hidden_state.cpu().to(torch.float64).numpy()
        del last_hidden_state
        del output

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
                
        embeddings.append(inputs)
    # Return vector for [CLS] token
    return inputs







def setStatus(brief_type):
    
    def status(briefs):
        return { f"status_{brief_type}": False  , f'{brief_type}_hidden_states': np.zeros((1,10,10)) }
    
    return status


paired = pd.read_csv('paired_testset.csv', index_col=0)

# # # convert to Dataset
# testset = load_dataset("csv", data_files='paired_testset_embeddings.csv', index_col=0,)
# # testset = testset.map(getIdsType("support"), batched=False, batch_size=None )#, remove_columns=["support", "opposition", "outcome", "folder_id", "data_type"])
# # testset = testset.map(getIdsType("opposition"), batched=False, batch_size=None )#, remove_columns=["support", "opposition", "outcome", "folder_id", "data_type"])


# testset = testset.map(setStatus("support"), batched=False, batch_size=None )
# testset = testset.map(setStatus("opposition"), batched=False, batch_size=None )

# testset

batch_size = 175
start = 100
paired['support_embeddings'] = ""
paired['opposition_embeddings'] = ""

for i in range(start, len(paired), batch_size):

    paired['support_embeddings'].iloc[i:i+batch_size] = paired['support'].iloc[i:i+batch_size].map(supportEmbeddings)
    paired['opposition_embeddings'].iloc[i:i+batch_size] = paired['opposition'].iloc[i:i+batch_size].map(oppositionEmbeddings)

    paired.iloc[i:i+batch_size].to_csv(f'paired_testset_embeddings_{i}-{i+batch_size}.csv')

# %%
