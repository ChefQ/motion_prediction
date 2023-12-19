# %%
from transformers import pipeline, set_seed
from datasets import load_dataset
import pandas as pd
import json
import regex as re
import nltk
from nltk.tokenize import sent_tokenize
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import ast
import torch
from huggingface_hub import scan_cache_dir
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig
import ast
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
import os
from datasets import DatasetDict, concatenate_datasets
from sentence_transformers import SentenceTransformer
from joblib import dump, load
import argparse
import joblib

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

#watch -n0.1 nvidia-smi

# %%
### Load model


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


def meanEmbeddings(briefs):

    briefs = ast.literal_eval(briefs)
    # Place model inputs on the GPU
    embeddings = []
    for brief in briefs:
        argument = tokenizer(brief, max_length = max_input_size , padding="max_length" ,truncation= True ,return_tensors="pt").to(device) 
        # Extract last hidden states
        model.eval()
        with torch.no_grad():
            argument = argument.to(device)
            output = model(**argument)
            argument = argument.to(device)
            last_hidden_state = output.last_hidden_state


        last_hidden_state = last_hidden_state.reshape(( max_input_size, config.hidden_size))
        mask = argument['attention_mask'].to(device).bool()
        mask = mask.reshape(max_input_size)
        mask = mask.nonzero().squeeze()
        hidden_states = torch.index_select(last_hidden_state, 0, mask)
        
        #print(last_hidden_state)
        inputs = get_mean_embedding(hidden_states.cpu().to(torch.float64).numpy()).tolist()
        del last_hidden_state
        del output
        del mask
        del hidden_states
        del argument
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
                
        embeddings.append(inputs)
    # Return vector for [CLS] token
    return embeddings

def summerizeEmbeddings(brief):
    embedding = sentence_model.encode(brief).tolist()
    return embedding



def get_mean_embedding(embedding):
    return np.mean(embedding, axis=0)


##Arguments


# model_name = "KNN"
# feature = 'tfidf'
# #for model_name in ["RFT", "SGD"]: #["KNN","LinearSVC", "Logistic", "RFT", "SGD"]:
# data = 'dataset/testset.csv'
if __name__ == "__main__": #True:
    model_types = ["KNN","LinearSVC", "Logistic", "RFT", "SGD"]
    parser = argparse.ArgumentParser(description='End to end pipeline from briefs to predictions')
    parser.add_argument('--model_name', default='RFT', help=f'There are {len(model_types)} models to choose from: {model_types}')
    parser.add_argument('--data',  default= 'embeddings.csv')
    parser.add_argument('--feature', default='tfidf', help='There are two features to choose from: sentence_embeddings and tfidf')



    arg = parser.parse_args()

    testset = pd.read_csv(arg.data, index_col=0)# pd.read_csv(data, index_col=0)#

    testset["completion"] = list(map(lambda x : x.strip() ,testset["completion"].to_list()))
    

    summerize = True

    if arg.feature == 'sentence_embeddings':#if arg.feature == 'sentence_embeddings':
        if 'embeddings' not in testset.columns:
            if not summerize:
                model  = "mistralai/Mistral-7B-v0.1"
                config = AutoConfig.from_pretrained(model)
                max_input_size =  config.max_position_embeddings  
                tokenizer = AutoTokenizer.from_pretrained(model, device=device, )

                tokenizer.pad_token = tokenizer.eos_token
                model = AutoModel.from_pretrained(model, torch_dtype= torch.bfloat16 ).to(device)  #,device_map = "auto"
                #parallel_model = torch.nn.DataParallel(model)
            else:
                sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)

            testset['feature'] = ""
            testset['feature'] = testset['prompt'].map(summerizeEmbeddings)
            testset.to_csv('embeddings.csv')


        else:
            #get model for predictions
            turn2list = lambda x: ast.literal_eval(x)
            testset['feature'] = testset['embeddings'].map(turn2list)


    elif arg.feature == 'tfidf':# elif arg.feature == 'tfidf':
        print("Loading tfidf pipes")
        support_pipe = joblib.load('pipes/support-tfidf.joblib')
        opposition_pipe = joblib.load('pipes/oppose-tfidf.joblib')

        testset['feature'] = ""

        sparse_matrix = support_pipe.transform(testset["prompt"].loc[testset["brief_type"]=="support"]).toarray()

        testset['feature'].loc[testset["brief_type"]=="support"] =   sparse_matrix.tolist()

        sparse_matrix = opposition_pipe.transform(testset["prompt"].loc[testset["brief_type"]=="opposition"]).toarray()

        testset['feature'].loc[testset["brief_type"]=="opposition"] =   sparse_matrix.tolist()
        


    x_support = np.array(testset["feature"].loc[(testset["brief_type"]=="support") & (testset["data_type"]=="test") ].to_list())  #  np.array(testset['file_path'].to_list())
    x_opposition = np.array(testset["feature"].loc[(testset["brief_type"]=="opposition") & (testset["data_type"]=="test") ].to_list())  #  np.array(testset['file_path'].to_list())
        

    #get model for predictions

    support_model_path = f'models/{arg.model_name}-support-{arg.feature}.pkl'
    opposition_model_path = f'models/{arg.model_name}-opposition-{arg.feature}.pkl'

    clfs = {"sup" : load(support_model_path)  , "opp" : load(opposition_model_path) }

    if hasattr(clfs["sup"], 'predict_proba') and callable(getattr(clfs["sup"], 'predict_proba')):
        scores_support = clfs['sup'].predict_proba(x_support)
        scores_opposition = clfs['opp'].predict_proba(x_opposition)
    else:
        scores_support = clfs['sup'].decision_function(x_support)
        scores_opposition = clfs['opp'].decision_function(x_opposition)

    prediction_opposition = clfs['opp'].predict(x_opposition)
    prediction_support = clfs['sup'].predict(x_support)

    testset.rename(columns={"file_name": "brief","completion": "truth"} , inplace = True  )

    support = testset.loc[(testset["brief_type"]=="support") & (testset["data_type"]=="test") ].copy()
    opposition = testset.loc[(testset["brief_type"]=="opposition") & (testset["data_type"]=="test") ].copy()

    support.drop(['data_type','prompt' , 'brief_type','file_path','feature'], axis=1, inplace=True)
    opposition.drop(['data_type', 'prompt' , 'brief_type','file_path','feature'], axis=1, inplace=True)

    support['predict'] = ""
    opposition['predict'] = ""

    support['predict'] = prediction_support
    opposition['predict'] = prediction_opposition

    support['score'] = ""
    opposition['score'] = ""

    support['score'] = list(map( np.max ,scores_support.tolist()))
    opposition['score']= list(map(  np.max ,scores_opposition.tolist() ))

    support = support[["brief","predict","score","truth"]]
    opposition = opposition[["brief","predict","score","truth"]]

    support.to_csv(f'{arg.model_name}-{arg.feature}-supppredictions.csv' , index = False)
    opposition.to_csv(f'{arg.model_name}-{arg.feature}-oppopredictions.csv', index = False)
