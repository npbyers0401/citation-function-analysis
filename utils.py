import os
import datetime
import pandas as pd
import transformers
import requests
import json
import torch
import json
import pdb
import re
import time
import pprint
import ast



import copy
from string import punctuation

def get_cuda_footprint():
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))


def get_quant_config(eigth_bit, four_bit):
    config = None
    if eigth_bit is True:
        config = get_8bit_config()
    if four_bit is True:
        config = get_4bit_config()
    return config


def get_8bit_config():
    quantization_config = transformers.BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None,
        llm_int8_enable_fp32_cpu_offload=False,
        llm_int8_has_fp16_weight=False
    )
    return quantization_config

def get_4bit_config():
    quantization_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    return quantization_config

#can pass in l8,l70 or l405 -- or can just pass the full model_name
def get_model_name(model_name='l70'):
    if model_name == 'l1':
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
    if model_name == 'l8':
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
    if model_name == 'l70':
        model_name = "meta-llama/Llama-3.3-70B-Instruct"
    if model_name == 'l405':
        model_name = "meta-llama/Meta-Llama-3.1-405B-Instruct"
    if model_name == 'qwen':
        model_name = "Qwen/Qwen2.5-72B-Instruct"
    if model_name == "smol1.7":
        model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    if model_name == "smol0.35":
        model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
    if model_name == 'l405':
        model_name = "meta-llama/Meta-Llama-3.1-405B-Instruct"
    if model_name == 'qwen':
        model_name = "Qwen/Qwen2.5-72B-Instruct"
    if model_name == "smol1.7":
        model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    if model_name == "smol0.35":
        model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
    return model_name


def determine_hardware():
    device = torch.device("cuda" if torch.cuda.is_available()
                          #else "" if torch.backends.mps.is_available() 
                          else "cpu")
    return device


def get_pipeline(model='l8',
                 four_bit=False,
                 eigth_bit=False,
                 device='auto',
                 ):
    model_name = get_model_name(model)
    # Get appropriate quantization config values
    quantization_config = get_quant_config(four_bit=four_bit, eigth_bit=eigth_bit)
    #if determine_hardware() != 'cuda':
    #    cpu = determine_hardware()
    #    device =  cpu
    pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            token=os.getenv('HF_TOKEN'),
            #torch_dtype=torch.float16, #needed for 70B with bigger context windows
            model_kwargs={
                "quantization_config" : quantization_config,
            },
                device_map=device,
            )
    return pipeline
    


def get_time(path_version=False):
    if path_version:
        now = datetime.datetime.now()
        now.strftime("%Y%m%d_%H%M%S")
        now_time = now.strftime("%Y%m%d_%H%M%S")
        return now_time
    else:
        dt = datetime.datetime.now()
        date = f'{dt.month}_{dt.day}'
        time = f'{dt.hour}_{dt.minute}'
        return date, time
   
    
            
def get_output_local(**kwargs):
    pipeline = kwargs['pipeline']

    if kwargs['do_sample']==False:
        outputs = pipeline(
            kwargs['message_param'],
            max_new_tokens=kwargs['max_completion_tokens'],
            return_full_text=False,
            eos_token_id=128009,
            pad_token_id = 128009,
            do_sample=kwargs['do_sample'],
            temperature=None,
            top_p = None
            )
    else:
        outputs = pipeline(
            kwargs['message_param'],
            max_new_tokens=kwargs['max_completion_tokens'],
            return_full_text=False,
            eos_token_id=128009,
            pad_token_id = 128009,
            temperature=kwargs['temp'],
            top_p = kwargs['top_p']
            )     
    
    return outputs[0]["generated_text"]


def chat_message(message, resp_type='bool', **kwargs):
    output = ''

    _kwarg = kwargs
    _kwarg['message_param'] = message
    if resp_type=='prose':
        _kwarg['max_completion_tokens'] = 500
    if resp_type=='list':
        _kwarg['max_completion_tokens'] = 500
    
    if resp_type=='bool':
        output = get_output_local(**_kwarg).strip(punctuation+" "+"\n")
    else:
        output = get_output_local(**_kwarg).strip()

    # Add bits about a given exchange to the param log
    param_dict = {}
    param_dict = {'resp_type':resp_type}
    merged_dict = {**param_dict, **_kwarg}
    
    return output, merged_dict


def add_user_prompt(case, message_param, target_key=None, accession_meta=None, article_text=None, rsp_example=None):
    user_prompt = ''
    if case == 'experiment':
        user_prompt += 'You are given a scientific article. Determine whether or not the article presents any original experiments. '
        user_prompt += 'Respond with one word only, "True" or "False", and refrain from providing any other information.\nScientific Article:\n"""'
        user_prompt += article_text
        user_prompt += '"""'
    elif case == 'insilico':
        user_prompt += 'Given that the article does not include any original experiments, does the article present any original in silico bioinformatics analyses, such as targeted database searches, comparative analyses, or re-annotations, to support its statements? '
        user_prompt += 'Respond with one word only, "True" or "False", and refrain from providing any other information.'
    elif case == 'meta_incl_exp':
        user_prompt = ''
        user_prompt += 'You are now provided with an accession number that refers to a specific genomic dataset and some information about that dataset. '
        user_prompt += f'Using your previous responses to guide your answer, determine whether or not the authors of the scientific article included this dataset in either A) original experiments or B) bioinformatics analyses. '
        user_prompt += 'Respond "True" if the dataset was included. Respond "False" if the dataset was not included. '
        user_prompt += 'Respond with one word only and refrain from providing any other information.\n'
        user_prompt += '###.\n'
        user_prompt += f'ACCESSION NUMBER: {target_key}\n'
        user_prompt += f'DATASET INFORMATION: """{accession_meta}"""\n'
        user_prompt += '###.\n'
        user_prompt += 'Response:'
    elif case == 'meta_incl_example':
        user_prompt = ''
        user_prompt += 'You are now provided with an accession number that refers to a specific genomic dataset and some information about that dataset. '
        user_prompt += f'Given that the article does not include any novel experimentation or analyses, do the authors mention the genomic dataset with accession number {target_key} as an example to demonstrate available datasets or database features? '
        user_prompt += 'Respond with one word only, "True" or "False", and refrain from providing any other information.\n'
        user_prompt += '###.\n'
        user_prompt += f'ACCESSION NUMBER: {target_key}\n'
        user_prompt += f'DATASET INFORMATION: """{accession_meta}"""\n'
        user_prompt += '###.\n'
        user_prompt += 'Response:'
    elif case == 'meta_incl_final':
        user_prompt = ''
        user_prompt += f'You declared that the article does not include any novel experimentation or analyses and that the genomic dataset with accession number {target_key} is mentioned as an example. '
        user_prompt += 'Is the dataset used for any other purposes besides providing an example? '
        user_prompt += 'Respond with one word only, "True" or "False", and refrain from providing any other information. '
        user_prompt += 'Response:'
    elif case == 'genom_experiment':
        user_prompt += 'Given that the article includes descriptions of original experiments or other data analyses, determine whether or not these experiments or analyses included any analyses of genomic sequence data. '
        user_prompt += 'Respond with one word only, "True" or "False", and refrain from providing any other information.'
    elif case == 'follow_up':
        user_prompt += 'Provide 1-3 sentences explaining your previous answer.'
    elif case=='use_cases':
        user_prompt += 'Provide a list of all distinct, unique, non-overlapping use cases in the paper that made direct use of this dataset. '
        user_prompt += 'By "use case", I refer to specific operations, analyses, processes, or workflows that tangibly leveraged or produced a given dataset. '
        user_prompt += 'These use cases do not constitute the chief findings of a paper, but rather constitute steps taken by the authors to reach those findings. '
        user_prompt += 'I am interested in use cases across all stages of the research data lifecycle, including data creation, processing, analysis, preservation, access, and reuse. '
        user_prompt += 'Avoid redundant or overlapping descriptions, and condense similar use cases into single descriptions. '
        user_prompt += 'List use cases with as high a level of granularity as possible. '
        user_prompt += 'Provide your responses as a Python list and follow the formatting in the provided example. Avoid any explanations for including or excluding specific use cases.\n'
        user_prompt += f'EXAMPLE:\n"""["use case 1", use case 2", ...]"""\n'
        user_prompt += 'RESPONSE:'
    elif case == 'tools_software':
        user_prompt += 'Given this list of use cases, provide a list of all software tools leveraged by the authors as part of these use cases. '
        user_prompt += 'By "software tool", I refer to computer programs, search tools, or analytical algorithms that either received the dataset as an input or produced it as an output. '
        user_prompt += 'Refrain from including physical tools or wet lab tools, such as PCR kits or sequencing platforms, in your list. '
        user_prompt += 'Provide your responses as a Python list and follow the formatting in the provided example. If no tools were used, provide an empty list.\n'
        user_prompt += f'EXAMPLE:\n"""["tool 1", tool 2", ...]"""\n'
        user_prompt += 'RESPONSE:'


    

    message_param.append({'role':'user', 'content':user_prompt})

    return message_param


def get_article(pmcid):

    article_text = ''

    with open('./data/papers/'+pmcid+'.txt', 'r') as f:
        article_text = f.read()

    return article_text  
    
def get_meta(target_key, key_meta_dir):
    accession_meta = ''
    acc_df = pd.read_csv(key_meta_dir)
    accession_meta = acc_df.loc[acc_df['acc']==target_key, :]['prompt'].values[0]

    return accession_meta


def art_suppress(message_param, article_text):
    print_message = [{'role':k['role'], 'content':k['content'].replace(article_text, '[ARTICLE_TEXT]')} for k in message_param]
    return print_message  
     
def chat_convo(pmcid, target_key, system_prompt, key_meta_dir, argdict, verbose=False):

    rsp = ''
    rsp_ex = ''
    rsp_da = ''
    rsp_uc = ''
    rsp_ts = ''

    data_accessed = ''
    use_cases = ''
    tools_software = ''

    if verbose:
        print(pmcid, ">>>>", target_key)
        print("---------")

    if verbose:
        print("Retrieving article text.")
    article_text = get_article(pmcid)
    if verbose:
        print("Retrieving accession details.")
    accession_meta = get_meta(target_key, key_meta_dir)
    if verbose:
        print("---------")

    message_param = []
    param_log = []

    if system_prompt!='':
        message_param.append({"role": "system", "content": system_prompt})

    # Structure 2 - This one appears to break against PMC5210664 >>>> Gs0095506. Also PMC8920927 >>>> CU928158
    message_param = add_user_prompt('experiment', message_param, article_text=article_text)
    rsp, param_dict = chat_message(message=message_param, resp_type='bool', **argdict)
    if verbose:
        print("Experiments?", rsp)
    message_param.append({'role':'assistant', 'content':rsp})
    param_dict['message_param'] = art_suppress(copy.deepcopy(message_param), article_text)
    param_log.append(param_dict)

    if rsp=='True':
        # Add bit about genome experiments
        message_param = add_user_prompt('genom_experiment', message_param, article_text=article_text)
        rsp, param_dict = chat_message(message=message_param, resp_type='bool', **argdict)
        if verbose:
            print("Genomic Experiments?", rsp)
        message_param.append({'role':'assistant', 'content':rsp})
        param_dict['message_param'] = art_suppress(copy.deepcopy(message_param), article_text)
        param_log.append(param_dict)

        # Add bit about target key
        message_param = add_user_prompt('meta_incl_exp', message_param, accession_meta=accession_meta, target_key=target_key)
        rsp_da, param_dict = chat_message(message=message_param, resp_type='bool', **argdict)
        if verbose:
            print("This specific data? ***", rsp_da, "***")
        message_param.append({'role':'assistant', 'content':rsp_da})
        param_dict['message_param'] = art_suppress(copy.deepcopy(message_param), article_text)
        param_log.append(param_dict)
    elif rsp=='False':

        # In Silico
        message_param = add_user_prompt('insilico', message_param, article_text=article_text)
        rsp, param_dict = chat_message(message=message_param, resp_type='bool', **argdict)
        if verbose:
            print("Data Analyses?", rsp)
        message_param.append({'role':'assistant', 'content':rsp})
        param_dict['message_param'] = art_suppress(copy.deepcopy(message_param), article_text)
        param_log.append(param_dict)

        if rsp == 'True':
            # Add bit about genome experiments
            message_param = add_user_prompt('genom_experiment', message_param, article_text=article_text)
            rsp, param_dict = chat_message(message=message_param, resp_type='bool', **argdict)
            if verbose:
                print("Genomic analyses?", rsp)
            message_param.append({'role':'assistant', 'content':rsp})
            param_dict['message_param'] = art_suppress(copy.deepcopy(message_param), article_text)
            param_log.append(param_dict)

            # Add bit about target key
            message_param = add_user_prompt('meta_incl_exp', message_param, accession_meta=accession_meta, target_key=target_key)
            rsp_da, param_dict = chat_message(message=message_param, resp_type='bool', **argdict)
            if verbose:
                print("This specific data? ***", rsp_da, "***")
            message_param.append({'role':'assistant', 'content':rsp_da})
            param_dict['message_param'] = art_suppress(copy.deepcopy(message_param), article_text)
            param_log.append(param_dict)


        else:
            # False on in silico and false on experiments
            # Ask if included as an example
            message_param = add_user_prompt('meta_incl_example', message_param, accession_meta=accession_meta, target_key=target_key)
            rsp_ex, param_dict = chat_message(message=message_param, resp_type='bool', **argdict)
            if verbose:
                print("Used as an example?", rsp_ex)
            message_param.append({'role':'assistant', 'content':rsp_ex})
            param_dict['message_param'] = art_suppress(copy.deepcopy(message_param), article_text)
            param_log.append(param_dict)
            if rsp_ex == 'False':
                rsp_da=rsp_ex

            if rsp_ex == 'True':
                # Add bit about target key
                message_param = add_user_prompt('meta_incl_final', message_param, accession_meta=accession_meta, target_key=target_key)
                rsp_da, param_dict = chat_message(message=message_param, resp_type='bool', **argdict)
                if verbose:
                    print("Any other uses? ***", rsp_da, "***")
                message_param.append({'role':'assistant', 'content':rsp_da})
                param_dict['message_param'] = art_suppress(copy.deepcopy(message_param), article_text)
                param_log.append(param_dict)

    data_accessed = rsp_da

    if verbose:
        print("---------")


    message_param = add_user_prompt('follow_up', message_param)
    rsp, param_dict = chat_message(message=message_param, resp_type='prose', **argdict)
    if verbose:
        print()
        print(rsp)
    message_param.append({'role':'assistant', 'content':rsp})
    param_dict['message_param'] = art_suppress(copy.deepcopy(message_param), article_text)
    param_log.append(param_dict)


    if rsp_da == 'True':
        message_param = add_user_prompt('use_cases', message_param)
        rsp_uc, param_dict = chat_message(message=message_param, resp_type='list', **argdict)
        if verbose:
            print()
            print(rsp_uc)
        message_param.append({'role':'assistant', 'content':rsp_uc})
        param_dict['message_param'] = art_suppress(copy.deepcopy(message_param), article_text)
        param_log.append(param_dict)
        use_cases = rsp_uc

        message_param = add_user_prompt('tools_software', message_param)
        rsp_ts, param_dict = chat_message(message=message_param, resp_type='list', **argdict)
        if verbose:
            print()
            print(rsp_ts)
        message_param.append({'role':'assistant', 'content':rsp_ts})
        param_dict['message_param'] = art_suppress(copy.deepcopy(message_param), article_text)
        param_log.append(param_dict)
        tools_software = rsp_ts
        pass
    else:
        use_cases = "[]"
        tools_software = "[]"

    return message_param, param_log, data_accessed, use_cases, tools_software
