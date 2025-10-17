#!/bin/bash/python

import pdb
import os
from time import sleep
import time
from utils import *
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help='Path of config Json')
    config_path = parser.parse_args().config
    with open(config_path, 'r') as config_fp:
        config = json.load(config_fp)


    run_name = config['run_name']
    model_type = config['model_type']
    eight_bit_quant = config['eight_bit_quant']
    four_bit_quant = config['four_bit_quant']
    key_pair_dir = config['key_pair_dir']
    key_meta_dir = config['key_meta_dir']
    system_prompt = config['system_prompt']
    do_sample = config['do_sample']
    top_p = config['top_p']
    temperature = config['temperature']
    
    print('Process begun.')
    torch_version = torch.__version__
    print(f'The torch version available is {torch_version}')

    print(f'Our HF_HOME is {os.environ["HF_HOME"]}')
    print('Starting')
    #get_cuda_footprint()
    print('Retrieving model / pipeline.')
    m = get_model_name(model_name=model_type)
    model_type = m
    print(model_type)

    pipeline = get_pipeline(model_type,
                            eigth_bit=eight_bit_quant,
                            four_bit=four_bit_quant)
    print('----'*15)
    print('Here is the memory footrint after loading the model:')
    get_cuda_footprint()
    print(f'Pipeline retrieved: we are using {model_type}\n')
    print(f'The temperature for this run is set to: {temperature}')

    results_path = './data/results/chats/'+run_name
    
    if os.path.exists(results_path):
        print("Run directory already exists.")
    else:
        os.mkdir(results_path)
        print("New run directory created.")
    print(f'Path for results directory is {str(results_path)}')

    all_pairs = pd.read_csv(key_pair_dir)
    complete = os.listdir(results_path)
    com_pmcids = [p.split('-')[0] for p in complete if 'json' in p]
    com_tks = [p.split('-')[1] for p in complete if 'json' in p]


    max_completion_tokens = 10 # Set to 10 for initial boolean questions to prevent runover
    message_param = []

    args = {'pipeline':pipeline,
               'm':m,
               'stream':False,
               'max_completion_tokens':max_completion_tokens,
               'do_sample':do_sample,
               'temp':temperature,
               'top_p':top_p}
   

    counter = 0
    start_run_time = time.time()
    for (pmcid, target_key) in all_pairs.itertuples(index=False, name=None):
        start_pass_time = time.time()
        print(pmcid)
        print(target_key)
        if target_key not in com_tks:
            try:
                message_param, param_log, data_accessed, use_cases, tools_software =  chat_convo(pmcid, target_key, system_prompt, key_meta_dir, args)
    
                export_message = {'message':message_param, 'target_key':target_key, 'pmcid':pmcid}
    
                export_results = {'data_accessed':data_accessed, 'use_cases':use_cases, 'tools_software':tools_software, 'target_key':target_key, 'pmcid':pmcid}
                print('Results generated.')
    
                export_params = param_log
                for b in export_params:
                    b['pipeline'] = str(b['pipeline'])
                    b['target_key'] = target_key
                    b['pmcid'] = pmcid
    
                now_time = get_time(path_version=True)
                dump_file_results = f'{results_path}/{pmcid}-{target_key}-RESULTS_{now_time}.json'
                dump_file_message = f'{results_path}/{pmcid}-{target_key}-MESSAGE_{now_time}.json'
                dump_file_params = f'{results_path}/{pmcid}-{target_key}-PARAMS_{now_time}.json'
    
    
                # Three output files
                with open(dump_file_message, 'w') as f:
                    # One for the entire final message chain
                    json.dump(export_message, f)
                with open(dump_file_results, 'w') as f:
                    # One for the three items we're looking for (use cases, data_accessed, tools_software)
                    json.dump(export_results, f)
                with open(dump_file_params, 'w') as f:
                    # One for all of the params, individualized per chat message
                    json.dump(export_params, f)
    
                # break
            except Exception as e:
                print()
                print(f'ERROR: {pmcid}, {target_key} |||| {e}')
                print()
            end_pass_time = time.time()
            pass_runtime = end_pass_time - start_pass_time
            print(f'Total seconds for pass: {pass_runtime}')
            print('GPU ram usage:')
            print("------------")
            print()
            get_cuda_footprint()
    end_run_time = time.time()
    total_runtime = end_run_time - start_run_time
    print(f'Total seconds for run: {total_runtime}')
    print(f'Total minutes for run: {total_runtime/60}')



