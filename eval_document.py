import json
import os
import re
import pandas as pd
import ast
from datetime import datetime
import argparse

    
def raw_comparisons(outdir, labeldir):
    compare_pairs = []
    for fp in os.listdir(outdir):
        output_dict = {}
        if 'RESULTS' in fp and '.json' in fp:
            with open(outdir+"/"+fp, 'r') as f:
                output = json.loads(f.read())
                try:
                    if output['use_cases']!='[]':
                        output['use_cases'] = [{'case':p} for p in ast.literal_eval(output['use_cases'])]
                    else:
                        output['use_cases'] = ast.literal_eval(output['use_cases'])
                except:
                    pass
                try:
                    if output['use_cases']!='[]':
                        output['tools_software'] = [{'tool':p} for p in ast.literal_eval(output['tools_software'])]
                    else:
                        output['use_cases'] = ast.literal_eval(output['tools_software'])
                except:
                    pass
                output_dict['output'] = output
                output_dict['target_key'] = output['target_key']

            for fp2 in os.listdir(labeldir):
                if output_dict['target_key'].split('.')[0] in fp2:
                    with open(labeldir+"/"+fp2, 'r') as h:
                        gs_dict  = json.loads(h.read())
                        output_dict['gs'] = gs_dict
                        pmcid = re.findall('PMC[0-9]+', fp2)[0]
                        output_dict['pmcid'] = pmcid
                        break
            compare_pairs.append(output_dict)

    return compare_pairs

def eval_format(compare_pairs):
    eval_df = []
    for p in compare_pairs:
        for set in ['output', 'gs']:
            for val_type in ['use_cases', 'tools_software']:
                add_df = pd.DataFrame(p[set][val_type])
                if add_df.shape[0]>0:
                    add_df.columns = ['value']
                    add_df['val_type'] = val_type
                    add_df['set'] = set
                    add_df['tk'] = p['target_key']
                    add_df['pmcid'] = p['pmcid']
                    add_df = add_df.loc[:, ['pmcid', 'tk', 'set', 'val_type', 'value']]
                    eval_df.append(add_df)
            
            add_df = pd.DataFrame([{'value':eval(str(p[set]['data_accessed']))}])
            add_df['val_type'] = 'data_accessed'
            add_df['set'] = set
            add_df['tk'] = p['target_key']
            add_df['pmcid'] = p['pmcid']
            eval_df.append(add_df)


    eval_df = pd.concat(eval_df)
    eval_df['sort_value'] = eval_df['value'].astype(str).str.lower()
    eval_df = eval_df.sort_values(by=['pmcid', 'tk', 'set', 'val_type', 'sort_value']).reset_index(drop=True)
    eval_df = eval_df.drop(['sort_value'], axis=1)
    eval_df.head()

    return eval_df

def row_color(eval_df):
# eval row colors
    pair_grps = eval_df.groupby(['pmcid', 'tk'])
    color = 1

    color_rows = []
    for pair_grp in pair_grps.groups.keys():
        matches = pair_grps.get_group(pair_grp).copy()
        matches['color'] = color
        if color==1:
            color+=1
        elif color==2:
            color-=1

        color_rows.append(matches)

    eval_df = pd.concat(color_rows).reset_index(drop=True)
    eval_df.head()

    eval_df['score'] = ''
    eval_df['notes'] = ''

    return eval_df

def auto_eval(eval_df):

    pair_grps = eval_df.groupby(['pmcid', 'tk'])

    auto_eval_rows = []
    for pair_grp in pair_grps.groups.keys():
        matches = pair_grps.get_group(pair_grp).copy()

        matches_op = matches.loc[matches['set']=='output', :].copy()
        matches_gs = matches.loc[matches['set']=='gs', :].copy()

        # string match tools. Add +1 to the output version of any exact matches
        matches_op.loc[matches_op['value'].isin(matches_gs['value'])==True, ['score']] = 1
        # string match tools. Add 0 to the corresponding gs version
        matches_gs.loc[matches_gs['value'].isin(matches_op['value'])==True, ['score']] = 0

        # Deal with wrong data_accessed values
        da_op = matches_op.loc[matches_op['val_type']=='data_accessed', :]['value'].values[0]
        da_gs = matches_gs.loc[matches_gs['val_type']=='data_accessed', :]['value'].values[0]
        if da_op!=da_gs:
            # data_accessed mismatches. Add -1 for all output rows if this value is wrongly true
            if da_op==True:
                matches_op.loc[:, ['score']] = -1
                matches_gs.loc[:, ['score']] = 0
            # data_accessed mismatches. Add -1 for all gs rows if this value is wrongly False
            elif da_op==False:
                matches_op.loc[:, ['score']] = 0
                matches_gs.loc[:, ['score']] = -1

        auto_eval_rows.append(matches_gs)
        auto_eval_rows.append(matches_op)

    eval_df = pd.concat(auto_eval_rows).reset_index(drop=True)

    return eval_df



def main():  
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outdir", help='Path of results directory')
    parser.add_argument("-l", "--labeldir", help='Path of labels directory')
    outdir = parser.parse_args().outdir
    labeldir = parser.parse_args().labeldir

    compare_pairs = raw_comparisons(outdir, labeldir)
    eval_df = eval_format(compare_pairs)
    eval_df = row_color(eval_df)
    eval_df = auto_eval(eval_df)

    nowtime = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    eval_df.to_csv(outdir+"/eval_"+nowtime+".csv", index=False)

    print("Evaluation file generated:", outdir+"/eval_"+nowtime+".csv")


if __name__ == "__main__":
    main()