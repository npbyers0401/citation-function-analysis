import pandas as pd
import argparse


def bool_convert(eval_df):
    mask = eval_df['val_type'].eq('data_accessed')

    eval_df.loc[mask, 'value'] = (
        eval_df.loc[mask, 'value']
        .replace({'TRUE': True, 'FALSE': False})
    )

    return eval_df

def eval_classes(eval_df):

    metric_df = eval_df.loc[:, ['val_type', 'set', 'score']].copy()
    # Do multi types first and then add the binary class types afterwards
    multi_types = ['tools_software', 'use_cases']
    metric_df = metric_df.loc[((metric_df['score']!=0)&(metric_df['val_type'].isin(multi_types)==True)), :]
    metric_df['eval_class'] = ''
    # Multi Types
    metric_df.loc[((metric_df['set']=='gs')&(metric_df['score']==-1)), ['eval_class']] = 'fn' # False negatives are missed by output
    metric_df.loc[((metric_df['set']=='output')&(metric_df['score']==-1)), ['eval_class']] = 'fp' # False positive are erroneously added by output
    metric_df.loc[((metric_df['set']=='output')&(metric_df['score']==1)), ['eval_class']] = 'tp' # True positive are correctly assigned by output
    # Binary Types
    da_grp = eval_df.loc[eval_df['val_type']=='data_accessed', :].groupby('tk')
    da_assignments = []
    for tk in da_grp.groups.keys():
        eval_class = ''
        matches = da_grp.get_group(tk)
        actual = matches.loc[matches['set']=='gs', :]['value'].values[0]
        predicted = matches.loc[matches['set']=='output', :]['value'].values[0]

        if actual==True and predicted==True:
            eval_class = 'tp'
        elif actual==False and predicted==True:
            eval_class = 'fp'
        elif actual==True and predicted==False:
            eval_class = 'fn'
        elif actual==False and predicted==False:
            eval_class = 'tn'
        da_assignments.append({'val_type':'data_accessed', 'eval_class':eval_class})
    da_assignments = pd.DataFrame(da_assignments)

    metric_df = pd.concat([metric_df, da_assignments]).reset_index(drop=True)

    return metric_df

def metricalc(metric_df):
    tp = metric_df.loc[metric_df['eval_class']=='tp', :].shape[0]
    fp = metric_df.loc[metric_df['eval_class']=='fp', :].shape[0]
    fn = metric_df.loc[metric_df['eval_class']=='fn', :].shape[0]

    r = tp/(tp+fn)
    p = tp/(tp+fp)

    f1 = 2*((p*r)/(p+r))

    return r,p,f1

def metric_display(metric_df, evalpath):

    print("")
    print("EVAL METRICS: ", "/".join(evalpath.split('/')[-2:]))
    print("----")

    rec,prec,f1_score = metricalc(metric_df)

    print("Overall")
    print("Recall:", round(rec, 3))
    print("Precision:", round(prec, 3))
    print("F1:", round(f1_score, 3))
    print()

    for val_type in ['data_accessed', 'tools_software', 'use_cases']:
        print(val_type)
        rec,prec,f1_score = metricalc(metric_df.loc[metric_df['val_type']==val_type, :].copy())

        print("Recall:", round(rec, 3))
        print("Precision:", round(prec, 3))
        print("F1:", round(f1_score, 3))
        print()

def format_check(eval_df):
    fcheck = True

    for z in eval_df['score'].unique():
        if z not in [-1, 0, 1]:
            fcheck = False

    return fcheck

def main():  
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--evalpath", help='Path of eval document (.csv)')
    evalpath = parser.parse_args().evalpath

    eval_df = pd.read_csv(evalpath)

    eval_df = bool_convert(eval_df)

    fcheck = format_check(eval_df)

    if not fcheck:
        print('Invalid score values present. Check and resubmit.')
    else:
        metric = eval_classes(eval_df)
        metric_display(metric, evalpath)




if __name__ == "__main__":
    main()