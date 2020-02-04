import pandas as pd
import numpy as np
import os
import argparse

def ensemble_vote(row,default_column):
    bins = np.bincount(row)
    if np.max(bins) == 1:
        return row[default_column]
    return np.argmax(bins)

def main():
    
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The data dir. Should contain the test.csv and code_descriptions.csv files in respective run folders")
    
    parser.add_argument("--result_dir", default=None, type=str, required=True,
                        help="The result data dir. Should contain the result.csv files in respective run folders")
    
    parser.add_argument("--runs", default=None, type=int, required=True,
                        help="The result data dir. Number of runs")
    
    parser.add_argument("--output_file", default=None, type=str, required=True,
                        help="The output file path. Output file is written in this path")
    
    args = parser.parse_args()
    
    test = pd.read_csv(os.path.join(args.data_dir,'test.csv'))
    events_df = pd.read_csv(os.path.join(args.data_dir,'code_descriptions.csv'))
                       
    events_df['index']= events_df.index
    
    sol = pd.DataFrame()
    
    for i in range(1, args.runs + 1):
        run = 'run' + str(i)
        print('Loading result file from ' + run)
        result = pd.read_csv(os.path.join(args.result_dir,run, 'result.csv'))
        result['result'] = result.idxmax(axis=1)
        sol[run] = result['result']
                             
    print('Ensembling results')
                             
    sol['index'] = sol.apply(lambda row: ensemble_vote(row,'run2'), axis=1)
                             
    merged_df = pd.merge(sol,events_df,how='left')
                             
    test['event'] = merged_df['event']
                             
    print('Writing ensembled results')
                             
    test.to_csv(args.output_file,index=False)
                             
    
    
    
if __name__ == "__main__":
    main()