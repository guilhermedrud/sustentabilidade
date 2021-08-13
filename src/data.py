# concat all dfs
import pandas as pd
import numpy as pd

def process_dir_files(dirname,dest_dir='.',merge=False,save_csv=False):
    df_list = []
    if not os.path.exists(dest_dir) and save_csv:
        os.mkdir(dest_dir)
    for filepath in glob.iglob(dirname+'/*.xls'):
        df = pd.read_excel(filepath,true_values='X')
        df['Unnamed: 0'] = pd.Series(df['Unnamed: 0']).fillna(method='ffill')
        df = df.groupby(by=['Unnamed: 0','Unnamed: 1']).sum()
        df = df.replace(0, np.nan)
        df = df.dropna(how='all', axis=0)
        df = df.replace(np.nan, 0)
        df = df.replace(True,1)
        df = df.T
        df_list.append(df)
        if save_csv:
            df.to_csv(dest_dir+'/'+filepath[16:-4]+'.csv')
    if merge:
        df = pd.concat(df_list, axis=1,join='outer')
        if save_csv: df.to_csv(dest_dir+'/'+'merged'+'.csv')
        return df
    return df_list









