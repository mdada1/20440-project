import pandas as pd
from IPython.display import Image
import dataframe_image as dfi
import matplotlib as plt
plt.rcParams['figure.figsize'] = [10, 5] # set default figure width to 10 inches and height to 5 inches
#from dataframe_image import StyleFrame, styling

path_to_processed_data = '..\\..\\data\\results\\differential_gene_expression\\'
#path_to_processed_data =  '/content/' # for google colab

path_to_save_figures = '.\\fig\\tables\\' # for github
#path_to_save_figures = '/content/' # for google colab

files_to_convert = ['activ_pers_vs_resolved_baseline_genomewide.csv',
                    'activ_pers_vs_resolved_followup_allergvsctrlbaselinedesubset.csv',
                    'activ_pers_vs_resolved_followup_genomewide.csv']


gene_names = pd.read_csv(path_to_processed_data + 'raw counts matrix and human gene annotation table.csv')
gene_names.head()

dfs = []

for file_name in files_to_convert:

    cols = ['GeneID', 'baseMean', 'log2(fold-change)', 'lfcSE', 'stat', 'p-value', 'Adjusted p-value']

    df = pd.read_csv(path_to_processed_data + file_name, names=cols, header=0)
    #df = df.drop(0)
    df = df.drop(['baseMean', 'lfcSE', 'stat'], axis=1)

    # merge the two dataframes based on the GeneID column
    merged_df = pd.merge(df, gene_names, on='GeneID')
    # select the Symbol column to replace the GeneID column
    merged_df = merged_df[['Symbol']]

    df['GeneID'] = merged_df[['Symbol']]

    df = df.rename({'GeneID': 'Gene'}, axis=1)

    df = df.head(10)

    #display(df)
    dfs.append(df)

#for df in dfs:
    # save the subset of the dataframe as a table/image
    #pd.set_option("display.max_colwidth", 100)
    #pd.set_option('display.width', 1000)
    #pd.set_option('display.max_rows', 10)

    pd.set_option('display.width', -1)

    # define a custom lambda function to format each value
    f = lambda x: '{:.4e}   '.format(x)
    df['p-value'] = df['p-value'].apply(f)
    df['p-value'] = df['p-value'].astype(str)
    df['p-value'] = df['p-value'].str.pad(width=20, side='left')
    df = df.rename({'p-value': 'p-value  '}, axis=1)

    f2 = lambda x: '{:.4f}   '.format(x)
    df['log2(fold-change)'] = df['log2(fold-change)'].apply(f2)
    df['log2(fold-change)'] = df['log2(fold-change)'].astype(str)
    df['log2(fold-change)'] = df['log2(fold-change)'].str.pad(width=30, side='left')

    df['Adjusted p-value'] = df['Adjusted p-value'].apply(f2)
    df['Adjusted p-value'] = df['Adjusted p-value'].astype(str)
    df['Adjusted p-value'] = df['Adjusted p-value'].str.pad(width=30, side='left')

    dfi.export(df.style.hide(axis='index'), path_to_save_figures + file_name + "_table.png", 
               table_conversion="matplotlib", max_cols=6, max_rows=10, dpi=400)#, max_column_width=3)