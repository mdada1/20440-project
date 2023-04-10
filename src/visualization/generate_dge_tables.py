import pandas as pd
import dataframe_image as dfi

path_to_processed_data = '..\\..\\data\\results\\differential_gene_expression\\'

# 

path_to_save_figures = '.\\fig\\tables\\' # for github
files_to_convert = ['activ_pers_vs_resolved_baseline_genomewide.csv',
                    'activ_pers_vs_resolved_followup_allergvsctrlbaselinedesubset.csv',
                    'activ_pers_vs_resolved_followup_genomewide.csv']



# remove
files_to_convert = ['activ_pers_vs_resolved_baseline_genomewide.csv']

gene_names = pd.read_csv('..\\..\\data\\raw\\raw counts matrix and human gene annotation table.csv')
gene_names.head()



for file_name in files_to_convert:

    cols = ['GeneID', 'baseMean', 'log2(fold-change)', 'lfcSE', 'stat', 'p-value', 'Adjusted p-value']

    df = pd.read_csv(path_to_processed_data + 'activ_pers_vs_resolved_baseline_genomewide.csv', names=cols)
    df = df.drop(0)
    df = df.drop(['baseMean', 'lfcSE', 'stat'], axis=1)

    # merge the two dataframes based on the GeneID column
    merged_df = pd.merge(df, gene_names, on='GeneID')
    # select the Symbol column to replace the GeneID column
    merged_df = merged_df[['Symbol']]

    df['GeneID'] = merged_df[['Symbol']]

    df = df.rename({'GeneID': 'Gene'}, axis=1)

    df.head()

    # save the subset of the dataframe as a table/image
    pd.set_option("display.max_colwidth", None)
    pd.set_option('display.width', -1)
    pd.set_option('display.max_rows', 10)

    dfi.export(df, path_to_save_figures + file_name + "_table.png", table_conversion="matplotlib", max_cols=12, max_rows=10)