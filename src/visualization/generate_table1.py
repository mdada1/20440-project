import pandas as pd
import dataframe_image as dfi

path_to_processed_data = '.\\data\\processed\\' # for github
#path_to_processed_data = '/content/' # for google colab

path_to_save_figures = '.\\fig\\main_fig\\' # for github
#path_to_save_figures = '/content/' # for google colab

annotation_df = pd.read_pickle(path_to_processed_data + 'GSE114065_series_matrix.pkl')


annotation_df_characteristics = annotation_df[annotation_df['Sample_title'].str.startswith('Sample_characteristics')]

sample_titles = annotation_df_characteristics['Sample_title']
sample_titles = [title.replace('Sample_characteristics_ch1_', '') for title in sample_titles]
sample_titles = [title.replace('_', ' ') for title in sample_titles]
sample_titles = [title.title() for title in sample_titles]
sample_titles[0] = 'Individual' # fix typo in annotation file

annotation_df_characteristics['Sample_title'] = sample_titles
annotation_df_characteristics = annotation_df_characteristics.set_index('Sample_title')


# save the subset of the dataframe as a table/image
pd.set_option("display.max_colwidth", None)
pd.set_option('display.width', -1)
pd.set_option('display.max_rows', None)

dfi.export(annotation_df_characteristics, path_to_save_figures + "Table1.png", table_conversion="matplotlib", max_cols=12)