import pandas as pd
import gzip

# read in RNAseq data and save to a dataframe
RNAseq_df = pd.read_csv('.\\data\\raw\\GSE114065_processed_RNAseq.txt.gz', sep='\t', header=0) # for github
# RNAseq_df = pd.read_csv('/content/GSE114065_processed_RNAseq.txt.gz', sep='\t', header=0) # for google colab
RNAseq_df = RNAseq_df.rename(columns={'Unnamed: 0': 'Gene'})
RNAseq_df.to_pickle('.\\data\\processed\\GSE114065_processed_RNAseq.pkl')


# read in data annotations and save to a dataframe
path_to_annotation_data = '.\\data\\raw\\GSE114065_series_matrix.txt.gz' # for github
# path_to_annotation_data = '/content/GSE114065_series_matrix.txt.gz' # for google colab

annotation_data = []
num_header_lines = 28 # number of lines to skip when reading in the dataframe

with gzip.open(path_to_annotation_data,'rt') as fin:
    for i, line in enumerate(fin):
        processed_line = line.strip()
        processed_line = processed_line.split('\t')
        processed_line = [item.strip('\"') for item in processed_line]
        processed_line[0] = processed_line[0].strip('!')
        if (i >= num_header_lines) and (len(processed_line) > 1):
          annotation_data.append(processed_line)

sample_titles = annotation_data.pop(0)
annotation_df = pd.DataFrame(annotation_data, columns=sample_titles)


# processing annotation data
repeat_row_titles_df = annotation_df[annotation_df.duplicated('Sample_title', keep=False) == True]

for i in repeat_row_titles_df.index:
  row = annotation_df.iloc[i]
  row_name = row[0]
  if ':' in row[-1]:
    new_row_name = row_name + '_' + row.str.split(': ').str[0][-1]
    new_row = row.str.split(': ').str[1]

    new_row[0] = new_row_name
    annotation_df.iloc[i] = new_row

annotation_df.to_pickle('.\\data\\processed\\GSE114065_series_matrix.pkl')  