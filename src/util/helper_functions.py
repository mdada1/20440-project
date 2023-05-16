import pandas as pd

#get sample IDs (columns in data df) matching a certain characteristic
def filter_samples(data, annotations, characteristic_name, desired_values):
    try:
        subset_df_list = []
        for desired_value in desired_values:
            index_of_characteristic = annotations.loc[annotations['Sample_title'] == characteristic_name].index[0]
            filtered_samples = data.columns.intersection(annotations.columns[annotations.iloc[index_of_characteristic].astype(str) == str(desired_value)])
            subset_df_one_value = data[filtered_samples]
            subset_df_list.append(subset_df_one_value)

        subset_df = pd.concat(subset_df_list, axis=1)
        subset_df = subset_df.loc[:,~subset_df.columns.duplicated()].copy()

    except:
        index_of_characteristic = annotations.loc[annotations['Sample_title'] == characteristic_name].index[0]
        filtered_samples = data.columns.intersection(annotations.columns[annotations.iloc[index_of_characteristic].astype(str) == str(desired_values)])
        subset_df = data[filtered_samples] 

    subset_df.insert(loc=0, column='Gene', value=data['Gene'])
    print(subset_df)
    return subset_df



df = pd.read_pickle('..\\..\\data\\processed\\GSE114065_processed_RNAseq.pkl')
annotation_df = pd.read_pickle('..\\..\\data\\processed\\GSE114065_series_matrix.pkl')

subset_df = filter_samples(df, annotation_df, 'Sample_characteristics_ch1_age_yrs', [2, 4])