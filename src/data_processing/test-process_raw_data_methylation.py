# https://geoparse.readthedocs.io/en/latest/usage.html
# run this file in jupyter/interactive mode (shift+enter) instead of all at once (cmd+enter)

import GEOparse
gse = GEOparse.get_GEO(filepath='..\\..\\data\\raw\\GSE114135_family.soft.gz')
# using the downloaded file gives empty tables rn

#gse = GEOparse.get_GEO(geo="GSE6207", destdir="./")
#gse = GEOparse.get_GEO(filepath='C:\\Users\\myrad\\Downloads\\GSE6207_family.soft.gz')
#gse = GEOparse.get_GEO(geo="GSE114135", destdir="./")

print()
print("GSM example:")
for gsm_name, gsm in gse.gsms.items():
    print("Name: ", gsm_name)
    print("Metadata:",)
    for key, value in gsm.metadata.items():
        print(" - %s : %s" % (key, ", ".join(value)))
    print ("Table data:",)
    print (gsm.table.head())
    break

print()
print("GPL example:")
for gpl_name, gpl in gse.gpls.items():
    print("Name: ", gpl_name)
    print("Metadata:",)
    for key, value in gpl.metadata.items():
        print(" - %s : %s" % (key, ", ".join(value)))
    print("Table data:",)
    print(gpl.table.head())
    break


# annotation info
gse.phenotype_data
gse.pivot_samples(values, index='ID_REF')

print(gse.pivot_and_annotate())
print(gse.gsms['GSM143385'].table)
print(gse.table)


RNAseq_df = pd.read_csv('C:\\Users\\myrad\\Downloads\\GSE114134_matrix_signal_intensities (3).csv.gz', sep='\t', header=0) # for github
import pandas as pd