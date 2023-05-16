### Overview

This project analyzes the RNA sequencing and DNA methylation data produced by David Martino et al.<sup>1</sup> Our objective is to use data analysis methods and machine learning to identify gene signatures associated with egg allergy and predict the resolution of egg allergy in children.



<sup>1</sup> Martino, D. et al. (2018). Epigenetic dysregulation of naive CD4+ T-cell activation genes in childhood food allergy. Nature communications, 9(1), 3308. https://doi.org/10.1038/s41467-018-05608-4



### Data

Martino et al. generated two main datasets containing genome-wide DNA methylation and RNA sequencing data of naive CD4+ T cells from 44 subjects with egg allergies and 21 non-allergic controls. The T cells were either activated or quiescent. Samples were taken from the patients at infancy, and the patients were reanalyzed during early childhood, for a total of 135 transcriptional profiles.

The raw data are in the directory **\\data\raw\\**, or can alternatively be accessed on GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE114065.


### Folder Structure

The folder structure of this project is shown below.

```
project
│   README.md
│   requirements.txt
├───data
│   ├───processed
│   ├───raw
│   └───results
├───fig
│   ├───main_fig
│   ├───supp_fig
│   └───tables
└───src
    ├───analysis
    ├───data_processing
    ├───util
    └───visualization
```

The purpose of each folder is as follows:

* **data/processed/**: intermediate data during processing (ex. dataframes)
* **data/raw/**: raw data from the authors
* **data/results/**: our results
* **fig/main_fig/**: our main figures
* **fig/supp_fig/**: our supplementary figures
* * **fig/supp_fig/**: our tables
* **src/analysis/**: code for producing results
* **src/data_processing/**: code for loading and cleaning the authors' data
* **src/util/**: helper functions and resused scripts
* **src/visualization/**: code for plotting results


### Installation and Setup

To create a fresh virtual environment with all the necessary packages, create a virtual environment:

```
py -m venv env
env\Scripts\activate
```

Then navigate to the project directory and run:

```
pip install -r requirements.txt
```



### Running code

To create Table 1, navigate to the main project directory and run:

```
python .\src\data_processing\process_raw_data.py
python .\src\visualization\generate_table1.py
```

The file "Table1.png" will be generated in the directory "figs/main_figs". This table shows an overview of the patients the dataset was derived from.
