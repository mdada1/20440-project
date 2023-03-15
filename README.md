### Overview

### Data

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
│   └───supp_fig
├───notebook
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
* **notebook/**: Jupyter notebooks containing exploratory data analysis
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

The file "Table1.png" will be generated in the directory "figs/main_figs".
