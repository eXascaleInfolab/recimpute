# RecImpute: A Recommender System for Imputation Techniques in Time Series Data

RecImpute is a recommendation system of imputation techniques for missing values in time series. It uses data mining and deep learning techniques to learn knowledge about the time series' characteristics. The system can be trained on custom datasets or used as-is with no prior configuration required. RecImpute is able to predict which algorithm is the most-fitted to reconstruct missing parts of a real-world time series. Technical details can be found in our paper: <a href="/">TODO</a>.

Our system uses:
- the popular recovery of missing values benchmark called ImputeBench presented by Khayati et al. (<a href="http://www.vldb.org/pvldb/vol13/p768-khayati.pdf">Mind the Gap: An Experimental Evaluation of Imputation of Missing Values Techniques in Time Series</a>).
- the efficient time series shape-based clustering method called k-Shape presented by Paparrizos, John and Luis Gravano (<a href="http://www1.cs.columbia.edu/~jopa/Papers/PaparrizosSIGMOD2015.pdf">k-Shape: Efficient and Accurate Clustering of Time Series</a>).
- the T-Daub strategy for efficient models training presented by Shah et al. (<a href="https://arxiv.org/pdf/2102.12347.pdf">AutoAI-TS: AutoAI for Time Series Forecasting</a>).

[**Prerequisites**](#prerequisites) | [**Training**](#training) | [**Execution**](#execution) | [**Extension**](#extension)  | [**Contributors**](#contributors) | [**Citation**](#citation)


___

## Prerequisites
TODO


___


## Training
TODO


___

## Execution
TODO

### Arguments
TODO

### Results
TODO

### Execution examples
TODO

### Parametrized execution
TODO


___

## Extension
- To use new data sets:
    - Each data set must be stored in its own zip archive.
    - This archive must contain:
        - One .txt or .csv file containing the time series. Each column is a time series. No headers. Delimiters is a single space. If the first column contains only date time objects, it will be used as index.
    - If the first column cannot be used as index, the archive can either contain:
        - a .index file containing a single column with the data set's index.
        - a .info file containing a header ("start periods freq") and the related information (e.g."'1900-01-01 00:00:00' 24 H").
    - The archive name is the name of the data set (e.g. "ArrowHead.zip").
    - Each file inside the archive must contain the data sets' name (e.g. "ArrowHead.info").
    - The data sets' archive must be placed in the ./Datasets/RealWorld/ directory.
    - By default, all data sets listed in the ./Datasets/RealWorld/ directory are loaded and used. To change this behaviour, modify the Config/datasets_config.yaml. If you only want to run the system on a subset of data sets, switch the "USE_ALL" setting to False and list the name of the data set to use in the "USE_LIST" setting.
    - It is recommended to z-normalize the time series before-hand.
- To add and train new models:
    -  TODO


___

## Contributors
Mourad Khayati (<a href="mkhayati@exascale.info">mkhayati@exascale.info</a>) and Guillaume Chacun (<a href="chacungu@gmail.com">chacungu@gmail.com</a>).


___

## Citation
TODO
