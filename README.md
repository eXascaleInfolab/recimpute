# RecImpute: A Recommender System for Imputation Techniques in Time Series Data

RecImpute is a recommendation system of imputation techniques for missing values in time series. It uses data mining and deep learning techniques to learn knowledge about the time series' characteristics. The system can be trained on custom datasets or used as-is with no prior configuration required. RecImpute is able to predict which algorithm is the most-fitted to reconstruct missing parts of a real-world time series. Technical details can be found in our paper: <a href="/">TODO</a>.

Our system uses:
- the popular recovery of missing values benchmark called ImputeBench presented by Khayati et al. (<a href="http://www.vldb.org/pvldb/vol13/p768-khayati.pdf">Mind the Gap: An Experimental Evaluation of Imputation of Missing Values Techniques in Time Series</a>).
- the efficient time series shape-based clustering method called k-Shape presented by Paparrizos, John and Luis Gravano (<a href="http://www1.cs.columbia.edu/~jopa/Papers/PaparrizosSIGMOD2015.pdf">k-Shape: Efficient and Accurate Clustering of Time Series</a>).
- the T-Daub strategy for efficient models training presented by Shah et al. (<a href="https://arxiv.org/pdf/2102.12347.pdf">AutoAI-TS: AutoAI for Time Series Forecasting</a>).

[**Prerequisites**](#prerequisites) | [**Training**](#training) | [**Execution**](#execution) | [**Extension**](#extension)  | [**Contributors**](#contributors) | [**Citation**](#citation)


___

## Prerequisites
- Clone this repository.
- Install all Python modules listed in recimpute/requirements.txt
    1. Move to the project's repository: `cd recimpute/`
    2. Create a Python virtual environment if you want to avoid installing the modules in the Python's system directories: `python3.8 -m venv venv`
    3. Activate your virtual environment: on Linux: `source venv/bin/activate` or on Windows: `\venv\Scripts\activate`
    4. Update PIP: `python3.8 -m pip install --upgrade pip`
    5. Install the requirements (if you are using another system than Linux, some additional modules may have to be installed, please refer to the outputs of this command): `pip install -r requirements.txt`
- Clone and setup the <a href="https://github.com/eXascaleInfolab/bench-vldb20/blob/master/README.md">ImputeBench repository</a> (follow their Prerequisites section). Once installed, specify the benchmark's path (up to the Debug folder) in the "Config/imputebenchlabeler_config.yaml" (variable "BENCHMARK_PATH").


___

## Execution

```bash
    $ cd recimpute/
    $ python recimpute.py [arguments]
```

### Arguments

- *-mode*: Specifies the program's execution mode. Either train and evaluate one or mulitple models, or use a pre-trained model to get recommendations for new time series. Expected: *train* or *use*.

Note: many parameters and strategies can be set from the configuration files stored in the Config/ repository.

#### In *train* mode:

 | -lbl<sup> (\*)</sup> | -truelbl | -fes<sup> (\*)</sup> | -models<sup> (\*)</sup> | 
 | ----------- | ----------- | ------- | ---------------------------------- |
 | ImputeBench | ImputeBench | TSFresh | kneighbors                         |
 | KiviatRules | KiviatRules | Kiviat  | maxabsscaler_catboostclassifier    |
 |             |             | *all*   | normalizer_randomforest            |
 |             |             |         | standardscaler_randomforest        |
 |             |             |         | standardscaler_svc                 |
 |             |             |         | *all*                              |

- *-lbl*: Name of the labeler used to label the time series. Expected: one labeler name.
- *-true_lbl* (optional): Name of the labeler used to label the time series of the test set only. If not specified, uses the labeler specified with the -lbl argument. Expected: one labeler name.
- *-fes*: Name of the features' extracter(s) to use to create time series' feature vectors. Expected: one or multiple values separated by commas.
- *-models*: Name of the models' descriptions files (without their .py extension) to instantiate, train and evaluate (files from the Training/ModelsDescription/ folder). Expected: one or multiple values separated by commas.
- *-train_on_all_data* (optional): Whether or not train the models on ALL data after their evaluation is complete. If not specified, trains on all data after evaluation. Expected: *True* or *False*.

#### In *use* mode:

- *-id*: Identifier of the save containing the model to use. The saves are stored in the Training/Results/ folder. The id of save is its file name (without its .zip extension). Expected: one identifier. Example: *0211_1723_53480*.
- *-model*: Name of the model to load and use. A model's name is the same as its description file. Expected: one model name. Example: *maxabsscaler_catboostclassifier*.
- *-ts*: File name of the file containing the time series for which recommendations are wanted. Expected: one file name. Example: *timeseries.csv*.
The sequence(s) are saved to a text (.csv, .txt) file in the Datasets/SystemInputs/ folder. The sequence(s) should have been preemptively z-normalized. In the file, each row corresponds to one time-series and each value is separated by a space. The file should have no header and no index.
- *-use_prod_model* (optional): Whether or not use the model trained on ALL data. If not specified, does not use the model trained on all data (since it may not exist depending on the arguments used for training). Expected: *True* or *False*.

### Results
TODO

### Execution examples

#### Training and evaluation

1. Train and evaluate the *kneighbors* model. All time series (train, validation, and test sets) are labeled with the *ImputeBench* labeler and the features are extracted using the *TSFresh* extracter. Once evaluated the model is trained on all data.
```bash
    $ python recimpute.py -mode train -lbl ImputeBench -fes TSFresh -models kneighbors -train_on_all_data True
```

2. Train and evaluate the *kneighbors* and *standardscaler_svc* models. The time series from the train and validation sets are labeled with the *KiviatRules* labeler and those from the test sets are labeled with the *ImputeBench* labeler. The features are extracted using the *TSFresh* and *Kiviat* extracters. Once evaluated the models are not trained on all data.
```bash
    $ python recimpute.py -mode train -lbl KiviatRules -true_lbl ImputeBench -fes TSFresh,Kiviat -models kneighbors,standardscaler_svc -train_on_all_data False
```

3. Train and evaluate all models. All time series (train, validation, and test sets) are labeled with the *ImputeBench* labeler and the features are extracted using all extracters available. Once evaluated the models are trained on all data.
```bash
    $ python recimpute.py -mode train -lbl ImputeBench -fes all -models all -train_on_all_data True
```

#### Usage

1. Use the trained *kneighbors* model (which was trained on all data) saved in the *0211_1723_53480* results' archive file. Time series to get recommendations for are stored in the Datasets/SystemInputs/my_timeseries.csv file.
```bash
    $ python recimpute.py -mode use -id 0411_1456_53480 -model kneighbors -ts my_timeseries.csv -use_prod_model True
```


### Parametrized execution
TODO


___

## Extension
- To train models on new data sets:
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
    - Copy paste the Training/ModelsDescription/_template.py file and rename it with the name of your new model.
    - Fill in all the required info (check other ModelsDescription files if you need examples).
    - TEMPORARY: add your model to the "models_descriptions_to_use" list defined in the "run_all" method in the recimpute.py file.
- To get recommendations for any new time series:
    - TODO
    - Save the sequence(s) to a text (.csv, .txt) file in the Datasets/SystemInputs/ directory. The sequence(s) should have been z-normalized. In the file, each row corresponds to one time-series and each value is separated by a space. The file should have no header and no index.
    - TEMPORARY: in recimpute.py set the 'ts_filename' value to the name of your file.


___

## Contributors
Mourad Khayati (<a href="mkhayati@exascale.info">mkhayati@exascale.info</a>) and Guillaume Chacun (<a href="chacungu@gmail.com">chacungu@gmail.com</a>).


___

## Citation
TODO
