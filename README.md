# RecImpute: A Recommender System for Imputation Techniques in Time Series Data

RecImpute is a recommendation system of imputation techniques for missing values in time series. The system can be trained on custom datasets or used as-is with no prior configuration required. RecImpute is able to predict the most-suitable algorithm to reconstruct missing parts of a real-world time series. Technical details can be found in our paper: <a href="/">TODO</a>.



[**Prerequisites**](#prerequisites) | [**Build**](#build) |  [**Execution**](#execution) | [**Extension**](#new-time-series)  | [**Contributors**](#contributors) | [**Citation**](#citation)


___

## Prerequisites
- Ubuntu 18 (including Ubuntu derivatives, e.g., Xubuntu) or the same distribution under WSL.
- A server with a least 128GB of RAM
- Clone this repository into the folder named `recimpute/`.

- Clone and setup the <a href="https://github.com/eXascaleInfolab/bench-vldb20/blob/master/README.md">ImputeBench repository</a> (follow their Prerequisites + Build section). Once installed, specify the benchmark's path (up to the Debug folder) in the "Config/imputebenchlabeler_config.yaml" (variable "BENCHMARK_PATH").



___

## Build

```bash
    $ cd recimpute/
    $ sh install_script.sh
```

___

## Execution

### Train the model


```bash
    $ source venv/bin/activate
    $ python recimpute.py -mode cluster
    $ python recimpute.py -mode label
    $ python recimpute.py -mode extract_features -fes all
    $ python recimpute.py -mode train -fes all -train_on_all_data False
```

The last command of the training step will output an `id` which should be used in the next part. 


### Use the model
To use the model, please replace `id_from_the_train_command` with the corresponding `id`

```bash
    $ python recimpute.py -mode eval -model_id -1 -id id_from_the_train_command
```

Users can apply the trained model on an example time series (my_timeseries.csv) using the following command:

```bash
    $ python recimpute.py -mode use -model_id -1 -id id_from_the_train_command -ts my_timeseries.csv -use_prod_model False
```

 The results of the model will be stored as `my_timeseries__recommendations.csv` under `Datasets/Recommendations/`
 
### Comparison with baselines

The results of all baselines reported in the paper can be reproduced using the scripts located in the folder `Experiments/`

___

## Extension

- To train the model on a new dataset (it is recommended to z-normalize the time series)
    -  The time series file can have as extension either .txt or .csv.  Each column is a time series. No headers. Delimiters is single space. 
      - If the first column contains only date time objects, it will be used as index.
      - If the first column cannot be used as index, the archive can either contain:
            - an .index file containing a single column with the data set's index.
            - an .info file containing a header ("start periods freq") and the related information (e.g."'1900-01-01 00:00:00' 24 H").
    - The dataset must be stored as a zip file in the ./Datasets/RealWorld/ directory
     - The archive name should have the name of the dataset (e.g. "ArrowHead.zip").
     - Each file inside the archive must contain the datasets' name (e.g. "ArrowHead.info"). 
     - By default, all data sets listed in the ./Datasets/RealWorld/ directory are loaded and used. To change this setup, modify the Config/datasets_config.yaml. If you only want to run the system on a subset of datasets, switch the "USE_ALL" setting to False and list the name of the data set to use in the "USE_LIST" setting.
- To add new classifiers or pre-processing steps in the search space of ModelRace:
    - Open the Config/pipelines_steps_params.py file.
    - Add your classifier. You can also specify the range of values that should be considered for parameters.
- To get recommendations for any new time series:
    - Save the sequence(s) as a flat file (.csv, .txt) in the Datasets/SystemInputs/ directory. The sequence(s) should be z-normalized. Each row corresponds to one time series and values are separated by a space. The file should have no header and no index.
    - See the section about using the system to find the command to run.

___

## Full documentation

```bash
    $ python recimpute.py -mode [arguments]
```

### Arguments

- `cluster`: Cluster the datasets' time series. All datasets listed in the configuration files will be clustered. This step is required for the labeling and training.
- `label`: Assign a label to each datasets' cluster. This step is required for the training.
- `extract_features`: Extract the features of each datasets' time series. This step is required for the training.
    - *-fes*: Name of the features' extractor(s) to use to create time series' feature vectors. Expected value: one or multiple values separated by commas (TSFresh, Topological, Catch22, Kats, all).
- `train`: Select the most promising data preprocessing steps, classifiers and their hyperparameters, then train them on the previously labeled time series and their previously extracted features:
    - *-fes*: Name of the features' extractor(s) to use to create time series' feature vectors. Expected value: one or multiple values separated by commas (TSFresh, Topological, Catch22, Kats, all).
    - *-train_on_all_data* (optional): Whether or not train the models on ALL data. If not specified, trains on all data. Expected value: *True* or *False*. 
    - Warning: a model trained on all data should only be used in production and shouldn't be evaluated on the test set anymore since these data samples will have been used for training.
- `eval`: Evaluate trained models:
    - *-id*: Identifier of the save containing the models to evaluate. The saves are stored in the Training/Results/ folder. The id of a save is its file name (without its .zip extension). Expected value: one identifier. Example: *0211_1723_53480*.
    - *-model_id* (optional): ID of the model to load and evaluate. If specified, only this model will be evaluated, otherwise, all models will be. The models' ID are listed in the outputs of the *train* modes. If set to -1, the model evaluated will always be the Voting Classifier that combines the knowledge of all the other classifiers. Expected value: one model ID. Example: *745*.
- `use`: Use a trained model to get recommendations for new time series.
    - *-id*: Identifier of the save containing the model to use. The saves are stored in the Training/Results/ folder. The id of a save is its file name (without its .zip extension). Expected value: one identifier. Example: *0211_1723_53480*.
    - *-model_id*: ID of the model to load and use. The models' ID are listed in the outputs of the *train* and *eval* modes. If set to -1, the model used will always be the Voting Classifier that combines the knowledge of all the other classifiers. Expected value: one model ID. Example: *745*.
    - *-ts*: File name of the file containing the time series for which recommendations are wanted. Expected value: one file name. Example: *timeseries.csv*.
The sequence(s) are saved to a text (.csv, .txt) file in the Datasets/SystemInputs/ folder. The sequence(s) should have been preemptively z-normalized. In the file, each row corresponds to one time-series and each value is separated by a space. The file should have no header and no index.
    - *-use_prod_model* (optional): Whether or not use the model trained on ALL data. If not specified, does not use the model trained on all data (since it may not exist depending on the arguments used for training). Expected value: *True* or *False*.
    - Note: after using ModelRace to select the most-promising classifiers, the remaining ones are combined in a Voting Classifier that uses majority voting. This classifier will usually outperform the individual models. Hence we recommend using this Voting Classifier which *model_id*'s -1.

Note: The parameters and strategies can be modified in the configuration files stored in the Config/ repository.
___

## Citation
TODO
