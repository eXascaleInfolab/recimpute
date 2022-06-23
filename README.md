# RecImpute: A Recommender System for Imputation Techniques in Time Series Data

RecImpute is a recommendation system of imputation techniques for missing values in time series. The system can be trained on custom datasets or used as-is with no prior configuration required. RecImpute is able to predict the most-suitable algorithm to reconstruct missing parts of a real-world time series. Technical details can be found in our paper: <a href="/">TODO</a>.



[**Prerequisites**](#prerequisites) | [**Training**](#training) | [**Execution**](#execution) | [**Extension**](#extension)  | [**Contributors**](#contributors) | [**Citation**](#citation)


___

## Prerequisites
- Clone this repository.
- Clone and setup the <a href="https://github.com/eXascaleInfolab/bench-vldb20/blob/master/README.md">ImputeBench repository</a> (follow their Prerequisites + Build section). Once installed, specify the benchmark's path (up to the Debug folder) in the "Config/imputebenchlabeler_config.yaml" (variable "BENCHMARK_PATH").
- Move to the project's repository: `cd recimpute/`
- Run the install script: `sh install_script.txt`


___

## Execution

Run all experiments:
```bash
    $ cd recimpute/
    $ source venv/bin/activate
    $ python recimpute.py -mode cluster
    $ python recimpute.py -mode label
    $ python recimpute.py -mode extract_features -fes all
    $ python recimpute.py -mode train -fes all -train_on_all_data False
```

The last command will output an `id` which should then be used in the next one:

```bash
    $ python recimpute.py -mode eval -model_id -1 -id id_from_the_train_command
```

___

## Full documentation

```bash
    $ cd recimpute/
    $ source venv/bin/activate
    $ python recimpute.py [arguments]
```

### Arguments

- *-mode*: Specifies which step of the system should be executed:
    - `cluster`: Cluster the datasets' time series;
    - `label`: Assign a label to each datasets' cluster;
    - `extract_features`: Extract the features of each datasets' time series;
    - `train`: Select the most promising data preprocessing steps, classifiers and their hyperparameters, then train them on the previously labeled time series and their previously extracted features;
    - `eval`: Evaluate trained models;
    - `use`: Use a trained model to get recommendations for new time series.

Note: many parameters and strategies can be set from the configuration files stored in the Config/ repository.

#### *cluster* mode:

Cluster the datasets. If the clusters have not been generated yet, this step is required to be executed before labeling and training.

No arguments. All data sets listed in the configuration files will be clustered.

#### *label* mode:

Labels the datasets' clusters. If the labels have not been attributed yet, this step is required to be executed before training.

<!-- | -lbl<sup> (\*)</sup> | -truelbl | -->
<!-- | ----------- | ----------- | -->
<!-- | ImputeBench | ImputeBench | -->
<!-- | KiviatRules | KiviatRules | -->
<!--  -->
 <!-- <sub>arguments marked with <sup>(\*)</sup> are mandatory</sub> -->
<!--  -->
<!-- - *-lbl*: Name of the labeler used to label the time series. Expected: one labeler name. -->
<!-- <!-- - *-true_lbl* (optional): Name of the labeler used to label the time series of the test set only. If not specified, uses the labeler specified with the -lbl argument. Expected: one labeler name. -->

#### *extract_features* mode:

Computes features for the datasets' time series. If the features have not been extracted yet, this step is required to be executed before training.

| -fes<sup> (\*)</sup> |
| ------------- |
| TSFresh       |
| Topological   |
| Catch22       |
| Kats          |
| *all*         |
<!-- | Kiviat        | -->

 <sub>arguments marked with <sup>(\*)</sup> are mandatory</sub>

- *-fes*: Name of the features' extractor(s) to use to create time series' feature vectors. Expected: one or multiple values separated by commas.


#### *train* mode:

Selects the most-promising pipelines and trains them on the prepared datasets (labeling and features extraction must be done prior to training).

 <!-- | -lbl<sup> (\*)</sup> | -truelbl | -fes<sup> (\*)</sup> | train_on_all_data |
 | ----------- | ----------- | ------------- | ----------------- |
 | KiviatRules | KiviatRules | Kiviat        |                   |
 | ImputeBench | ImputeBench | TSFresh       | True              |
 |             |             | Topological   | False             |
 |             |             | Catch22       |                   |
 |             |             | Kats          |                   |
 |             |             | *all*         |                   | -->

 | -fes<sup> (\*)</sup> | -train_on_all_data |
 | ------------- | ----------------- |
 | TSFresh       | True              |
 | Topological   | False             |
 | Catch22       |                   |
 | Kats          |                   |
 | *all*         |                   |

 <sub>arguments marked with <sup>(\*)</sup> are mandatory</sub>

<!-- - *-lbl*: Name of the labeler used to label the time series. Expected: one labeler name.
- *-true_lbl* (optional): Name of the labeler used to label the time series of the test set only. If not specified, uses the labeler specified with the -lbl argument. Expected: one labeler name. -->
- *-fes*: Name of the features' extractor(s) to use to create time series' feature vectors. Expected: one or multiple values separated by commas.
- *-train_on_all_data* (optional): Whether or not train the models on ALL data. If not specified, trains on all data. Expected: *True* or *False*. Warning: a model trained on all data should only be used in production and shouldn't be evaluated on the test set anymore since these data samples will have been used for training.

#### *eval* mode:

- *-id*: Identifier of the save containing the models to evaluate. The saves are stored in the Training/Results/ folder. The id of a save is its file name (without its .zip extension). Expected: one identifier. Example: *0211_1723_53480*.
- *-model_id* (optional): ID of the model to load and evaluate. If specified, only this model will be evaluated, otherwise, all models will be. The models' ID are listed in the outputs of the *train* modes. If set to -1, the model evaluated will always be the Voting Classifier that combines the knowledge of all the other classifiers. Expected: one model ID. Example: *745*.


#### *use* mode:

- *-id*: Identifier of the save containing the model to use. The saves are stored in the Training/Results/ folder. The id of a save is its file name (without its .zip extension). Expected: one identifier. Example: *0211_1723_53480*.
- *-model_id*: ID of the model to load and use. The models' ID are listed in the outputs of the *train* and *eval* modes. If set to -1, the model used will always be the Voting Classifier that combines the knowledge of all the other classifiers. Expected: one model ID. Example: *745*.
- *-ts*: File name of the file containing the time series for which recommendations are wanted. Expected: one file name. Example: *timeseries.csv*.
The sequence(s) are saved to a text (.csv, .txt) file in the Datasets/SystemInputs/ folder. The sequence(s) should have been preemptively z-normalized. In the file, each row corresponds to one time-series and each value is separated by a space. The file should have no header and no index.
- *-use_prod_model* (optional): Whether or not use the model trained on ALL data. If not specified, does not use the model trained on all data (since it may not exist depending on the arguments used for training). Expected: *True* or *False*.

Note: after using ModelRace to select the most-promising classifiers, the remaining ones are combined in a Voting Classifier that uses majority voting. This classifier will usually outperform the individual models. Hence we recommend using this Voting Classifier which *model_id*'s -1.

### Execution examples

#### Data sets' preparation

1. Cluster the data sets' time series.
```bash
    $ python recimpute.py -mode cluster
```
2. Label the data sets' clusters.
```bash
    $ python recimpute.py -mode label
```
3. Extract the data sets' time series features using the *TSFresh* and *Catch22* extractors.
```bash
    $ python recimpute.py -mode extract_features -fes TSFresh,Catch22
```

#### Training

1. Train the models selected by our ModelRace algorithm. All features' should be used. The models will be trained on all the data.
```bash
    $ python recimpute.py -mode train -lbl ImputeBench -fes all -train_on_all_data True
```

2. Train the models selected by our ModelRace algorithm. The features that should be used are *TSFresh*'s and *Catch22*'s. The models will not be trained on all the data.
```bash
    $ python recimpute.py -mode train -lbl KiviatRules -true_lbl ImputeBench -fes TSFresh,Catch22 -train_on_all_data False
```

#### Evaluation

1. Evaluate all models saved in the *0411_1456_53480*.zip results' archive file on their test set.
```bash
    $ python recimpute.py -mode eval -id 0411_1456_53480
```

#### Usage

1. Use the trained model #-1 (which is refering to the VotingClassifier that uses all the other classifiers). Since *use_prod_model* is set to True, it means this model was trained on all data. It is saved in the *0411_1456_53480*.zip results' archive file. Time series to get recommendations for are stored in the Datasets/SystemInputs/my_timeseries.csv file. The results will appear in the Datasets/Recommendations/my_timeseries__recommendations.csv file.
```bash
    $ python recimpute.py -mode use -id 0411_1456_53480 -model_id -1 -ts my_timeseries.csv -use_prod_model True
```


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
- To add new classifiers or pre-processing steps in the search space of ModelRace:
    - Open the Config/pipelines_steps_params.py file.
    - Add your classifier. For each parameter that should not use their default value, specify the range of values that should be considered..
- To get recommendations for any new time series:
    - Save the sequence(s) to a text (.csv, .txt) file in the Datasets/SystemInputs/ directory. The sequence(s) should have been z-normalized. In the file, each row corresponds to one time-series and each value is separated by a space. The file should have no header and no index.
    - See the section about using the system to find the command to run.


___

## Contributors
Mourad Khayati (<a href="mkhayati@exascale.info">mkhayati@exascale.info</a>) and Guillaume Chacun (<a href="chacungu@gmail.com">chacungu@gmail.com</a>).


___

## Citation
TODO
