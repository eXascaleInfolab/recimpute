# Experiments

Important: all experiments have Python dependencies. To install those, please refer to the main README file of this repository. It is recommended to use the same Python environment as the one you use to run the whole system.

Note: for all Python script executions we recommend redirecting the outputs to a file to obtain better readability of the results.
For example: `python -u features_analysis_1.py 2>&1 | tee Experiments/results/features_analysis_1_results.txt


## Simplified A-Darts execution

Running this code will give you the performance of A-Darts reported in tables 1 and 2 of the paper.

```
python adarts_generate_trainingset.py
python adarts_training_and_eval.py
```

Results will be printed on the command line.

## Clustering comparison

Running this code will give you the performance of various clustering techniques that use k-Shape. Those were used to create figures 5.a and 5.b of the paper.

The code is in the Jupyter Notebook named `clustering_comparisons.ipynb`. Please refer to the [Jupyter Notebook documentation](https://docs.jupyter.org/en/latest/running.html) if you need help opening and executing the code of a Jupyter Notebook.

Plots and results will appear directly in the Notebook.

## Features' analysis

### Features' extractors comparison

Running this code will give you the performance of A-Darts when using varying features extractor. Those were used to create figure 6.a of the paper.

```
python features_analysis_1.py
```

Results will be printed on the command line.

### Explanation of classifiers' decision process

Running this code will give you insights into the inner working of the classifiers trained by A-Dart. The SHAP plot of figure 6.b in the paper can be generated in this Notebook.

The code is in the Jupyter Notebook named `features_analysis_2.ipynb`. Please refer to the [Jupyter Notebook documentation](https://docs.jupyter.org/en/latest/running.html) if you need help opening and executing the code of a Jupyter Notebook.

Plots and results will appear directly in the Notebook.

## ModelRace experiments

### Comparison of classifiers' performance with and without model selection

Running this code will give you the classifiers' performance with and without model selection. Those were used to create figure 3.a of the paper.

```
python modelrace_accuracy.py
```

Results will be printed on the command line.

### Varying number of sampled pipelines to show pruning efficiency

Running this code will give you the number of pipelines pruned when varying the number of sampled pipelines. Those values were used to create figure 3.b of the paper.

```
python modelrace_accuracy.py
```

Results will be printed on the command line. 
- The number of selected pipelines can be counted by looking for the "Finished models' selection with X remaining candidates." prints.
- The number of pruned pipelines can be counted by looking for the "X have been eliminated by t-test." prints.
- The number of pipelines eliminated early can be counted by looking for the "X pipelines' training have been stopped prematurely due to poor performances." prints.
- The number of "last minute" eliminations can be counted by looking for the "X was eliminated due to significantly worse performances than another candidate." prints.

### Impact of the score function's parameters

Running this code will give you the classifiers' performance when varying the parameters' values of the score function. Those were used to create figures 4.a, 4.b, and 4.c of the paper.

```
python modelrace_score_parameters.py
```

Results will be printed on the command line.

##  Baselines

### AutoAI-TS

Running this code will give you the performance of the AutoAI-TS baseline reported in table 1 of the paper.

The code is in the Jupyter Notebook named `baseline - AutoAI-TS.ipynb`. Please refer to the [Jupyter Notebook documentation](https://docs.jupyter.org/en/latest/running.html) if you need help opening and executing the code of a Jupyter Notebook.

Plots and results will appear directly in the Notebook.

### Kiviat

Running this code will give you the performance of the Kiviat baseline reported in table 1 of the paper.

The code is in the Jupyter Notebook named `baseline - Kiviat.ipynb`. Please refer to the [Jupyter Notebook documentation](https://docs.jupyter.org/en/latest/running.html) if you need help opening and executing the code of a Jupyter Notebook.

Plots and results will appear directly in the Notebook.

### RAHA

Running this code will give you the performance of the RAHA baseline reported in table 2 of the paper.

The code is in the Jupyter Notebook named `baseline - RAHA.ipynb`. Please refer to the [Jupyter Notebook documentation](https://docs.jupyter.org/en/latest/running.html) if you need help opening and executing the code of a Jupyter Notebook.

Plots and results will appear directly in the Notebook.

### AutoFolio

Running this code will give you the performance of the AutoFolio baseline reported in table 2 of the paper.

The code is in the Jupyter Notebook named `baseline - AutoFolio.ipynb`. Please refer to the [Jupyter Notebook documentation](https://docs.jupyter.org/en/latest/running.html) if you need help opening and executing the code of a Jupyter Notebook.

You will also need to clone the [AutoFolio's GitHub repository](https://github.com/automl/AutoFolio) inside the Experiments folder and follow the instruction listed in their [README](https://github.com/automl/AutoFolio#installation) to install the needed dependencies.

Plots and results will appear directly in the Notebook.