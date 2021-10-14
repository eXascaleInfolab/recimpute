"""
RecImpute - A Recommendation System of Imputation Techniques for Missing Values in Time Series,
eXascale Infolab, University of Fribourg, Switzerland
***
ModelsEvaluater.py
@author: @chacungu
"""

#import

from Training.RecommendationModel import RecommendationModel

class ModelsEvaluater:
    """
    Class which evaluates the performances of trained models.
    """


    # constructor

    def __init__(self, training_set, models, training_results):
        """
        Initializes a ModelsEvaluater object.

        Keyword arguments:
        training_set -- TrainingSet instance
        models -- list of RecommendationModels' instances
        training_results -- TrainingResults' instance
        """
        self.training_set = training_set
        self.models = models
        self.training_results = training_results
    

    # public methods

    def test_models(self):
        """
        Evaluates the models on the test set and prints their scores.

        Keyword arguments: -

        Return: -
        """
        print('\n\n\n')
        print('Models evaluation:')
        print('------------------')
        X_test, y_test, labels_set = self.training_set.get_test_set()

        for model in self.models:
            # load the best trained pipeline of this model (from the TrainingResults)
            model_results = self.training_results.get_model_best_split(model)
            trained_pipeline = model_results['Trained Pipeline']

            # evaluate the model
            scores, cm = RecommendationModel.eval(trained_pipeline, X_test.to_numpy(), y_test.to_numpy().astype('str'), 
                                                  self.training_set.get_labeler_properties(), labels_set, plot_cm=True)

            # plot the evaluation results
            print(model)
            print(scores)
            print()