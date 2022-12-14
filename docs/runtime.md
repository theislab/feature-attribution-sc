<<<<<<< HEAD

# Runtime of feature attribution methods

=======

# On the runtime of feature attribution methods

> > > > > > > main

Feature attribution methods don't only strongly differ in the output of the most important features, but also in run time.
Hence, the selection of the feature attribution method of choice should depend on the model at hand and any potential limitations of computational resources.

In the following sections we will briefly discuss the complexity of the respective feature attribution methods and provide further intuition by describing empirical run time results of common models.

## Feature ablation

<<<<<<< HEAD
Test me math $x^2$
=======
The general approach of feature ablation can be summarized as follows:

1. Train the model on a train set and calculate a score on the test set using any metric.
2. For each of the m features, remove it from the training data and train the model. Then, calculate the score on the test set.
3. Rank the features by the difference between the original score (from the model with all features) and the score for the model using all features but one.

Reference: https://www.samueltaylor.org/articles/feature-importance-for-any-model.html

> > > > > > > main

The complexity of feature ablation is henceforth: O(n \* m²) where n == number of observations and m == number of features

## Permutation FA

# <<<<<<< HEAD

Permutation feature importance measures the increase in the prediction error of the model after we permuted the feature’s values, which breaks the relationship between the feature and the true outcome. This is done by measuring the importance of a feature by calculating the increase in the model’s prediction error after permuting the feature. A feature is “important” if shuffling its values increases the model error, because in this case the model relied on the feature for the prediction. A feature is “unimportant” if shuffling its values leaves the model error unchanged, because in this case the model ignored the feature for the prediction.

Reference: https://christophm.github.io/interpretable-ml-book/feature-importance.html

> > > > > > > main
> > > > > > > The complexity of permutation feature attribution is therefore: O(n \* m²) where n == number of observations and m == number of features

## SHAP

# <<<<<<< HEAD

The goal of SHAP is to explain the prediction of a data point by computing the contribution of each feature to the prediction. The SHAP explanation method computes Shapley values from coalitional game theory. The feature values of a data instance act as players in a coalition. Shapley values tell us how to fairly distribute the “payout” (= the prediction) among the features. A player can be an individual feature value, e.g. for tabular data. A player can also be a group of feature values. For example to explain an image, pixels can be grouped to superpixels and the prediction distributed among them. One innovation that SHAP brings to the table is that the Shapley value explanation is represented as an additive feature attribution method, a linear model.

Reference: https://christophm.github.io/interpretable-ml-book/shap.html

> > > > > > > main
> > > > > > > The complexity of SHAP is: O(????)

## LIME

# <<<<<<< HEAD

Local surrogate models are interpretable models that are used to explain individual predictions of black box machine learning models. Local interpretable model-agnostic explanations (LIME) is a paper in which the authors propose a concrete implementation of local surrogate models. Surrogate models are trained to approximate the predictions of the underlying black box model. Instead of training a global surrogate model, LIME focuses on training local surrogate models to explain individual predictions.

1. Select your instance of interest for which you want to have an explanation of its black box prediction.
2. Perturb your dataset and get the black box predictions for these new points.
3. Weight the new samples according to their proximity to the instance of interest.
4. Train a weighted, interpretable model on the dataset with the variations.
5. Explain the prediction by interpreting the local model.

Reference: https://christophm.github.io/interpretable-ml-book/lime.html

> > > > > > > main
> > > > > > > The model complexity of LIME is O(n _ m _ z) where n == number of observations, m == number of features, z == complexity of surrogate model

## Integrated gradients

<<<<<<< HEAD
The complexity of O(????)
=======
The determination of feature attribution using integrated gradients relies on two primary features:

1. Sensitivity: If an input feature changes the classification score in any way, this input should have an attribution value not equal to 0.
2. Implementation invariance: The result of the attribution should not depend on the design and structure of the neural network. Thus, if two different neural networks provide the same prediction for the same inputs, their attribution values should also be identical.

Reference: https://databasecamp.de/en/ml/integrated-gradients-nlp

The complexity of
O(????)

> > > > > > > main

## Expected gradients

# <<<<<<< HEAD

Similar to **Integrated Gradients**, but samples baseline values from data instead of fixing one value.

> > > > > > > main
> > > > > > > O(????)

## Example models, example runtime measurements and expected runtime

TODO
