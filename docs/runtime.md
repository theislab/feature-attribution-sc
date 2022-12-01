# Runtime of feature attribution methods

Feature attribution methods don't only strongly differ in the output of the most important features, but also in run time.
Hence, the selection of the feature attribution method of choice should depend on the model at hand and any potential limitations of computational resources.

In the following sections we will briefly discuss the complexity of the respective feature attribution methods and provide further intuition by describing empirical run time results of common models.

## Feature ablation

Test me math $x^2$

The complexity of feature ablation is henceforth: O(n * m²) where n == number of observations and m == number of features

## Permutation FA

The complexity of permutation feature attribution is therefore: O(n * m²) where n == number of observations and m == number of features

## SHAP

The complexity of SHAP is: O(????)

## LIME

The model complexity of LIME is O(n * m * z) where n == number of observations, m == number of features, z == complexity of surrogate model

## Integrated gradients

The complexity of  O(????)


## Expected gradients

O(????)


## Example models, example runtime measurements and expected runtime

TODO
