# Propulsion Academy Final Project - MRI Classifier

## Authors: Norbert Bräker, Cornelia Schmitz

### July 2020

## Notebooks Directory

This notebook creates a jsonfile needed by the other notebooks and therefore it should be run in advance:

* `create_json.ipynb`: creates a jsonfile containing information about the images of the data used like `patient-id`, `image-name`, and labels for `perspective`, `sequence` and these combined and too an `image-number` for counting the images taken from the same original DCIM image sequence.

The following notebooks depend on the created jsonfile but they can be run independently from each other.
All these contain data loading, data sampling, image augmentation, model definition and training, model evaluation with validation data and model predicting with test data

* `Transfer_Learning_ResNet.ipnb`           Transferlearnin (uses `perspective`label by default, changeable)
* `Transfer_Learning_ResNet_GradCAM.ipnb`   Transferlearning (uses `sequence`label by default, changeable, contains additionally GradCAM heatmap)
* `DualOutput_Transfer_Learning_ResNet.ipnb` Transferlearning (uses combined `perspective-sequence` label) by default and model outputs predictions for both labels)
* `Bayesian_Transfer_Learning_ResNet.ipnb`  Transferlearning (uses `sequence`label by default, additionally it contains an uncertainty evaluation using Bayesian theory)
