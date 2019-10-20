# Human Binary Classification Suite
 
Hello! Thank you for visiting this repo!

This repo contains almost everything you need in getting started for Human Binary classification

## Table of Contents

   * [Human Binary Classification Suite](#human-binary-classification-suite)
      * [Dependencies](#dependencies)
      * [Dataset Creation](#dataset-creation)
      * [Premade Dataset](#premade-dataset)
      * [Model Training &amp; Visualization](#model-training--visualization)
      * [Pretrained Models](#pretrained-models)
      * [Detection Engine](#detection-engine)

## Dependencies
These are packages that is tested in my system to run the suite:
* Python 3.6.9
* OpenCV 3.4.7
* Numpy 1.17.2
* Matplotlib 3.1.1
* Tensorflow 1.10.0
* h5py 2.10.0
* scipy 1.3.1
* scikit-learn 0.21.3
* pydot 1.4.1
* graphviz 2.40.1

For more information of installed packages in my system, refer to the `packages.txt`

## Dataset Creation
`dataset_generator` folder contains basic tool for creating the dataset. 

The dataset is a binary class data, consisting of human region-of-interest images and non-human images. The images varies in size, with smallest being 25 X 25 px.

The ROI images are retrieved from a foreground mask by using contour detection to make a bounding rectangle for cropping the original frame. The foreground mask is made by using background subtraction technique using Mixture of Gaussian (MOS)

Please refer to the included `_INSTRUCTIONS.txt` for steps & information.

## Premade Dataset
`dataset` folder contains two subfolders representing two classes: human and nonhuman. There are 1229 human ROI images and 7126 nonhuman ROI images, retrieved from [VIRAT 2.0 dataset](http://www.viratdata.org/) using the dataset generator in this repo.

Huge thanks to Afriezal Lienardi and Vincenza Steven Ber for creating and doing manual labeling of the dataset!

## Model Training & Visualization
In the root of the directory, there are three python files, run them in order:
* `dataset_pickler.py` : Packs the dataset images in the `dataset` directory into separate training data and test data and saves it as a pickle file (`train_data.pkl` & `test_data.pkl`) for convenience.
* `model_maker.py` : Makes pretrained model using Tensorflow and Keras. It saves the model(s) as `.h5` file with an image of its architecture and loss & acc graph in a folder named from datetime in the `models` directory. Feel free to experiment with this file!
* `model_loader.py` : Loads pretrained `.h5` model for evaluation and visualization.

## Pretrained Models
`models` directory contains several pretrained models, some are included with test data. You can use these pretrained model for anything you like! Make sure to read the architecture.

## Detection Engine
`Detection_Engine` folder contains ready-to-use suite of files, models, and sample videos for human detection systems.

Run `demo.py` or `HumanDetectionVisualization.py` for demonstration, or see each source code for more in-depth analysis.

Featuring `HumanDetector.py` used in the [TREYESpassing project](https://github.com/mstrassassin1st/TREYESpassing/tree/master). TREYESpassing is an app that alerts it's user when someone entered an area that is owned by the user and access to the area is prohibited for everyone. 