# Semantic Segmentation Project
The goals / steps of this project are the following:

* Reuse a pre-trained VGG neural network for extracting low-level image features.
* Build a Fully Convolutional Network (FCN) to upsample the extracted features for classifying every pixel of an image.
* Train the FCN with road images.
* Callsify road pixels of sample images and videos (optional).

## [Rubric](https://review.udacity.com/#!/rubrics/989/view) Points
#### Here I consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Build the Neural Network

#### 1. Does the project load the pretrained vgg model?

Yes, see `load_vgg` function in `main.py`.

#### 2. Does the project learn the correct features from the images?

Yes, see `layers` function in `main.py`. The network architecture is inspired by paper [Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) by Jonathan Long et al.




### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder
 
 ## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).
