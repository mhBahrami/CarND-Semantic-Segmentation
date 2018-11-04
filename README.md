# Semantic Segmentation
### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).



[//]: # "Image References"

[image1]: ./images/loss.png "Loss (%) over time"
[image2]: ./images/result.gif "Fast Preview of Labeled Images"



### Reflections. Rubric

#### Build the Neural Network

Because the implementation in [`main.py`] has passed all the test functions so it has met the criteria.

- The function `load_vgg` is implemented correctly.

- > From [line] to [line]. 

- The function `layers` is implemented correctly.

- > From [line] to [line]. 

- The function `optimize` is implemented correctly.

- > From [line] to [line]. 

- The function `train_nn` is implemented correctly.

- > From [line] to [line]. 

#### Neural Network Training

##### Does the project train the model correctly?

Yes, it does. you can see the loss (%) over time in the following figure:

| Loss (%) Over Time  |
| :-----------------: |
| ![alt text][image1] |

##### Does the project use reasonable hyperparameters?

I chose the following hyper parameters for the network and training:

```python
image_shape = (160, 576)
num_classes = 2
epochs = 20
batch_size = 8
keep_prob_value = 0.75
learning_rate_value = 1e-4
```

##### Does the project correctly label the road?

Yes, it does. You can find the labeled images in [./runs/1541284958.1358018]. Also I put a fast preview of them below:

| Fast Preview of Labeled Images |
| :----------------------------: |
|      ![alt text][image2]       |

### Discussion 

#### Network Architecture

The network architecture for the network is as follows:

```js
  [vgg3]     [vgg4]     [vgg7]
    |          |          |
    |          |          \--->[conv2d:1x1]---\
    |          |                              |
    | [scale] [x]                             V
    |  0.01    |                      [conv2d transpose] #1
    |          |                              |
    |          |                              V
    |          \--->[conv2d:1x1]------------>[+] [add] #1
   [x] [scale]                                |
    |  0.0001                                 V
    |                                 [conv2d transpose] #2
    |                                         |
    |                                         V
    \-------------->[conv2d:1x1]------------>[+] [add] #2
                                              |
                                              V
                                      [conv2d transpose] (output layer)
```

#### Dataset

I used the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip). Also I generated new jittered date and add them to the data set to have a better training data set. 

> I add them to `helper.py` (From [line] to [line]).

To generate the jittered images I used [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_setup/py_table_of_contents_setup/py_table_of_contents_setup.html#py-table-of-content-setup) to scale up, rotate CW/CCW, and flip them (From [line] to [line]). Also, I adjust the gamma of images (From [line] to [line]).

***It all helps to bring more variety to data set of images and improve road recognition at the end.***

> Note
>
> I used Udacity's classroom workspace. So, `data_dir = '/data'`. If you want to run it somewhere else make sure change it to `data_dir = './data'`.

### Setup

##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
 - [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_setup/py_table_of_contents_setup/py_table_of_contents_setup.html#py-table-of-content-setup).
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

 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip).
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [post](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/forum_archive/Semantic_Segmentation_advice.pdf) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.

### License
[MIT License].