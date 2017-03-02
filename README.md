# Clone driving behaviour using Deep Learning.

* [The Goal](#the-goal)
* [Code base](#code-base)
* [Model architecture and training strategy](#model-architecture-and-training-strategy)
* [Simulation](#simulation)
* [References](#references)

---
## The Goal

The goal of this project is to build a machine learning model that mimics 
a human driving a car. A driving simulator is provided, which will be used 
to gather training data, and as well, to test how the trained model performs.

The training dataset consist of images coming from camaras mounted on the car
( 3 in total: left, center and right cameras), and the steering angle 
at the moment the image is captured.

All this data is feeded to a model that learns the driving behaviour and predicts
the right steering angle of the vehicule given an image from the center.

### Project requirements
* The code in this repository.
* The simulator which is available for [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip), [macOS](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4594_mac-sim.app/mac-sim.app.zip), [Windows](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip)
* [Training data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)

---
## Code base

The code base is composed of the following files:

* model.py: All code used to define architecture of CNN and to train it.
* drive.py: Script that starts a server the simulator connects to, to make predictions.
* util.py: Helper code. Mainly image manipulation helpers.
* model.h5: HDF5 file that contains model architecture, weights of the model, training configuration and state of optimizer.
* model.json: JSON file specifying the model architecture.
* README.md: This file, the report.

---
## Model architecture

The base model architecture was inspired by the NVIDIA paper 
[End to End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf "NVIDIA paper")
which describes how a CNN is trained to map raw pixels from a single front-facing camera directly to steering angles.

The final CNN architecture is composed of the following layers (All layers marked *** were added or modified from base CNN):

* Input layer with shape (None, 160, 320, 3) *** : The images the simulator generates have dimensions: 160 x 320.
* Pre-processing layer for Cropping images *** : All images are cropped from top and bottom by 40 pixels each region, to keep only the area of interest.
* Pre-processing layer for Normalization (called Lamba in the image below): Each image is normalized.
* 3 strided convolutional layers with 2x2 stride and a 5x5 kernel.
* 2 non-strided convolutional layers with 3x3 kernel.
* Flatten layer.
* 3 fully connected layers with RELU activation followed by dropout of 50% ***.

We end up having **15 layers and 770,619 trainable parameters**.

The layers that do pre-processing in the CNN could be taken out of the network, they are not adjusted during the learning process,
but having them hard-coded in the network allows this kind of processing to take advante of GPU processing 
in case we train the network in such scenario.

The final CNN architecture used for the project is the following:

<p align="center">
 <img src="https://github.com/vguerra/behavioral-clonning/blob/master/imgs/model.png" width="300">
</p>

## Training strategy

### Training data
The dataset used for training is the images provided by Udacity, that you can find in the section [above](#project-requirements).

The first step we took was to investigate how the distribution of steering angles looks like:
<p align="center">
 <img src="https://github.com/vguerra/behavioral-clonning/blob/master/imgs/angle_dist_original.png" width="500">
</p>

We can see the big spike for steering angle 0. Training our model with the data as it is would bias it and not generalize
to other scenarios when driving. For example, how to steer to the center of the track when it starts to drive
close to the boarders of the road.

Thus it is crucial that we perform some data augmentation to try to equalize the histogram
of steering angles distribution, trying to achieve a more uniform-like distribution.

#### Data Agumentation

For data augmentation, a [data generator](https://github.com/vguerra/behavioral-clonning/blob/master/model.py#L83-L114) is used.
It takes as input metadata of camera images and performs several perturbations to them. The idea is to grow the amount
of images used for training keeping in mind that a better distribution of steering angles should be achieved.

A row of metadata includes an image from all 3 cameras and a steering angle. Taking a row randomly we can view the images for the 3 cameras:

<table border=0>
<tr><th align="center">Left camera</th><th align="center">Center camera</th><th align="center">Right camera</th></tr>
<tr>
<td>
<img src="https://github.com/vguerra/behavioral-clonning/blob/master/imgs/left.png" />
</td>
<td>
<img src="https://github.com/vguerra/behavioral-clonning/blob/master/imgs/original_0.0.png" />
</td>
<td>
<img src="https://github.com/vguerra/behavioral-clonning/blob/master/imgs/right.png" />
</td>
</tr>
</table>

In this case the steering angle has a value of 0.

To agument the training data we change properties of the images provided in the metadata and change correspondly the steering angles.

The [pipeline of augmenting data](https://github.com/vguerra/behavioral-clonning/blob/master/model.py#L66-L80) goes as follows:

1. For each row of metadata we randomly pick one image from all 3. If either of the side cameras is chosen, then the angle
is adjusted by a [STEERING factor](https://github.com/vguerra/behavioral-clonning/blob/master/model.py#L29).
 Ref: [Code](https://github.com/vguerra/behavioral-clonning/blob/master/model.py#L43-L63).

2. The picked image is afterwards [translated](https://github.com/vguerra/behavioral-clonning/blob/master/utils.py#L40-L51) vertically
and horizontally and it's angle is re-computed accordingly. This will provide the model with recovery data for when
the car starts to drive towards the boarders. An example of a translated image would be:

<p align="center">
<img src="https://github.com/vguerra/behavioral-clonning/blob/master/imgs/translate_with_angle_-0.34267774858030714.png" width="400">
</p>

The steering angle for this image was modified **from 0.0 to -0.3426** due to the transitions.

3. The [brightness](https://github.com/vguerra/behavioral-clonning/blob/master/utils.py#L53-L60) is as well modified with random factor.
The purpose of this perturbation is to help the model to learn under different lightning conditions. An example
of an image with modified brightness. 

<p align="center">
<img src="https://github.com/vguerra/behavioral-clonning/blob/master/imgs/brightness.png" width="400">
</p>

4. Next step is to [randomly mark a shadow region](https://github.com/vguerra/behavioral-clonning/blob/master/utils.py#L62-L91)
within the image. Objectcs in the road ( trees, mountains, etc ) generate shadows that project on the road so the model needs to be able to generalize for those scenarios as well. 

<p align="center">
<img src="https://github.com/vguerra/behavioral-clonning/blob/master/imgs/shadow.png" width="400">
</p>

5. Last perturbation is to [flip an image](https://github.com/vguerra/behavioral-clonning/blob/master/utils.py#L27-L32) w.r.t. y axis. The first track in the simulator lacks of left turns,
thus most images of turns are right turns. This is a technique that helps to produce more samples for left turns.

<p align="center">
<img src="https://github.com/vguerra/behavioral-clonning/blob/master/imgs/flipe_0.34267774858030714.png" width="400">
</p>

The steering angle is modified accordignly. For this specific example the resulting steering angle would be **0.3426**

There is a technique as well used in the data generator that helps tp reduce bias towards 0 steering angles hence
improving generalization of the model. That is discussed in the [training section][#training-the-model].

The distribution of steering angles that results from the data generator looks as follows:

<p align="center">
 <img src="https://github.com/vguerra/behavioral-clonning/blob/master/imgs/angle_dist_augmented.png" width="500">
</p>

### Training the model

All hyper-parameter values used during the training process are found [here](https://github.com/vguerra/behavioral-clonning/blob/master/model.py#L29-L34).
But we explain each of them through out this section.

The metadata is [split](https://github.com/vguerra/behavioral-clonning/blob/master/model.py#L203) into training 
metadata ( 80% ) and validation metadata ( 20% ).

Thereafter we [build up the model](https://github.com/vguerra/behavioral-clonning/blob/master/model.py#L117-L139)
and proceed to [train it](https://github.com/vguerra/behavioral-clonning/blob/master/model.py#L161-L197).

Our loss function we try to optimize for this case is [MSE](https://github.com/vguerra/behavioral-clonning/blob/master/model.py#L172), which fits best the problem of trying to predict
the steering angle, which is a continuous real value.

For training our model we use the [Adam optimizer](https://github.com/vguerra/behavioral-clonning/blob/master/model.py#L171)
which computes adaptive learning rate for each of the parameters and implements a similar idea to [momentum](http://www.sciencedirect.com/science/article/pii/S0893608098001166). Two characteristics that help it perform better than SGD.
After experimenting with different values of [learning rate](https://github.com/vguerra/behavioral-clonning/blob/master/model.py#L33) we decided to go with **0.0001**,
as the optimizer was converging with less epochs.

For the training process itself we used **12** [EPOCHS](https://github.com/vguerra/behavioral-clonning/blob/master/model.py#L30).
Given that we introduced dropout layers in the model we needed to train the model a bit longer to compensate for that.
The batch size that worked best for our setting was **256** samples. Based on the batch size, we computed
the number of training samples needed for the training and validation phase: [10240 and 1024 respectively](https://github.com/vguerra/behavioral-clonning/blob/master/model.py#L174-L175)

After our model is trainned, we plot the loss over the training and validation sets.

<p align='center'>
<img src="https://github.com/vguerra/behavioral-clonning/blob/master/imgs/loss.png" width="500">
</p>

#### Model generalization

* The CNN has dropout layers that introduce regularization, helping our model to not overfit and generalize better.
50% drop out is used for the [dropout layers](https://github.com/vguerra/behavioral-clonning/blob/master/model.py#L132-L136).
* As we saw in previous sections the training data as it is would get biased towards predicting 0.0 steering angles,
so during the data augmentation phase we introduce a [bias parameter](https://github.com/vguerra/behavioral-clonning/blob/master/model.py#L83).
This parameter starts at 1.0 and is adjusted [after each epoch](https://github.com/vguerra/behavioral-clonning/blob/master/model.py#L182).
By decreasing it after each epoch we decrease the probability of small steering angles to be present in the data
generated by the [data generator](https://github.com/vguerra/behavioral-clonning/blob/master/model.py#L99-L107). This idea was taken from a great post by one of the [SDC Nanodegree students](https://medium.com/@mohankarthik/cloning-a-car-to-mimic-human-driving-5c2f7e8d8aff#.mit3nxrrj).

---
## Simulation

To train the model, use the `mode.py` script as follows:
```
$> python model.py
```
Once the training is done, this will [persist the model](https://github.com/vguerra/behavioral-clonning/blob/master/model.py#L142-L157) into a model.h5 file.

Then you can start the backend that will be used by the simulator to get predictions.
```
$> python drive.py model.h5 run1
```

You can start now the simulator and put it on autonomous mode. Your driving session will save images
into the `run1` directory. With this images you can generate a cool video as follows:
```
python video.py --fps=48 run1
```

This is a video of autonomous driving in the simulator using the model in this project.

[![Video Simulation](https://img.youtube.com/vi/-h-7HzRmsV8/0.jpg)](https://youtu.be/-h-7HzRmsV8)

## References
* [NVIDIA paper](https://arxiv.org/pdf/1604.07316.pdf)
* [Post by Mohan Karthik](https://medium.com/@mohankarthik/cloning-a-car-to-mimic-human-driving-5c2f7e8d8aff#.suozegl1c)
* [Post by Vivek Yadav](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.s4guq3ua7)
* Ideas discussed with fellow students in slack channel #behavioral-clonning