## A Spiking Neural Network-based UNET Model for Segmentation of the Mandibular Bone

This project aims at investigating a SNN(Spiking Neural Network)-based
UNET model for locating the mandible regions in panoramic dental images.
UNET is known to win the ISBI challenge 
by outperforming other teams.
U-Net, as a kind of CNN, gets its name from its ¡°U¡± shaped architecture that is composed of   
convolutional layers and two networks.
Spiking Neural Network(SNN) simulates human brain activity. 
In previous works, there have been much research on
using UNET in biomedical image segmentation.

In this project, we devised a SNN version of UNET
model for recognition and segmentation of mandibular bone
in 2D panoramic images.
Recently SNN attracts many attentions since
it is efficient and hardware-friendly.
We investigated how to apply SNN technology to
UNET, and the performance of our SNN-based UNET model.

## Data Set
- 2D dental panoramic X-ray images, released in Kaggle 
(https://www.kaggle.com/datasets/daverattan/dental-xrary-tfrecords)

## Implementation
The code is implemented in Python.
One file (.py file) is a version for running on a local PC, and
another file (.ipynb file) is for running in GPU mode on 
the google Colab.

The size of the raw images
is large (3100x1300).
The code includes the blocks that
resizes the raw images before feeding them 
to the Input layer, enabling the speed up of debugging.


## Results
The SNN version of UNET model showed reasonably 
good performance.
We need further research to fully analyze the
performance according to various settings of operational parameters.


## Acknowledgement
I was helped on this project by my colleagues for 
data acquisition and tips on implementation.


## Reference

