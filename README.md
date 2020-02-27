# Images-Similarity-Model
An Images Similarity Model using Siamese NN

utils.py - Script containing all the functions of the pre-processing and the construction of the dataset, including the reading of the CIFAR-10 dataset, the transformations, and the visualization of the model’s outputs

DeepModel.py - Class of the image similarity model; architecture, training, testing. Includes the option of plotting montages of pairs of images and their feature maps

main.py -	Main file, which creates an object of the similarity model using DeepModel.py. Train and test using this file

comparison.py -	Comparison of the model results to MSE DSSIM and IMGonline website
