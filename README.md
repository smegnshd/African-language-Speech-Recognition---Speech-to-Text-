The World Food Program wants to deploy an intelligent form that collects nutritional information of food bought and sold at markets in two different countries in Africa - Ethiopia and Kenya. This project attempts to create a web app that does just that. It will allow users to register the list of items they bought using just their voice. This project utilizes deep learning models, Which are capable of transcribing a speech to text and deliver speech-to-text technology for the choosen two African languages: Amharic and Swahili.
In this work we did amharic speech recognition and feature will do Swahili speech recognition.

Project Structure

Data

Dataset for Amharic https://github.com/getalp/ALFFA_PUBLIC

Data Features

Input features (X): audio clips of spoken words

Target labels (y): a text transcript of what was spoken

Requirements

Pytorch/Tensorflow ,

librosa, scikit-learn, Python,

Model Architecture

CNN (Convolutional Neural Network) plus RNN-based (Recurrent Neural - Network) architecture

RNN-based sequence-to-sequence network

Tasks:

 Setting up DVC and MLflow
 
 Exploring the data and Extracting useful information
 
 Preprocessing and Augmenting the data
 
 Extracting features
 
 Modelling and Deployment using MLOps
 
 Serving predictions on a web interface
 
Current Status

Integrating Preprocessing and Augmentation to the code base

Coming Changes

Modelling and Deployment using MLOps

Reference

https://towardsdatascience.com/audio-deep-learning-made-simple-automatic-speech-recognition-asr-how-it-works-716cfce4c706 https://www.kaggle.com/CVxTz/audio-data-augmentation

https://github.com/udacity/AIND-VUI-Capstone/blob/master/requirements.txt
