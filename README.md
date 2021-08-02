# African-language-Speech-Recognition---Speech-to-Text-
The World Food Program wants to deploy an intelligent form that collects nutritional information of food bought and sold at markets
in two different countries in Africa - Ethiopia and Kenya. In this work I build a deep learning model that is capable of transcribing
a speech to text and deliver speech-to-text technology for the choosen two  African languages: Amharic and Swahili.

1. Project Structure

   a) Data
	 
Dataset for Amharic    https://github.com/getalp/ALFFA_PUBLIC 

Dataset for Swahili      https://github.com/getalp/ALFFA_PUBLIC

  b) Data Features
	
Input features (X): audio clips of spoken words

Target labels (y): a text transcript of what was spoken

   c)libs
	 
classification (All scripts used for training and evaluation)

  d) notebooks
	
  scripts (Executable scripts)
	
  models (Pretrained Models)
	
	e) Requirements
	
 Pytorch/Tensorflow ,  
 librosa,
 scikit-learn.
 Python 
 
 f) models arctecture
 
 - CNN (Convolutional Neural Network) plus RNN-based (Recurrent Neural Network) architecture
 
 - RNN-based sequence-to-sequence network
 
 Reference
 
 1. https://towardsdatascience.com/audio-deep-learning-made-simple-automatic-speech-recognition-asr-how-it-works-716cfce4c706
 2. 
 
