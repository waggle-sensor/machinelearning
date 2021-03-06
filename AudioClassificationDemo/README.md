# Demo of Audio Classification using a CNN in TensorFlow 

## Data
Demo uses the dataset UrbanSounds8K (https://urbansounddataset.weebly.com/urbansound8k.html). Below is a Python function to download the data. 

```python
import os

def downloadUrbanSound8k():
    os.system('wget https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz -O urban8k.tgz')
    os.system('tar -xzf urban8k.tgz')
```

## Part 1: Data Prep
Notebook dataPrep.ipynb shows how to convert an .wav file of an audio clip into spectrogram. 

## Part 2: Make tfRecord Files 
Notebook makeRecords.ipynb converts the dataset into tfRecord files so that a tf.keras model can be used.

## Part 3: Train and Test Model 
File train_model.py trains a CNN over UrbanSounds8K dataset in tfRecord format.

## Comparison of different data prep methods over UrbanSounds8K
![Alt text](imgs/urbansound8k_spect_test_plot.png)
