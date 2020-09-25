# ALE: Active Learning Exploratorium

ALE is a Python library for testing Active Learning algorithims.

## TODO
* Implement inheritance over custom classes to ensure user safety
    ** model class
* Make acc work for get metric, need to round yh to one hot vector
* Make util to test model and algo compatibility
* Add controls for tracking and plotting metrics
* Dynamically show changes in cache size to debug if 
  cache is removing and adding data correctly ** saved in log, just need to display
* Improve UI

## DONE
* Added ToyA dataset and ToyA_NN model
* Implement engine.save_log(path="log.csv")
* Add controls for saving performance and configuration

## DataSets
* MNIST
* ToyA

## Models
* mnistCNN
* ToyA_NN

## Algos
* Uniform sampling (i.e., passive learning, serves as baseline)
* Least Confidence

 Use the package manager pip to install WL
```bash
pip install ale
```

## Usage
```python
import ale
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
