# ALE: Active Learning Exploratorium

ALE is a Python library for testing Active Learning algorithims.

## TODO
* Improve UI
* Fix train, val, eval to use remainder of data in last batch

## DONE
* Added ToyA dataset and ToyA_NN model
* Implement engine.save_log(path="log.csv")
* Add controls for saving performance and configuration
* Dynamically show changes in cache size to debug if 
  cache is removing and adding data correctly ** saved in log, just need to display
* Make util to test model and algo compatibility
* Implement inheritance over custom classes to ensure user safety
* Make acc work for get metric, need to round yh to one hot vector

## DataSets
* MNIST
* ToyA

## Models
* mnistCNN
* ToyA_NN

## Algos
* Uniform sampling (i.e., passive learning, serves as baseline)
* Least Confidence
* Ratio Confidence

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
