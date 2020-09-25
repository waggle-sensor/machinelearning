from Data.data import *
from AL import algos
from engine import Engine
from Zoo import zoo


#######################################################

def main():
    """
    How to use:

    # 1. Select data

    # 2. Select Active Learning algorithm

    # 3. Select model

    # 4. Run algorithm and log results

    """

    # | ----------------------------
    # | 1. Select data
    # | ---------------------------

    dataName = "MNIST"
    split = (.2, .2, .6)  # (train, val, unlabeled)
    rounds = 3
    keep_bins = True

    """
    # Case 1: data is in raw form and data_tab files not made
    # parse raw data
    dataClass = data.DataManager(dataName)
    dataClass.parseData(split,rounds,keep_bins)
    dataClass.loadCaches()
    
    # Case 2: data has been parsed and just needs to load 
    dataClass = DataManager(dataName, rounds)
    dataClass = CustomGenerator(dataName,rounds)
    dataClass.loadCaches()
    
    """

    # Case 3: call DataManager for pre-made test class
    dataClass = mnistLoader(rounds, True)

    # | ----------------------------
    # | 2. Select Active Learning algorithm
    # | ----------------------------

    """
    # algo() call takes cache and number of samples and returns samples based on algo method 
    print(algo(dataClass.unlabeled_cache, 10))
    """

    algo = algos.uniformSample()
    algo.reset()

    # | ----------------------------
    # | 3. Select model
    # | ----------------------------

    modelName = "mnistCNN"
    zk = zoo.zooKeeper(dataName, modelName, show_model=True)

    # | ----------------------------
    # | 4. Run algorithm and log results
    # | ----------------------------

    # Declare engine
    sample_size = 100
    engine = Engine(algo, dataClass, zk, sample_size)

    # Initial training of model on original training data
    engine.initialTrain(epochs=2, batch_size=64, val=True, plot=True)

    # Run active learning algo
    # Round is how many times the active learning algo samples
    # cycles is how many epochs the model is retrained each time a round occurs of sampling
    engine.run(rounds=2, cycles=1, batch_size=64, val=True, plot=True)

    # | ----------------------------
    # | Done
    # | ----------------------------

    print("Finshed running")


if __name__ == "__main__":
    main()
