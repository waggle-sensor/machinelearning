from Data.data import *
from AL import algos
from engine import testCallEngine
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
    split = (.2, .2, .6)  # (train, val, unlabeled)\
    rounds = 3
    keep_bins = True

    # parse raw data
    # dataClass = data.DataManager(dataName)
    # dataClass.parseData(split,rounds,keep_bins)

    # load data that is already parsed
    dataClass = DataManager(dataName, rounds)
    dataClass.loadCaches()

    # | ----------------------------
    # | 2. Select Active Learning algorithm
    # | ----------------------------
    algo = algos.uniformSample()
    #print(algo(dataClass.unlabeled_cache, 10))
    #print(algo(dataClass.unlabeled_cache, 10))
    #print(algo(dataClass.unlabeled_cache, 10))
    algo.reset()

    # | ----------------------------
    # | 3. Select model
    # | ----------------------------
    """
    Load keras model
    """

    modelName = "mnistCNN"
    zk = zoo.zooKeeper(dataName,modelName)

    # | ----------------------------
    # | 4. Run algorithm and log results
    # | ----------------------------
    """
    Run simulation, make class to run 
    engine = alEngine(model, AL algo, data_class)
    engine.run(n_cycles, keras_model_run_param, log=True, verbose=#)
    """
    testCallEngine()


    # | ----------------------------
    # | Done
    # | ----------------------------
    print("Finshed running")


if __name__ == "__main__":
    main()
