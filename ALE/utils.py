import gdown
from zipfile import ZipFile
import os
import shutil

def downloadData():
    """ Downloads example datasets: MNIST, CIFAR10, and a toy dataset """

    url = "https://drive.google.com/uc?export=download&id=1ZaT0nRFVO2kvQT1fbh6b3dqsJAUEINJN"
    output_path = "Data/DataSetZip.zip"
    gdown.download(url,output_path,quiet=False)

    with ZipFile(output_path, 'r') as zipObj:
        # Extract all the contents of zip file in current directory
        zipObj.extractall("Data")

    os.remove(output_path)
    os.rename("Data/DataSetsZip","Data/DataSets")

    if os.path.isdir("Data/__MACOSX"):
        shutil.rmtree("Data/__MACOSX")

