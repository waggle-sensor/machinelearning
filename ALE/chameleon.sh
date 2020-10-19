

# ssh-keygen -R ---.-.--.---

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh

bash ~/miniconda.sh -b -p

rm ~/miniconda.sh

source $HOME/miniconda3/bin/activate

conda create --name tf2 python=3.8

conda activate tf2

conda install -c anaconda tensorflow-gpu

pip install scikit-image

pip install pandas

pip install tqdm

pip install typeguard
