# AsMac
demo code of AsMac Algorithm
## 1. Preparing training and testing data
Download `C++` library [SeqAn](https://github.com/seqan/seqan) or use the attached version to compile the c++ code for NW alignment.
```
unzip seqan.zip
g++ -I . -std=c++1z -o CppAlign/align CppAlign/main.cpp CppAlign/read_fasta.cpp
```

Then generate alignment distance result for the input sequences.
```
./CppAlign/align ./data/training_seq.fa 0
```
This alignment process using the NW algorithm might cost more than 1 day. The demo code use the finished result: training_dist_prepared.txt

## 2. Python virtualenv preparation (optional)
The algorithm demo is written in `Python 3.7`. If a python environment has not been setup yet, please follow these steps to create a virtual environment.<br />
Install virtualenv for python3
```
sudo pip3 install virtualenv
```
Create a virtual environment
```
virtualenv -p python3 venv3
```
or
```
python3 -m venv venv3
```

Activate the python virtual environment and install packages.
```
source ./venv3/bin/activate
pip install -r environment.txt
```


## 3. Compile the Cython code for soft-NW algorithm.
```
python setup_softnw.py build_ext --inplace
```
This will compile the `_softnw.pyx` file to a `.so` file. This file is required to run the `jupyter-notebook` demo


## 4. Run the demo code
The algorithm demo is written in `Python 3.7`, model constructed by `torch 1.6.0`
Simply run the notebook file `demo.ipynb`
```
jupyter notebook demo.ipynb
```



## 5. Reference
The code is a demo for the Continuous Sequence Matching model introduced in the paper:
[Alignment-free Sequence Comparisons via Approximate String Matching](https://www.biorxiv.org/content/10.1101/2020.05.24.113852v3)
