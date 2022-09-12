# AsMac
AsMac is a machine learning framework designed for pairwise comparison of [ribosomal RNA](https://en.wikipedia.org/wiki/Ribosomal_RNA) sequences.<br />
This algorithm implementation is written in `Python 3.7`. This instruction is for Linux/MacOS.

## 1. Installation
If a python environment for `Python 3.7` or later version has not been set up yet, please follow these steps to create a virtual environment.<br />
Install and create a virtual environment for python3
```
sudo pip3 install virtualenv
python3 -m venv venv3
```

Activate the python virtual environment. Then, install packages.<br />
```
source ./venv3/bin/activate
pip install -r environment.txt
```
Build the cython executable by:
```
python setup_softnw.py build_ext --inplace
```

## 2. Usage
Use -h or --help flags to get help text for the program.<br />
```
python AsMac.py -h
```
Choose a model with the -m or --model flags, and run the script like the example:
```
python AsMac.py -i path/to/input.fasta -o path/to/output.csv -m 16S-full
```
AsMac takes input sequences of fasta format and output a pairwise distance matrix in a CSV table. For example:
|       | seq_1 | seq_2 | seq_3 | 
| ----------- | ----------- | ----------- | ----------- | 
| seq_1 | 0   | 0.2   | 0.3 | 
| seq_2 | 0.2   | 0   | 0.1 | 
| seq_3 | 0.3   | 0.1   | 0 |


## 3. Train new model (optional)
## Preparing training and testing data
Download `C++` library [SeqAn](https://github.com/seqan/seqan) or use the attached version to compile the c++ code for NW alignment.
```
unzip seqan.zip
g++ -I . -std=c++1z -o CppAlign/align CppAlign/main.cpp CppAlign/read_fasta.cpp
```

Then generate alignment distance results for the input sequences.
```
./CppAlign/align ./data/training_seq.fa 0
```
The alignment process is very time-consuming on a personal computer. The demo code uses the finished result: training_dist_prepared.txt

## Run the demo code for training
```
jupyter notebook demo.ipynb
```


## Reference
1. [Chen, Jian, Le Yang, Lu Li, Steve Goodison, and Yijun Sun. "Alignment-free Sequence Comparisons via Approximate String Matching." bioRxiv (2022): 2020-05.](https://www.biorxiv.org/content/10.1101/2020.05.24.113852v3)<br />
2. [Koide, Satoshi, Keisuke Kawano, and Takuro Kutsuna. "Neural edit operations for biological sequences." Advances in Neural Information Processing Systems 31 (2018).](https://proceedings.neurips.cc/paper/2018/hash/d0921d442ee91b896ad95059d13df618-Abstract.html)

