# AsMac
AsMac is a machine learning model for pairwise comparison between full-length 16S rRNA sequences..<br />
The algorithm demo is written in `Python 3.7`, model constructed by `torch 1.6.0`. Demo code of AsMac Algorithm. This instruction is for Linux/MacOS.

## 1. Installation
If a python environment for `Python 3.7` or later version has not been setup yet, please follow these steps to create a virtual environment.<br />
Install and create a virtual environment for python3
```
sudo pip3 install virtualenv
python3 -m venv venv3
```

Activate the python virtual environment. Then, install packages.<br />
For Linux/MacOS:
```
source ./venv3/bin/activate
pip install -r environment.txt
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
just for machine learning researchers and programmers
## Preparing training and testing data
Download `C++` library [SeqAn](https://github.com/seqan/seqan) or use the attached version to compile the c++ code for NW alignment.
```
unzip seqan.zip
g++ -I . -std=c++1z -o CppAlign/align CppAlign/main.cpp CppAlign/read_fasta.cpp
```

Then generate alignment distance result for the input sequences.
```
./CppAlign/align ./data/training_seq.fa 0
```
This alignment process using the NW algorithm might cost more than 1 day. The demo code uses the finished result: training_dist_prepared.txt

## Run the demo code for training
```
jupyter notebook demo.ipynb
```


## Reference
The code is a demo for the Continuous Sequence Matching model introduced in the paper:<br />
[Alignment-free Sequence Comparisons via Approximate String Matching](https://www.biorxiv.org/content/10.1101/2020.05.24.113852v3)
