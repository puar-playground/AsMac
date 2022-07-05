# Alignment distance data preparation
The cpp code will do pair-wise global alignment by needleman-wunsch algorithm
for first given number of sequences or all sequences in a txt file.

please compile with the code
```console
g++ -I "compile_dir_of_seqan" -std=c++1z -o align main.cpp read_fasta.cpp
```

The executable take exact 2 command line input separated by space " "

    1. directory of the input file.
       The file should be in fasta format. With odd line of unique sequence name after ">"
       and even line sequence data.

    2. the number of sequences want to align.
       input 0 for all, or input a integer n for first n sequences.

    e.g.

```console
./align ~/usr/path/sequence.fa 0
```

The executable will create a output file in the same directory where the input file is located.
The alignment distance between sequences is written in the output file, with the name of input followed with `_dist`.<br />
e.g. input: `~/usr/path/sequence.fa`, output: `~/usr/path/sequence_dist.txt`

## Requirement
The code use high performance `C++` library [SeqAn](https://github.com/seqan/seqan)
