//
// Created by 陈戬 on 2017/12/13.
//
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <string>

using namespace std;

map<string, string> extract_sequence(string directory, int number_seq){

    clock_t start, end;
    start = clock();

    map<string, string> DNA_seq;
    vector<vector<string>> DNA_data;

    string line, key;
    ifstream seq_data;
    seq_data.open(directory);
    int i = 0;
    int seq_cnt = 0;

    if(number_seq != 0){
        while(getline(seq_data,line))
        {
            i++;
            if(i%2 != 0){
                key = line.substr(1);
            }
            else{
                DNA_seq[key] = line;
                seq_cnt += 1;
            }

            if(i==2*number_seq){break;}

        }
    }
    else{
        while(getline(seq_data,line))
        {
            i++;
            if(i%2 != 0){
                key = line.substr(1);
            }
            else{
                DNA_seq[key] = line;
                seq_cnt += 1;
            }

        }
    }
    end = clock();
    double total_t;
    total_t = (double)(end - start) / CLOCKS_PER_SEC;
    printf("read %i DNA sequence loaded, cost: %f seconds\n", seq_cnt, total_t  );

    return DNA_seq;
}
