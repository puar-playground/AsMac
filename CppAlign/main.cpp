//
// Created by Jian Chen on 2019/05/20.
//
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <math.h>
#include <thread>
#include <cstdlib>

#include <sstream>
#include <string>

#include <seqan/basic.h>
#include <seqan/align.h>
#include <seqan/stream.h>  // for printint strings

using namespace seqan;
using namespace std;

extern map<string, string> extract_sequence(string directory, int number_seq);

typedef char TChar;                             // character type
typedef String<char> TSequence;                // sequence type
typedef Align<TSequence, ArrayGaps> TAlign;     // align type
typedef Row<TAlign>::Type TRow;                 // gapped sequence type

int main(int argc, char** argv) {

//    initialize from input
string input_dir = argv[1];
string output_dir = input_dir;
string input_file_name, input_folder, output_file_name;
string seq_no = argv[2];
int seq_no_int;
istringstream(seq_no) >> seq_no_int;

//    get input file name
    for(int pos=input_dir.length()-1;pos>=0;pos--){
        if(input_dir.substr(pos, 1) == "/"){
            input_file_name = input_dir.substr(pos+1, input_dir.length()-pos);
            input_folder = input_dir.substr(0, pos+1);
            cout << input_file_name << '\n';
            cout << "at: " << input_folder << '\n';
            break;
        }
    }
    for(int pos=0;pos<input_file_name.length();pos++){
        if(input_file_name.substr(pos, 1) == "."){
            output_file_name = input_file_name.substr(0, pos) + "_dist.txt";
        }
    }

//    print input information
    cout << "Do needleman-wunsch for file: " << input_file_name;
    if(seq_no == "0"){
        cout << " for all seq" << "\n" << endl;
    }
    else{
        cout << " for " << seq_no << " seq" << "\n" << endl;
    }

//    create output file
    ofstream myfile;
    cout << input_folder + output_file_name << "\n";
    myfile.open(input_folder + output_file_name);

//    read input file
    map<string, string> seq_data = extract_sequence(input_dir, seq_no_int);

//    start doing needleman-wunsch
    map<string, string>::iterator itr, itr1, itr2, itr3;

//    timer started
    clock_t t_start, t_end;
    t_start = clock();

    int cnt1 = 0, cnt2 = 0;
    for (itr1 = seq_data.begin(); itr1 != seq_data.end(); ++itr1) {

        cnt1 += 1;
        cnt2 = 0;
        for (itr2 = seq_data.begin(); itr2 != seq_data.end(); ++itr2) {
            cnt2 += 1;
            if(cnt1 < cnt2){

//                print the pair the program is working on
                stringstream cnt1_strs, cnt2_strs;
                cnt1_strs << cnt1;
                cnt2_strs << cnt2;
                string cnt1_str = cnt1_strs.str();
                string cnt2_str = cnt2_strs.str();

                string show = itr1->first + "-" + itr2->first + " " + cnt1_str + "-" + cnt2_str + "               ";

                cout << show << '\r' << std::flush;

//                load the seqences
                Dna5String seqH = itr1->second;
                Dna5String seqV = itr2->second;

                TAlign align;
                resize(rows(align), 2);
                assignSource(row(align, 0), seqH);
                assignSource(row(align, 1), seqV);

                Score<int, Simple> scoringScheme(2, -3, -2, -5);
                AlignConfig<> alignConfig;

//                nw score
                int result = globalAlignment(align, scoringScheme, alignConfig);

//
//                std::cout << align;
                TRow &row1 = row(align,0);
                TRow &row2 = row(align,1);

                typedef Iterator<TRow>::Type TRowIterator;

                int mis_cnt = 0, gap_cnt = 0, len_cnt = 0;
                double align_dist;

                for (TRowIterator it1 = begin(row1), it2 = begin(row2); it1 != end(row1); ++it1, ++it2)
                {
                    char c1 = isGap(it1) ? gapValue<TChar>(): *it1;
                    char c2 = isGap(it2) ? gapValue<TChar>(): *it2;
                    len_cnt++;

                    if(c1 != c2){
                        if(c1 == '-' || c2 == '-'){gap_cnt++;}
                        else{mis_cnt++;}
                    }
                }
                align_dist = double(mis_cnt + gap_cnt) / double(len_cnt);

//                write to output file
                myfile << itr1->first;
                myfile << "-";
                myfile << itr2->first;
                myfile << " " << setprecision (16) << align_dist;
                myfile << '\n';
//                myfile << row(align, 0) << '\n';
//                myfile << row(align, 1) << '\n';

            }
        }

//        timer report time elapsed
        t_end = clock();
        double total_t;
        total_t = (double)(t_end - t_start) / CLOCKS_PER_SEC;
        cout << "seq" << cnt1 << " done                " << endl;
        printf("cost: %f seconds\n", total_t);

    }

//    close file
    myfile.close();

    return 0;
}
