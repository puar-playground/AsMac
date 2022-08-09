import argparse
import os
import time
from AsMac_model import AsMac
from AsMac_utility import *
import torch
import pandas as pd


def read_fasta(seq_dir):
    f_s = open(seq_dir, 'r')
    seq_list = []
    info_list = []
    while 1:
        info = f_s.readline()[:-1]
        if not info:
            break
        seq = f_s.readline()[:-1]

        if set(seq).issubset({'A', 'T', 'G', 'C'}):
            info_list.append(info)
            seq_list.append(seq)
        else:
            raise ValueError('input sequence has symbol other than {A,T,C,G}')

    len_list = [len(s) for s in seq_list]
    max_l = max(len_list)
    min_l = min(len_list)

    return info_list, seq_list, max_l, min_l


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='AsMac is a machine learning model for full length 16S rRNA sequence comparison')
    parser.add_argument("-i", "--input", required=True,
                        help="directory of the input fasta file)")
    parser.add_argument("-o", "--output", required=False
                        , help="prefered output directory. default: current working directory", default=os.getcwd())
    args = vars(parser.parse_args())

    input_file = os.path.split(args["input"])[1]
    input_file_name = input_file.split('.')[0]
    input_dir = os.path.split(args["input"])[0]
    output_file_default = input_file_name + '_similarity.' + 'csv'

    output_file = os.path.split(args["output"])[1]
    output_file_name = output_file.split('.')[0]

    if args["output"] == os.getcwd():
        out_path = os.path.join(args["output"], output_file_default)
    elif output_file.endswith(".csv"):
        out_path = args["output"]
    else:
        raise ValueError('Invalid output format, csv required.')

    # load sequences from input file
    info_list, seq_list, max_l, min_l = read_fasta(args["input"])
    print('%i sequences loaded. length range:[%i, %i]' % (len(seq_list), min_l, max_l))
    seq_oh = one_hot(seq_list)  # convert to one-hot

    # define model
    # number of kernel sequences
    embed_dim = 300
    # kernel length
    kernel_size = 20
    # initialize EINN object
    net = AsMac(4, embed_dim, kernel_size)
    net_state_dict = torch.load('model/final.pt')
    net.load_state_dict(net_state_dict)
    print('AsMac model loaded')

    # AsMac! Do the thing!
    tic = time.time()
    print('Computing embeddings for the sequences...')
    predictions = net.test_forward(seq_oh).detach().numpy().astype(np.float64)
    toc = time.time()
    print('Done!, cost %.2f seconds' % (toc - tic))
    np.fill_diagonal(predictions, 0)
    out_df = pd.DataFrame(predictions, columns=info_list, index=info_list)
    out_df.to_csv(out_path)

