from AsMac_model_parallel import AsMac_parallel
from AsMac_utility import *
import torch
torch.set_printoptions(profile="full")
np.set_printoptions(precision=10)
import plotly.express as px
from sklearn.decomposition import PCA

def read_seq(seq_dir):
    f_s = open(seq_dir, 'r')
    seq_list = []
    name_list = []
    while 1:
        name = f_s.readline()[1:-1]
        if not name:
            break
        name_list.append(name)
        seq = f_s.readline()[:-1]
        seq_list.append(seq)
    return name_list, seq_list


def show_pca(data):
    pca = PCA(n_components=3)
    components = pca.fit_transform(data)
    total_var = pca.explained_variance_ratio_.sum() * 100
    fig = px.scatter_3d(
        components, x=0, y=1, z=2,
        title=f'Total Explained Variance: {total_var:.2f}%',
        labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
    )
    fig.show()


if __name__ == "__main__":

    net = AsMac_parallel(4, 300, 20)
    net_state_dict = torch.load('./model/16S-full.pt')
    net.load_state_dict(net_state_dict)

    seq_fp = './data/testing_seq.fa'
    name_list, seq_list = read_seq(seq_fp)

    # show sequences:
    for i in range(10):
        print('sequence %i, name: %s' % (i, name_list[i]))
        print(seq_list[i])

    # convert to one hot sequences
    seq_oh = one_hot(seq_list)
    embeddings = net(seq_oh).detach().numpy().astype(np.float64)

    print('%i embedding vectors done' % len(seq_oh))
    print(embeddings.shape)
    # show PCA plot
    show_pca(embeddings)




