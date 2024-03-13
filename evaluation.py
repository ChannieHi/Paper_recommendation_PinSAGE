import numpy as np
import pandas as pd
import torch
import pickle
import dgl
import argparse
from sklearn.neighbors import NearestNeighbors

def check_param_num(model):
    '''
    check num of model parameters

    :model: pytorch model object
    :return: int
    '''
    param_num = 0 
    for parameter in model.parameters():
        param_num += parameter.shape[0]
    return param_num

def node_to_author(nodes, id_dict, category_dict):
    '''
    Transform node id to real item id

    :items: node id list
    :id_dict: {node id: item category id}
    :category_dict: {item category id: real item id}
    '''
    ids = [id_dict[i] for i in nodes]
    ids = [category_dict[i] for i in ids]   
    return ids

def get_blocks(seeds, author_ntype, textset, sampler):
    blocks = []
    for seed in seeds:
        block = sampler.get_block(seed, author_ntype, textset)
        blocks.append(block)
    return blocks

def get_all_emb(gnn, seed_array, textset, author_ntype, neighbor_sampler, batch_size, device='cuda'):
    seeds = torch.arange(seed_array.shape[0]).split(batch_size)
    testset = get_blocks(seeds, author_ntype, textset, neighbor_sampler)

    gnn = gnn.to(device)
    gnn.eval()
    with torch.no_grad():
        h_author_batches = []
        for blocks in testset:
            for i in range(len(blocks)):
                blocks[i] = blocks[i].to(device)

            h_author_batches.append(gnn.get_repr(blocks))
        h_author = torch.cat(h_author_batches, 0)
    return h_author
    
def author_by_paper_batch(graph, paper_ntype, author_ntype, paper_to_author_etype, weight, args):
    '''
    :return: list of interacted node ids by every users 
    '''
    rec_engine = LatestNNRecommender(
        paper_ntype, author_ntype, paper_to_author_etype, weight, args.batch_size)

    graph_slice = graph.edge_type_subgraph([rec_engine.paper_to_author_etype])
    n_papers = graph.number_of_nodes(rec_engine.paper_ntype)  # paper 개수
    latest_interactions = dgl.sampling.select_topk(graph_slice, args.k, rec_engine.timestamp, edge_dir='out')
    paper, latest_authors = latest_interactions.all_edges(form='uv', order='srcdst')
    # paper, latest_authors = (k * n_papers)

    authors_df = pd.DataFrame({'paper': paper.numpy(), 'author': latest_authors.numpy()}).groupby('paper')
    authors_batch = [authors_df.get_group(i)['author'].values for i in np.unique(paper)]
    return authors_batch

def prec(recommendations, ground_truth):
    n_papers, n_authors = ground_truth.shape
    K = recommendations.shape[1]
    paper_idx = np.repeat(np.arange(n_papers), K)
    author_idx = recommendations.flatten()
    relevance = ground_truth[paper_idx, author_idx].reshape((n_papers, K))
    hit = relevance.any(axis=1).mean()
    return hit

class LatestNNRecommender(object):
    def __init__(self, paper_ntype, author_ntype, paper_to_author_etype, timestamp, batch_size):
        self.paper_ntype = paper_ntype
        self.author_ntype = author_ntype
        self.paper_to_author_etype = paper_to_author_etype
        self.batch_size = batch_size
        self.timestamp = timestamp

    def recommend(self, full_graph, K, h_paper, h_author):
        """
        Return a (n_user, K) matrix of recommended items for each user
        """
        graph_slice = full_graph.edge_type_subgraph([self.paper_to_author_etype])
        n_papers = full_graph.number_of_nodes(self.paper_ntype)
        latest_interactions = dgl.sampling.select_topk(graph_slice, K, self.timestamp, edge_dir='out')
        paper, latest_authors = latest_interactions.all_edges(form='uv', order='srcdst')
        # each user should have at least one "latest" interaction
        assert torch.equal(paper, torch.arange(n_papers))

        recommended_batches = []
        paper_batches = torch.arange(n_papers).split(self.batch_size)
        for paper_batch in paper_batches:
            latest_author_batch = latest_authors[paper_batch]
            dist = h_author[latest_author_batch] @ h_author.t()

            # 기존 인터랙션 삭제
            # 이 부분을 주석처리했음
            # for i, u in enumerate(user_batch.tolist()):
            #     interacted_items = full_graph.successors(u, etype=self.user_to_item_etype)
            #     dist[i, interacted_items] = -np.inf
            recommended_batches.append(dist.topk(K, 1)[1])

        recommendations = torch.cat(recommended_batches, 0)
        return recommendations


def evaluate_nn(dataset, h_author, k, batch_size):
    g = dataset['train-graph']
    val_matrix = dataset['val-matrix'].tocsr()
    test_matrix = dataset['test-matrix'].tocsr()
    item_texts = dataset['item-texts']
    paper_ntype = dataset['paper-type']
    author_ntype = dataset['author-type']
    paper_to_item_etype = dataset['paper-to-author-type']
    timestamp = dataset['timestamp-edge-column']

    rec_engine = LatestNNRecommender(
        paper_ntype, author_ntype, paper_to_item_etype, timestamp, batch_size)

    recommendations = rec_engine.recommend(g, k, None, h_author).cpu().numpy()
    return prec(recommendations, val_matrix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('item_embedding_path', type=str)
    parser.add_argument('-k', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    with open(args.dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    #with open(args.item_embedding_path, 'rb') as f:
    #    emb = torch.FloatTensor(pickle.load(f))
    with open(args.item_embedding_path, 'rb') as f:
        file_content = f.read()
        emb = torch.FloatTensor(pickle.loads(file_content))
        
    print(evaluate_nn(dataset, emb, args.k, args.batch_size))
