import os
import json

import dgl
import pickle
import argparse
import torch
import torchtext
import pandas as pd
import numpy as np
from builder import PandasGraphBuilder
from collections import defaultdict

print('Processing Started!')

parser = argparse.ArgumentParser()
directory = ''

output_path = ''

#Paper
with open(os.path.join(directory, 'Papers.json')) as f:
    paper_json = json.load(f)
papers = pd.DataFrame(paper_json)

#Author
with open(os.path.join(directory, 'Authors.json')) as f:
    author_json = json.load(f)
authors = pd.DataFrame(author_json)

#Citation
    #Train
with open(os.path.join(directory, 'Train.json')) as f:
    train_json = json.load(f)
train = pd.DataFrame(train_json)

    #Test
with open(os.path.join(directory, 'Test.json')) as f:
    test_json = json.load(f)
test= pd.DataFrame(test_json)


citation = pd.concat([train, test], axis=0, ignore_index=True)

# Build Graph
citation_filter = [k for k, v in citation['Author_id'].value_counts().items() if v > 1]
citation = citation[citation['Author_id'].isin(citation_filter)]
paper_intersect = set(citation['PMID'].values) & set(papers['PMID'].values)
author_intersect = set(citation['Author_id'].values) & set(authors['Author_id'].values)

new_papers = papers[papers['PMID'].isin(paper_intersect)]
new_authors = authors[authors['Author_id'].isin(author_intersect)]
new_citations = citation[citation['PMID'].isin(paper_intersect) & citation['Author_id'].isin(author_intersect)]
new_citations = new_citations.sort_values('PMID')

label = []
for PMID, df in new_citations.groupby('PMID'):
    idx = int(df.shape[0] * 0.8)
    timestamp = [0] * df.shape[0]
    timestamp = [x if i < idx else 1 for i, x in enumerate(timestamp)]
    label.extend(timestamp)
new_citations['timestamp'] = label

graph_builder = PandasGraphBuilder()
graph_builder.add_entities(new_papers, 'PMID', 'paper')
graph_builder.add_entities(new_authors, 'Author_id', 'author')
graph_builder.add_binary_relations(new_citations, 'PMID', 'Author_id', 'cited')
graph_builder.add_binary_relations(new_citations, 'Author_id', 'PMID', 'cited-by')
g = graph_builder.build()

# Assign features.
node_dict = { 
    'paper': [new_papers, ['PMID','Encoded_Journal','Year','Encoded_keyword','cited_num'], ['cat', 'cat', 'cat', 'cat', 'int']],
    'author': [new_authors, ['Author_id', 'Author_feats'], ['cat', 'int']]
}
edge_dict = { 
    'cited': [new_citations, ['Pair_number', 'timestamp']],
    'cited-by': [new_citations, ['Pair_number', 'timestamp']]
}

for key, (df, features ,dtypes) in node_dict.items():
    for value, dtype in zip(features, dtypes):
        if dtype == 'int':
            array = np.array([i for i in df[value].values])
            g.nodes[key].data[value] = torch.FloatTensor(array)
        elif dtype == 'cat':
            g.nodes[key].data[value] = torch.LongTensor(df[value].astype('category').cat.codes.values)

for key, (df, features) in edge_dict.items():
    for value in features:
        g.edges[key].data[value] = torch.LongTensor(df[value].values.astype(np.float32))


# 실제 ID와 카테고리 ID 딕셔너리
paper_cat = new_papers['PMID'].astype('category').cat.codes.values
author_cat = new_authors['Author_id'].astype('category').cat.codes.values

paper_cat_dict = {k: v for k, v in zip(paper_cat, new_papers['PMID'].values)}
author_cat_dict = {k: v for k, v in zip(author_cat, new_authors['Author_id'].values)}

# Label
val_dict = defaultdict(set)
for PMID, df in new_citations.groupby('PMID'):
    val_dict[PMID] = set(df[df['timestamp'] == 1]['Author_id'].values)
    
# Build title set
textual_feature = {'name': new_authors['Author_name'].values}

dataset = {
    'train-graph': g,
    'paper-data': new_papers,
    'author-data': new_authors, 
    'citation-data': new_citations,
    'val-matrix': None,
    'test-matrix': torch.LongTensor([[0]]),
    'testset': val_dict, 
    'item-texts': textual_feature,
    'item-images': None,
    'paper-type': 'paper',
    'author-type': 'author',
    'paper-category': paper_cat_dict,
    'author-category': author_cat_dict,
    'paper-to-author-type': 'cited',
    'author-to-paper-type': 'cited-by',
    'timestamp-edge-column': 'timestamp'}

with open(output_path, 'wb') as f:
    pickle.dump(dataset, f)
    
print('Processing Completed!')