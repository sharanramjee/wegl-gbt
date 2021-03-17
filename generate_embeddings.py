# Import required modules
import torch
import ogb
from ogb.graphproppred import PygGraphPropPredDataset
from WEGL.WEGL import WEGL

# Set the random seed
random_seed = 55

# Load the dataset
dataset = PygGraphPropPredDataset(name="ogbg-molhiv")
print('# of graphs = {0}\n# of classes = {1}\n# of node features = {2}\n# of edge features = {3}'.\
         format(len(dataset), dataset.num_classes, dataset.num_node_features, dataset.num_edge_features))
if isinstance(dataset, PygGraphPropPredDataset):
    print('# of tasks = {}'.format(dataset.num_tasks))

# Specify the parameters
# num_hidden_layers = range(3, 9)
num_hidden_layers = [4]
# node_embedding_sizes = [100, 300, 500]
node_embedding_sizes = [300]
# final_node_embeddings = ['concat', 'avg', 'final']
final_node_embeddings = ['final']
num_pca_components = -1
num_experiments = 10
classifiers = ['RF']
# device = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: {}'.format(device))

# Run the algorithm
for final_node_embedding in final_node_embeddings:
    WEGL(dataset=dataset,
         num_hidden_layers=num_hidden_layers,
         node_embedding_sizes=node_embedding_sizes,
         final_node_embedding=final_node_embedding,
         num_pca_components=num_pca_components,
         num_experiments=num_experiments,
         classifiers=classifiers,
         random_seed=random_seed,
         device=device)
