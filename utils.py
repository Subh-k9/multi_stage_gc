import torch
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import networkx as nx

#from gensim.models import Word2Vec
#from gensim.utils import simple_preprocess

import torch_geometric
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_dense_adj, get_laplacian
import cvxpy as cp
from scipy.spatial.distance import pdist, squareform

def projection_distance(features):
   Wl = torch.FloatTensor(100, features.shape[1]).normal_(0,1)
   values = np.array(torch.matmul(features,Wl.T))
   
   return np.abs(values)


def pair_wise_distance(data):
    pairwise_distances = pdist(data, 'euclidean')  # You can use different distance metrics if needed.
    distance_matrix = squareform(pairwise_distances)
    return distance_matrix


def different_bin_frequency(pairwise_distances):
    # Find minimum and maximum distances
    #print(pairwise_distances)
    print("average distance ",sum(pairwise_distances)/len(pairwise_distances))
    print(sum(pairwise_distances))
    print(len(pairwise_distances))
    min_distance = np.min(pairwise_distances)
    max_distance = np.max(pairwise_distances)
    print(min_distance)
    print(max_distance)
    # Define epsilon and number of intervals
    epsilon = 0.2
    num_intervals = int((max_distance - min_distance) / epsilon) + 1

    # Initialize bins for each interval
    bins = [min_distance + i * epsilon for i in range(num_intervals + 1)]

    # Count number of entries in each interval
    hist, _ = np.histogram(pairwise_distances, bins=bins)

    # Add numbers on top of bars
    for i, v in enumerate(hist):
        plt.text(bins[i] + epsilon/2, v + 0.1, str(v), color='blue', ha='center')


    # Plot the bar plot
    plt.bar(bins[:-1], hist, width=epsilon, align='edge')
    plt.xlabel('Distance Intervals')
    plt.ylabel('Number of Entries')
    plt.title('Distribution of Pairwise Distances')
    plt.show()

"""
def data_generation_deepwalk(G):
  number_of_random_walks = 30
  length_of_random_walk = 150
  # Use DeepWalk to generate node embeddings
  walks = []
  for node in G.nodes():
      # Perform 10 random walks of length 80 for each node
      for i in range(number_of_random_walks):
          walk = [str(node)]
          current_node = node
          for j in range(length_of_random_walk):
              neighbors = [n for n in G.neighbors(current_node)]
              if len(neighbors) == 0:
                  break
              current_node = random.choice(neighbors)
              walk.append(str(current_node))
          walks.append(walk)

  # Train a Word2Vec model on the walks to generate node embeddings
  embedding_size = G.W.shape[0]
  model = Word2Vec(walks, vector_size=embedding_size, window=5, min_count=0, sg=1, workers=2)

  # Get the embeddings for all nodes in the graph
  embeddings = []
  for node in G.nodes():
      embeddings.append(model.wv[str(node)])

  embeddings = torch.tensor(embeddings)
  
  return embeddings

def karateClub_data_generation_deepwalk():
  print("Started in karateClub_data_generation using deep walk")
  G = nx.karate_club_graph()

  number_of_random_walks = 30
  length_of_random_walk = 150
  # Use DeepWalk to generate node embeddings
  walks = []
  for node in G.nodes():
      # Perform 10 random walks of length 80 for each node
      for i in range(number_of_random_walks):
          walk = [str(node)]
          current_node = node
          for j in range(length_of_random_walk):
              neighbors = [n for n in G.neighbors(current_node)]
              if len(neighbors) == 0:
                  break
              current_node = random.choice(neighbors)
              walk.append(str(current_node))
          walks.append(walk)

  # Train a Word2Vec model on the walks to generate node embeddings
  embedding_size = 600
  model = Word2Vec(walks, vector_size=embedding_size, window=5, min_count=0, sg=1, workers=2)

  # Get the embeddings for all nodes in the graph
  embeddings = []
  for node in G.nodes():
      embeddings.append(model.wv[str(node)])

  embeddings = torch.tensor(embeddings)

  
  data = KarateClub()[0]
  data.x = embeddings
  n_classes = len(set(np.array(data.y)))
  feature_size = embeddings.size(1)

  return data, feature_size, n_classes

      
def karateClub_data_generation():
  print("Started in karateClub_data_generation")
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  dataset = KarateClub()
  n_classes = len(set(np.array(dataset[0].y)))
  data = dataset[0].to(device)
  A=to_dense_adj(data.edge_index)[0]
  X=data.x  
  A=A.numpy()
  print((A!=0).sum())
  print(X)
  b=np.ones(34)
  z=A@b
  D=np.diag(z)
  L=D-A

  # if you want only two class this logic is also followed in torch geometric KarateClub() for 2 classes
  # if number_of_classes == 2:
  #   data.y[data.y==3]=1
  #   data.y[data.y==2]=0

  y=data.y.numpy()

  # Creating features for zachary's karate club dataset.
  feature_size = 600
  X = np.random.multivariate_normal(np.zeros(34), np.linalg.pinv(L), feature_size).T.astype(np.float32)
  data.x = torch.tensor(X)
  return data, feature_size, n_classes

def HashFunction(fea,Wl,bin_width, bias):
  h = math.floor((1/bin_width)*((np.dot(fea,Wl)) + bias))
  return h
  """

def fix_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def t_sne_visualize_graph(data_subset,targets,name): 
  tsne = TSNE(n_components=2,n_iter=1000)
  tsne_results = tsne.fit_transform(data_subset)
  
  fig, ax = plt.subplots(figsize=(7, 7))
  ax.scatter(
      tsne_results[:, 0],
      tsne_results[:, 1],
      c=targets,
      s=15,
      cmap="jet",
      alpha=0.7,
  )
  ax.set(
      aspect="equal",
      xlabel="$X_1$",
      ylabel="$X_2$",
      )
  plt.savefig(name)
  plt.show()


def visualize_graph(edge_index,num_node,labels,name,pos=None):
  data = torch_geometric.data.Data(edge_index=edge_index,num_nodes = num_node,y = labels)
  g = torch_geometric.utils.to_networkx(data, to_undirected=True)#,num_nodes = num_node)
  
  degrees = [deg for node, deg in g.degree()]
  avg_degree = sum(degrees) / len(g)
  print("Average degree:", avg_degree)
  
  if pos == None:
    pos = nx.spring_layout(g)
  
  nx.draw(g,node_size = 10,node_color=labels,pos=pos)
  plt.savefig(name)
  plt.show()
  return pos, g, avg_degree

def one_hot(x, class_count):
    return torch.eye(class_count)[x, :]

def get_key(val, g_coarsened):
  KEYS = []
  for key, value in g_coarsened.items():
    if val == value:
      KEYS.append(key)
  return len(KEYS),KEYS

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

###### unused functions 
def matrix_sampling(matrix,valid_nodes_mask):
    new_nodes = torch.nonzero(valid_nodes_mask)
    answer = torch.zeros((new_nodes.shape, new_nodes.shape))
    for i in range(new_nodes.shape):
        answer[i] = matrix[new_nodes[i]]
    return answer

def detect_missclassified(matrix, labels):
    num_missclassified = 0
    for i in range(matrix.shape[1]):
        non_zeros_assignments = np.where(matrix[:, i] != 0)
        current_cluster_labels = labels[non_zeros_assignments]

        # print(current_cluster_labels)

        counts = np.bincount(current_cluster_labels)
        max_count = np.max(counts)
        max_indices = np.where(counts == max_count)
        # print("cluster label value",max_indices[0][0])

        for j in range(matrix.shape[0]):
            if matrix[j, i] > 0 and labels[j] != max_indices[0][0]:
                num_missclassified += 1
    return num_missclassified

# def get_smooth_feature_matrix_coarsened_graph(original_edge_index, partition_matrix, original_features):
#   Lap =  get_laplacian(original_edge_index)
#   number_of_coarsened_nodes = partition_matrix.shape[1]
#   feature_size = original_features.shape[1]

#   print(number_of_coarsened_nodes,feature_size) 

#   learned_feature_matrix = cp.Variable((number_of_coarsened_nodes, feature_size))
#   print(learned_feature_matrix.shape)


#   # Define the objective function
#   objective = cp.Minimize(cp.trace(cp.matmul(learned_feature_matrix, cp.matmul(partition_matrix, cp.matmul(Lap, cp.matmul(partition_matrix.T, learned_feature_matrix.T))))))

#   # Define the constraints
#   constraints = [learned_feature_matrix == cp.matmul(partition_matrix.T, original_features)]

#   # Define the problem
#   problem = cp.Problem(objective, constraints)

#   # Solve the problem
#   problem.solve()

#   # Extract the optimal solution
#   learned_feature_matrix_opt = learned_feature_matrix.value

#   trace_smoothness_original = np.sqrt(np.trace(original_features.T @ Lap @ original_features))

#   coarsened_lap = cp.matmul(partition_matrix, cp.matmul(Lap, cp.matmul(partition_matrix.T, learned_feature_matrix.T)))
#   trace_smoothness_coarsened = np.sqrt(np.trace(learned_feature_matrix_opt.T @ coarsened_lap @ learned_feature_matrix_opt))

#   epsilion_bound = (trace_smoothness_original - trace_smoothness_coarsened)/trace_smoothness_original

#   return epsilion_bound


def get_smooth_feature_matrix_coarsened_graph(original_edge_index, partition_matrix, original_features):
    L = get_laplacian(original_edge_index)
    Lap = to_dense_adj(edge_index= L[0], edge_attr= L[1])[0]

    number_of_coarsened_nodes = partition_matrix.shape[1]
    feature_size = original_features.shape[1]

    learned_feature_matrix = cp.Variable((number_of_coarsened_nodes, feature_size))
    print("learned_feature_matrix ",learned_feature_matrix.shape)

    # Convert partition_matrix to a NumPy array
    partition_matrix_dense = partition_matrix.to_dense().numpy()

    print("partition_matrix_dense ",partition_matrix_dense.shape)
    print(Lap.shape)

    # Define the objective function
    objective = cp.Minimize(
        cp.trace(
            cp.matmul(
                learned_feature_matrix.T,
                cp.matmul(
                    partition_matrix_dense.T,
                    cp.matmul(
                        Lap,
                        cp.matmul(
                            partition_matrix_dense,
                            learned_feature_matrix
                        )
                    )
                )
            )
        )
    )

    # Define the constraints
    constraints = [learned_feature_matrix == cp.matmul(partition_matrix_dense.T, original_features)]
    
    # Define the problem
    problem = cp.Problem(objective, constraints)

    coarsened_lap = np.matmul(partition_matrix_dense.T, np.matmul(Lap, partition_matrix_dense))
    print(coarsened_lap)
    #print(is_psd(coarsened_lap))
    print(is_psd(Lap))

    # Solve the problem
    problem.solve()

    # Extract the optimal solution
    learned_feature_matrix_opt = learned_feature_matrix.value

    trace_smoothness_original = np.sqrt(np.trace(original_features.T @ Lap @ original_features))

    coarsened_lap = cp.matmul(partition_matrix_dense, cp.matmul(Lap, partition_matrix_dense.T))
    trace_smoothness_coarsened = np.sqrt(np.trace(learned_feature_matrix_opt.T @ coarsened_lap @ learned_feature_matrix_opt))

    epsilon_bound = (trace_smoothness_original - trace_smoothness_coarsened) / trace_smoothness_original

    return epsilon_bound

def is_psd(matrix):
    eigenvalues, _ = np.linalg.eig(matrix)
    return np.all(eigenvalues >= 0)


def get_smooth_features(original_edge_index, partition_matrix, original_features):
  L = get_laplacian(original_edge_index)
  Lap = to_dense_adj(edge_index= L[0], edge_attr= L[1])[0]

  # Convert partition_matrix to a NumPy array
  partition_matrix_dense = partition_matrix.to_dense().numpy()

  print("partition_matrix_dense ",partition_matrix_dense.shape)
  print(Lap.shape)

  print(original_features.shape)

  learned_feature_matrix = np.matmul(partition_matrix_dense.T, original_features)
  print("learned_feature_matrix ",learned_feature_matrix.shape)


  alpha = 0.7

  second_term =  np.linalg.inv(np.subtract(np.matmul(partition_matrix_dense.T, partition_matrix_dense), 2/alpha * np.matmul(partition_matrix_dense.T,np.matmul(Lap,partition_matrix_dense))))

  # print(second_term)
  # print(original_features)
  # print(partition_matrix_dense)

  learned_feature_matrix = np.matmul(second_term,np.matmul(partition_matrix_dense.T, original_features))
  
  trace_smoothness_original = np.sqrt(np.trace(np.matmul(original_features.T, np.matmul(Lap, original_features))))

  coarsened_lap = np.matmul(partition_matrix_dense.T, np.matmul(Lap, partition_matrix_dense))
  trace_smoothness_coarsened = np.sqrt(np.trace(np.matmul(learned_feature_matrix.T, np.matmul(coarsened_lap, learned_feature_matrix))))

  print(trace_smoothness_original)

  print(trace_smoothness_coarsened)

  epsilon_bound = (trace_smoothness_original - trace_smoothness_coarsened) / trace_smoothness_original

  return epsilon_bound

