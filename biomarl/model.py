# Automated Feature Selection: A Reinforcement Learning Perspective
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import GCNConv
import torch_geometric.data as data
from itertools import combinations
import numpy as np
import random
import tqdm
import os
import datetime
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from torch.autograd import Variable
import torch.utils.data as Data
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score
from collections import namedtuple
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader


from biomarl.feature_env import FeatureEvaluator

import warnings

from biomarl.utils.logger import info

warnings.filterwarnings("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Check if GPU is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

GMM = GaussianMixture(n_components=2)
BATCH_SIZE = 64
LR = 0.0003
EPSILON = 0.95
GAMMA = 0.85
TARGET_REPLACE_ITER = 50  # After how much time you refresh target network
MEMORY_CAPACITY = int(os.environ.get('BIOMARL_MEMORY_CAPACITY', 1700)) # replay buffer size / DQN-learn warmup threshold
SYNERGY_ETA = 1.0  # weight of the shared-memory synergy signal in action selection (tunable)
EXPLORE_STEPS = 3000  # How many exploration steps you'd like, 
                            # should be larger than MEMORY_CAPACITY, feature

class FeatureGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeatureGNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Dynamically determine hidden dimensions
        self.hidden_dim = min(512, max(64, input_dim // 2))
        
        self.conv1 = GCNConv(input_dim, self.hidden_dim)
        self.conv2 = GCNConv(self.hidden_dim, output_dim)
        self.pooling = torch.nn.AdaptiveAvgPool1d(1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.pooling(x.t()).t().squeeze(0)
        return x.cpu().detach().numpy()

def create_gene_mapping(dataset, pathways):
    # Get gene names from dataset columns
    dataset_genes = dataset.columns[:-1].tolist()  # Exclude target column
    
    # Create mapping from gene names to indices
    gene_to_idx = {gene: idx for idx, gene in enumerate(dataset_genes)}
    
    # Filter pathways to only include genes present in dataset
    filtered_pathways = {}
    for pathway, genes in pathways.items():
        valid_genes = [gene for gene in genes if gene in gene_to_idx]
        if valid_genes:  # Only include pathways with at least one valid gene
            filtered_pathways[pathway] = [gene_to_idx[gene] for gene in valid_genes]
            
    return gene_to_idx, filtered_pathways

def create_pathway_embeddings(filtered_pathways, n_genes, embedding_dim=64):
    pathway_matrix = np.zeros((len(filtered_pathways), n_genes))
    
    for i, (_, gene_indices) in enumerate(filtered_pathways.items()):
        pathway_matrix[i, gene_indices] = 1
                
    pca = PCA(n_components=embedding_dim)
    pathway_embeddings = pca.fit_transform(pathway_matrix)
    
    gene_embeddings = np.zeros((n_genes, embedding_dim))
    for gene_idx in range(n_genes):
        pathways_containing_gene = pathway_matrix[:, gene_idx]
        if np.sum(pathways_containing_gene) > 0:
            gene_embeddings[gene_idx] = np.average(pathway_embeddings[pathways_containing_gene > 0], axis=0)
        else:
            # gene_embeddings[gene_idx] = np.random.normal(0, 0.1, embedding_dim)
            base_embedding = np.mean(pathway_embeddings, axis=0)
            gene_embeddings[gene_idx] = base_embedding
        
    return torch.FloatTensor(gene_embeddings)

def create_adaptive_edges(features, pathway_metrics = None, correlation_weight = 0.7):
    '''
    create edges based on adaptive correlation and pathway relationships
    '''
    corr_matrix = np.abs(np.corrcoef(features.T))
    np.fill_diagonal(corr_matrix, 0)

    n_features = features.shape[1]
    edge_weight_matrix = corr_matrix.copy()

    if pathway_metrics is not None:
        #create pathway similarity matrix
        pathway_sim = np.zeros((n_features, n_features))
        for i in range(n_features):
            for j in range(n_features):
                #calculate jaccard similarity of pathway memebrship
                pathways_i = set(pathway_metrics.gene_pathway_membership[i])
                pathways_j = set(pathway_metrics.gene_pathway_membership[j])
                if pathways_i or pathways_j:
                    pathway_sim[i, j] = len(pathways_i & pathways_j) / len(pathways_i | pathways_j)

        #combine correlation and pathway information
        edge_weight_matrix = correlation_weight * corr_matrix + (1 - correlation_weight) * pathway_sim

    # zero the diagonal so a gene is never linked to itself; GCNConv adds its own self-loops
    np.fill_diagonal(edge_weight_matrix, 0)

    #adaptive threshold using mean + std of the combined weights
    threshold = np.mean(edge_weight_matrix) + np.std(edge_weight_matrix)
    min_edges_per_node = max(2, int(np.log2(n_features)))

    edge_index = []
    for i in range(n_features):
        connected = np.where(edge_weight_matrix[i] > threshold)[0]
        if len(connected) < min_edges_per_node:
            top_connected = np.argsort(edge_weight_matrix[i])[::-1][:min_edges_per_node]
            connected = np.union1d(connected, top_connected)
        for j in connected:
            edge_index.append([i, j])

    return torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)

def build_gene_graph(train_expression, pathway_metrics=None):
    """Build the gene-gene interaction graph (nodes = genes).

    Edges come from expression correlation over the training data, optionally blended with
    pathway Jaccard similarity (create_adaptive_edges). `train_expression` is the
    [n_samples, n_genes] training feature matrix; the returned edge_index indexes genes
    (0..n_genes-1), matching the agent / action_list / synergy-matrix index space.
    """
    return create_adaptive_edges(np.asarray(train_expression), pathway_metrics=pathway_metrics)


def get_gnn_representation(action_list, edge_index, gnn, pathway_embeddings=None):
    """Encode the current selection as a fixed-size state with the gene-graph GNN.

    Nodes = genes (the pre-filtered gene set). The node feature for gene i is [is_selected_i]
    concatenated with is_selected_i * pathway_embedding_i, so the pooled graph embedding reflects
    which genes are currently selected within the gene interaction / pathway graph. The GNN runs in
    eval mode over the prebuilt edge_index and returns a pooled state vector.
    """
    sel = torch.tensor(np.asarray(action_list), dtype=torch.float, device=device).unsqueeze(1)  # [n_genes, 1]
    if pathway_embeddings is not None:
        emb = pathway_embeddings.to(device)              # [n_genes, emb_dim]
        x = torch.cat([sel, sel * emb], dim=1)           # gate the pathway embedding by selection
    else:
        x = sel
    graph = data.Data(x=x, edge_index=edge_index)
    return gnn(graph)



def moving_average(data, window_size):
    weights = np.ones(window_size)/ window_size
    return np.convolve(data, weights, 'valid')

def moving_std(data, window_size):
    return np.array([np.std(data[max(0, i-window_size):i+1]) for i in range(len(data))])


# %%
class Net(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, N_ACTIONS)
        
        self.ln1 = nn.LayerNorm(256)
        self.ln2 = nn.LayerNorm(128)
        self.ln3 = nn.LayerNorm(64)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        single_sample = x.dim() == 1
        if single_sample:
            x = x.unsqueeze(0)
            
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        x = self.out(x)
        
        # If it was a single sample, remove the batch dimension
        if single_sample:
            x = x.squeeze(0)
            
        return x

class CentralizedCritic(nn.Module):
    def __init__(self, N_STATES, N_AGENTS):
        super().__init__()
        self.N_STATES = N_STATES
        self.N_AGENTS = N_AGENTS
        
        self.compress = nn.Linear(N_STATES,  N_AGENTS)
        self.gate = nn.Linear( N_AGENTS,  N_AGENTS)
        self.dynamic_layers = nn.ModuleList([
            nn.Linear( N_AGENTS,  N_AGENTS) for _ in range(3)
        ])
        self.output1 = nn.Linear(N_AGENTS, 128)
        self.output2 = nn.Linear(128, 1)
    
    def forward(self, state):
        # Compress the single state
        compressed = self.compress(state)
        
        # Apply gating mechanism
        gate_weights = torch.sigmoid(self.gate(compressed))
        # print("Shape of compressed:", compressed.shape)
        # print("Shape of gate_weights:", gate_weights.shape)
        gated = compressed * gate_weights
        
        # Dynamic depth processing
        x = gated
        layer_outputs = []
        for layer in self.dynamic_layers:
            layer_output = layer(x)
            x = F.relu(layer_output) * gate_weights + x * (1 - gate_weights)
            layer_outputs.append(x)
        
        # Output processing
        x = F.relu(self.output1(x))
        return self.output2(x)

### ensemble meta learner ##    
class MetaLearner(nn.Module):
    def __init__(self, input_dim):
        super(MetaLearner, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.predictor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return self.predictor(encoded)
    
class EnsembleMetaLearner:
    def __init__(self, n_features, device):
        self.device = device
        self.base_models =[
            RandomForestRegressor(n_estimators=100, n_jobs=-1),
            XGBRegressor(n_estimators=100),
            LGBMRegressor(n_estimators=100, verbose=-1)
            # SVR(kernel='rbf')
            # MLPRegressor(hidden_layer_sizes=(100,50), max_iter=1000)
        ]
        self.meta_model = LinearRegression()
        self.nn_model = MetaLearner(n_features).to(device)
        self.nn_optimizer = optim.Adam(self.nn_model.parameters())
        self.nn_criterion = nn.MSELoss()
        self.kf = KFold(n_splits=4, shuffle=True)

    def fit(self, X, y):
        X_cpu = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
        y_cpu = y.cpu().numpy() if isinstance(y, torch.Tensor) else y

        #train base models
        base_preds = np.zeros((X_cpu.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            model.fit(X_cpu, y_cpu)
            # base_preds[:,i] = model.predict(X_cpu)

        #train meta model
        base_preds = np.column_stack([model.predict(X_cpu) for model in self.base_models])
        self.meta_model.fit(base_preds, y_cpu)

        #train neural net
        self.nn_model.train()
        X_tensor = torch.FloatTensor(X_cpu).to(self.device)
        y_tensor = torch.FloatTensor(y_cpu).unsqueeze(1).to(device)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        for epoch in range(400):
            for batch_X, batch_y in dataloader:
                self.nn_optimizer.zero_grad()
                outputs = self.nn_model(batch_X)
                loss = self.nn_criterion(outputs, batch_y)
                loss.backward()
                self.nn_optimizer.step()

    def predict_with_uncertainty(self, X):
        X_cpu = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
        # print(f'shape of X_cpu is: {X_cpu.shape}')
        # X_cpu_T = X_cpu.T
        #get preds from base models
        # Get predictions from base models
        base_preds = np.column_stack([model.predict(X_cpu) for model in self.base_models])
        # predictions = []
        # print(f'Shape of X is: {X_cpu.shape}')
        # Get prediction from meta model
        meta_pred = self.meta_model.predict(base_preds)

        # Get prediction from neural net
        self.nn_model.eval()
        with torch.no_grad():
            nn_pred = self.nn_model(torch.FloatTensor(X_cpu).to(self.device)).cpu().numpy()

        # print(f'some base preds are: {base_preds[:10]}')
        # print(f'some nn preds are: {nn_pred[:10]}')
        # print(f'some meta preds are: {meta_pred[:10]}')
        # print(f'shape of base preds: {base_preds.shape}')
        # print(f'shape of nn preds: {nn_pred.shape}')
        # print(f'shape of meta preds: {meta_pred.shape}')

        #Combine predictions
        ensemble_pred = 0.3 * meta_pred + 0.4 * nn_pred.squeeze() + 0.3 * np.mean(base_preds, axis=1)
        # ensemble_pred = 0.5 * meta_pred + 0.5 * np.mean(base_preds, axis=1)


        # Calculate uncertainty (you may want to adjust this based on your needs)
        uncertainty = np.std(base_preds, axis=1)
        return ensemble_pred, uncertainty


    def online_update(self, X, y, update_base_models=False):
        # info('DOING ONLINE UPDATE NOW')
        X_cpu = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
        y_cpu = y.cpu().numpy() if isinstance(y, torch.Tensor) else y

        if update_base_models:
            for model in self.base_models:
                if hasattr(model, 'partial_fit'):
                    model.partial_fit(X_cpu, y_cpu)
                else:
                    model.fit(X_cpu, y_cpu)

        base_preds = np.column_stack([model.predict(X_cpu) for model in self.base_models])
        self.meta_model.fit(base_preds, y_cpu)

        #update neural net
        self.nn_model.train()
        X_tensor = torch.FloatTensor(X_cpu).to(self.device)
        y_tensor = torch.FloatTensor(y_cpu).unsqueeze(1).to(device)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        for epoch in range(50):
            for batch_X, batch_y in dataloader:
                self.nn_optimizer.zero_grad()
                outputs = self.nn_model(batch_X)
                loss = self.nn_criterion(outputs, batch_y)
                loss.backward()
                self.nn_optimizer.step()
        
def train_ensemble_meta_learner(feature_env, n_samples=1500):
    n_features = feature_env.ds_size
    ensemble_meta_learner = EnsembleMetaLearner(n_features, device)

    #generate training data
    X = torch.zeros((n_samples, n_features), device=device)
    for i in range(n_samples):
        num_selected = random.randint(int(0.1 * n_features), int(0.5 * n_features))
        selected_features = random.sample(range(n_features), num_selected)
        X[i, selected_features] = 1

    X = X.float()
    y = torch.zeros(n_samples, device=device)

    for i in tqdm.tqdm(range(n_samples), desc='Collecting samples'):
        result, _,  _, _ = feature_env.report_performance(X[i].cpu().numpy(), flag='train', rp=False)
        y[i] = result

    ensemble_meta_learner.fit(X, y)
    return ensemble_meta_learner

def estimate_performance_change(ensemble_meta_learner, action_list, current_performance):
    current_features = torch.tensor(action_list, dtype=torch.float32, device=device)
    n_features = len(action_list)
    
    # print(f"Number of features: {n_features}")
    # print(f"Shape of current_features: {current_features.shape}")

    #create matrix of all possible feature flips
    all_flipped_features = current_features.repeat(n_features,1)
    for i in range(n_features):
        all_flipped_features[i,i] = 1 - all_flipped_features[i,i]

    # print(f"Shape of all_flipped_features: {all_flipped_features.shape}")

    #make a single prediction for all flipped states
    all_estimated_performances, uncertainties = ensemble_meta_learner.predict_with_uncertainty(all_flipped_features)

    # print(f"Shape of all_estimated_performances: {all_estimated_performances.shape}")
    # print(f"Shape of uncertainties: {uncertainties.shape}")

    #calculate the change in performance for each flip
    # current_performance = ensemble_meta_learner.predict(current_features.unsqueeze(0))[0]
    performance_changes = all_estimated_performances - current_performance

    # print(f"Shape of performance_changes: {performance_changes.shape}")
    # print(f'some performance changes are: {performance_changes[:5]}')

    # return performance_changes.flatten(), uncertainties.flatten()
    return performance_changes, uncertainties

def calculate_uncertainty_aware_rewards(estimated_changes, uncertainties, improvement):
    # estimated_changes[i] is the predicted effect of flipping gene i from its current action.
    # The reward credits the action the agent took (the negative of the flip's value): keeping a
    # beneficial gene (flip effect < 0) yields a positive reward, dropping one yields a negative
    # reward -- weighted by confidence and penalised by uncertainty.
    confidence = 1 / (1 + uncertainties) #bounded between 0 and 1
    weighted_changes = -estimated_changes * confidence #value of the action taken
    penalty = np.log1p(uncertainties) # reduce reward for high uncertainty areas, but with diminishing returns
    rewards = weighted_changes - penalty + improvement
    return rewards

## shared memory implementation
class SharedMemory:
    """Pairwise gene-synergy memory.

    Estimates pairwise interaction between genes from the per-gene marginal contributions produced
    by the meta-learner (``estimate_performance_change``). A pair (i, j) is credited by the product
    of their team-relative (mean-centred) driver contributions, signed by whether the realized set
    improved: two genes both more valuable than the current team average that co-occur in an
    improving set accumulate positive synergy; the same drivers co-occurring in a set that made
    things worse accumulate negative synergy, while below-average genes are left neutral. Updates
    are focused on the top-m most influential genes (O(m^2), not O(k^2)) and decay over time,
    yielding a bounded, signed synergy signal used to bias action selection.
    """

    def __init__(self, n_features, top_m=32, update_lr=0.1):
        self.n_features = n_features
        self.synergy_matrix = np.zeros((n_features, n_features))
        self.decay_factor = 0.99
        self.top_m = top_m          # focus credit on the most influential co-selected genes
        self.update_lr = update_lr  # step size for accumulating interaction evidence

    def update(self, selected_idx, contribution, set_improvement):
        """Accumulate signed pairwise interaction evidence.

        selected_idx    : indices of the currently selected genes.
        contribution    : per-gene value-of-presence vector (length n_features);
                          typically ``-estimated_changes`` so that a higher value means
                          the gene is more beneficial to keep.
        set_improvement : realized set-level performance change (its sign directs credit).
        """
        sel = np.asarray(selected_idx).astype(int)
        if sel.size < 2 or set_improvement == 0:
            return

        c = np.asarray(contribution, dtype=float)[sel]
        c = c - c.mean()  # team-relative contribution (zero-mean)
        if not np.any(np.abs(c) > 1e-12):
            return  # no discriminative marginal signal (e.g. meta-learner disabled)

        # focus credit on the above-team-average "driver" genes: below-average genes get 0,
        # so weak genes are never spuriously linked. A pair of joint drivers then accumulates
        # positive synergy in an improving set and negative synergy in a worsening one.
        drivers = np.maximum(c, 0.0)

        # keep only the most influential co-selected genes (O(top_m^2))
        if sel.size > self.top_m:
            keep = np.argsort(-drivers)[:self.top_m]
            sel, drivers = sel[keep], drivers[keep]

        delta = self.update_lr * np.sign(set_improvement) * np.outer(drivers, drivers)
        np.fill_diagonal(delta, 0.0)
        self.synergy_matrix[np.ix_(sel, sel)] += delta

    def decay(self):
        self.synergy_matrix *= self.decay_factor

    def partner_synergy(self, agent_id, current_selections):
        """Bounded, signed coordination signal for one agent given the current team.

        Returns tanh(mean synergy of ``agent_id`` with the genes currently selected),
        in [-1, 1]: positive when the agent's currently-selected partners are ones it has
        historically been synergistic with, negative otherwise, 0 when no partner is on.
        """
        if current_selections is None:
            return 0.0
        partners = np.where(np.asarray(current_selections) == 1)[0]
        partners = partners[partners != agent_id]
        if partners.size == 0:
            return 0.0
        return float(np.tanh(self.synergy_matrix[agent_id, partners].mean()))


## dqn learning with prioritized replay buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity 
        self.alpha = alpha
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.buffer = []
        self.pos = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        # self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total*probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        states, actions, rewards, next_states, dones = np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), np.array(batch[3]), np.array(batch[4]) 

        return (states, actions, rewards, next_states, dones), indices, weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

class DQN(object):
    def __init__(self, N_STATES, N_ACTIONS, shared_memory, agent_id, epsilon=EPSILON, epsilon_decay=0.99, min_epsilon=0.1, explore_steps=3000):
        self.N_STATES = N_STATES
        self.N_ACTIONS = N_ACTIONS
        self.eval_net, self.target_net = Net(N_STATES, N_ACTIONS).to(device), Net(N_STATES, N_ACTIONS).to(device)
        
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = PrioritizedReplayBuffer(MEMORY_CAPACITY)
        # self.priority_scale = 0.7 #detemines how much prioritization is used
        # self.priority_max = 1.0  #max priority for new transitions
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR, weight_decay=1e-5)
        # self.loss_func = nn.MSELoss(reduction='none')
        self.loss_func = nn.HuberLoss(reduction='none', delta=1.0) 
        self.loss_history = []
        self.shared_memory = shared_memory
        self.agent_id = agent_id
        self.td_errors = []
        self.state_values = []

        self.raw_rewards = [] #tracks b_r
        self.future_estimates = [] #tracks q_next.max
        self.init_q_targets = [] #tracks combined q_target
        self.final_q_targets = []

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.explore_steps = explore_steps

        self.initial_lr = LR
        self.min_lr = LR * 0.1
    
    def choose_action(self, x, current_selections=None):
        x = torch.unsqueeze(torch.FloatTensor(x).to(device), 0)

        # Only use shared memory logic if it exists
        if self.shared_memory is not None:
            if np.random.uniform() < self.epsilon:
                # explore
                action = np.random.randint(0, self.N_ACTIONS)
                q_vals = self.eval_net.forward(x).detach().cpu().numpy()[0]
                action_value = q_vals[action]
            else:
                # exploit with synergy bias: P(select) = sigmoid(Q(s,1) - Q(s,0) + eta * partner_synergy)
                # thresholding sigmoid at 0.5 is equivalent to logit > 0
                q_vals = self.eval_net.forward(x).detach().cpu().numpy()[0]
                partner_bias = self.shared_memory.partner_synergy(self.agent_id, current_selections)
                logit = (q_vals[1] - q_vals[0]) + SYNERGY_ETA * partner_bias
                action = 1 if logit > 0 else 0
                action_value = q_vals[action]

        else:
            # Simple epsilon-greedy without shared memory
            if np.random.uniform() > self.epsilon:
                action_value = self.eval_net.forward(x)
                action = torch.max(action_value, 1)[1].data.cpu().numpy()[0]
                action_value = action_value[0][action].item()
            else:
                action = np.random.randint(0, self.N_ACTIONS)
                q_vals = self.eval_net.forward(x).detach().cpu().numpy()[0]
                action_value = q_vals[action]

        return action, action_value

    def store_transition(self, s, a, r, s_, done):
        self.memory.push(s, a, r, s_, done)
        self.memory_counter += 1

    def get_q_values(self, state):
        with torch.no_grad():
            return self.eval_net(torch.FloatTensor(state).to(device)).cpu().numpy()
        
    # def learn(self, global_value):
    #     if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
    #         self.target_net.load_state_dict(self.eval_net.state_dict())
    #     self.learn_step_counter += 1

    #     batch, indices, weights = self.memory.sample(BATCH_SIZE, beta=0.4)
    #     b_s, b_a, b_r, b_s_, b_done = batch

    #     b_s = torch.FloatTensor(b_s).to(device)
    #     b_a = torch.LongTensor(b_a).to(device)
    #     b_r = torch.FloatTensor(b_r).to(device)
    #     b_s_ = torch.FloatTensor(b_s_).to(device)
    #     b_done = torch.FloatTensor(b_done).to(device)
    #     weights = torch.FloatTensor(weights).to(device)

    #     q_eval = self.eval_net(b_s).gather(1, b_a.unsqueeze(1)).squeeze(1)
    #     # q_next = self.target_net(b_s_).detach()
    #     # q_target = b_r + GAMMA * q_next.max(1)[0] * (1 - b_done)

    #     # Use Double DQN:
    #     next_actions = self.eval_net(b_s_)
    #     if len(next_actions.shape) == 1:
    #         next_actions = next_actions.unsqueeze(0)
    #     next_actions = next_actions.argmax(1)
    #     q_next = self.target_net(b_s_)
    #     if len(q_next.shape) == 1:
    #         q_next = q_next.unsqueeze(0)
    #     q_next = q_next.gather(1, next_actions.unsqueeze(1)).squeeze(1)
    #     q_target = b_r + GAMMA * q_next * (1 - b_done)

    #     # Store values for plotting
    #     self.raw_rewards.append(np.mean(b_r.cpu().numpy()))
    #     # self.future_estimates.append(np.mean((GAMMA * q_next.max(1)[0]).cpu().numpy()))
    #     self.init_q_targets.append(np.mean(q_target.cpu().numpy()))


    #     if len(q_next.shape) == 1:
    #         future_estimate = GAMMA * q_next.max().cpu().numpy()
    #     else:
    #         future_estimate = GAMMA * q_next.max(1)[0].cpu().numpy()
    #     self.future_estimates.append(np.mean(future_estimate))

    #     # print(f'b_r is: {b_r}')
    #     # print(f'q_next.max(1)[0] is:{q_next.max(1)[0]}')

    #     # q_target = 0.7 * q_target + 0.3 * global_value

    #     progress = min(self.learn_step_counter / EXPLORE_STEPS, 1.0)
    #     global_weight = max(0.05, 0.3 * (1 - progress))  # Reduces global influence over time
    #     q_target = (1 - global_weight) * q_target + global_weight * global_value

    #     self.final_q_targets.append(np.mean(q_target.cpu().numpy()))

    #     loss = self.loss_func(q_eval, q_target)
    #     loss = (loss * weights).mean()

    #     # loss = self.loss_func(q_eval, q_target)
    #     self.loss_history.append(loss.item())

    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     for param in self.eval_net.parameters():
    #         param.grad.data.clamp_(-1, 1)

    #     #updte learning rate
    #     current_lr = max(self.min_lr, self.initial_lr * (1 - self.learn_step_counter/EXPLORE_STEPS))
    #     for param_group in self.optimizer.param_groups:
    #         param_group['lr'] = current_lr
        
    #     self.optimizer.step()

    #     # Update priorities
    #     td_errors = torch.abs(q_target - q_eval).detach().cpu().numpy()
    #     self.td_errors.append(np.mean(td_errors))

    #     new_priorities = td_errors + 1e-6  # Add small constant to avoid zero priority
    #     self.memory.update_priorities(indices, new_priorities)

    #     # Decay EPSILON
    #     self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


    def learn(self, global_value):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        batch, indices, weights = self.memory.sample(BATCH_SIZE, beta=0.4)
        b_s, b_a, b_r, b_s_, b_done = batch

        b_s = torch.FloatTensor(b_s).to(device)
        b_a = torch.LongTensor(b_a).to(device)
        b_r = torch.FloatTensor(b_r).to(device)
        b_s_ = torch.FloatTensor(b_s_).to(device)
        b_done = torch.FloatTensor(b_done).to(device)
        weights = torch.FloatTensor(weights).to(device)

        # Current Q values
        q_eval = self.eval_net(b_s).gather(1, b_a.unsqueeze(1)).squeeze(1)

        # Double DQN: Use eval net to select action, target net to get value
        with torch.no_grad():  # Don't track gradients for target network
            q_eval_next = self.eval_net(b_s_)  # Get Q values from eval net
            best_actions = q_eval_next.max(dim=1)[1]  # Get best actions from eval net
            q_next = self.target_net(b_s_).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            q_target = b_r + GAMMA * q_next * (1 - b_done)

        # Store values for plotting
        self.raw_rewards.append(np.mean(b_r.detach().cpu().numpy()))
        self.future_estimates.append(np.mean((GAMMA * q_next.detach().cpu().numpy())))
        self.init_q_targets.append(np.mean(q_target.detach().cpu().numpy()))

        if global_value is not None:  # None -> no-critic ablation: skip the global blend entirely
            progress = min(self.learn_step_counter / self.explore_steps, 1.0)
            global_weight = max(0.05, 0.3 * (1 - progress))
            q_target = (1 - global_weight) * q_target + global_weight * global_value

        self.final_q_targets.append(np.mean(q_target.detach().cpu().numpy()))

        loss = self.loss_func(q_eval, q_target)
        loss = (loss * weights).mean()
        self.loss_history.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.eval_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Update priorities
        td_errors = torch.abs(q_target - q_eval).detach().cpu().numpy()
        self.td_errors.append(np.mean(td_errors))

        new_priorities = td_errors + 1e-6
        self.memory.update_priorities(indices, new_priorities)

        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

def sim_decisions(feature_env, dqn_list, shared_memory, state, current_selections=None):
    action_list = np.zeros(len(dqn_list))
    action_values = []
    plot_action_values = []
    for agent, dqn in enumerate(dqn_list):
        action, action_value = dqn.choose_action(state, current_selections=current_selections)
        action_list[agent] = action
        plot_action_values.append((agent, action_value))
        if action == 1:
            action_values.append((agent, action_value))
    return action_list, action_values, plot_action_values

def seq_decision(feature_env, dqn_list, shared_memory, state):
    action_list = np.zeros(len(dqn_list))
    action_values = []
    plot_action_values = []
    for agent, dqn in enumerate(dqn_list):
        action, action_value = dqn.choose_action(state, current_selections = action_list)
        action_list[agent] = action
        plot_action_values.append((agent, action_value))
        if action == 1:
            action_values.append((agent, action_value))
        #update state after each agent's decision
    return action_list, action_values, plot_action_values

def calculate_weighted_average_II(action_values, decay_factor=0.95):
    weights = np.array([decay_factor**i for i in range(len(action_values)-1, -1, -1 )])
    weighted_sum = np.sum(np.array(action_values) * weights)
    return weighted_sum / np.sum(weights)

def calculate_weighted_average(action_0_values, action_1_values, decay_factor=0.98):
    weights = np.array([decay_factor**i for i in range(len(action_0_values)-1, -1, -1 )])
    q_value_differences = np.array(action_1_values) - np.array(action_0_values)
    # weighted_sum = np.sum(np.array(action_values) * weights)
    weighted_sum = np.sum(np.array(q_value_differences) * weights)
    return weighted_sum / np.sum(weights)

def get_weighted_average_ranking(all_action_values, optimal_set):
    feature_avg_values = []
    for feature_idx, is_selected in enumerate(optimal_set):
        if is_selected:
            feature_values = [step[feature_idx][1] for step in all_action_values if feature_idx in dict(step)]
            if feature_values:
                weighted_avg = calculate_weighted_average(feature_values)
                feature_avg_values.append((feature_idx, weighted_avg))

    #sort feature by their weighted avg action vals
    return sorted(feature_avg_values, key=lambda x: x[1], reverse=True)

def get_weighted_average_ranking_II(all_action_values, optimal_set):
    feature_avg_values = []
    for feature_idx, is_selected in enumerate(optimal_set):
        if is_selected:
            feature_values = [step[feature_idx][1] for step in all_action_values if feature_idx in dict(step)]
            if feature_values:
                weighted_avg = calculate_weighted_average_II(feature_values)
                feature_avg_values.append((feature_idx, weighted_avg))

    #sort feature by their weighted avg action vals
    return sorted(feature_avg_values, key=lambda x: x[1], reverse=True)

class PathwayMetrics:
    def __init__(self, filtered_pathways, n_genes):
        self.filtered_pathways = filtered_pathways
        self.n_genes = n_genes
        self.pathway_matrix = self._create_pathway_matrix()
        self.gene_pathway_membership = self._create_gene_pathway_membership()
        self.centrality_scores = self._calculate_centrality_scores()

    def _create_pathway_matrix(self):
        '''
        creates binary matrix of pathway membership
        '''
        matrix = np.zeros((len(self.filtered_pathways), self.n_genes))
        for i, (_, gene_indices) in enumerate(self.filtered_pathways.items()):
            matrix[i, gene_indices] = 1
        return matrix
    
    def _create_gene_pathway_membership(self):
        '''
        maps each gene to its pathways
        '''
        membership = {i: [] for i in range(self.n_genes)}
        for pathway_idx, (pathway_name, gene_indices) in enumerate(self.filtered_pathways.items()):
            for gene_idx in gene_indices:
                membership[gene_idx].append(pathway_idx)
        return membership
    
    def _calculate_centrality_scores(self):
        '''
        calculate gene centrality based on pathway connections
        '''
        centrality = np.zeros(self.n_genes)
        for gene_idx in range(self.n_genes):
            #number of pathways the gene belongs to
            pathway_count = len(self.gene_pathway_membership[gene_idx])
            if pathway_count > 0:
                #consider both pathway memberships and cross-pathway connectivity
                connected_genes = set()
                for pathway_idx in self.gene_pathway_membership[gene_idx]:
                    connected_genes.update(np.where(self.pathway_matrix[pathway_idx] == 1)[0])
                centrality[gene_idx] = pathway_count * len(connected_genes)

        #normalize scores
        if centrality.max() > centrality.min():
            centrality = (centrality - centrality.min()) / (centrality.max() - centrality.min())
        return centrality

    def calculate_pathway_coverage(self, selected_features):
        '''
        calculate pathway coverage score for selected features
        '''
        if not any(selected_features):
            return 0.0
        
        selected_indices = np.where(selected_features == 1)[0]
        
        coverage_scores = []
        for pathway_genes in self.filtered_pathways.values():
            pathway_size = len(pathway_genes)
            if pathway_size > 0:
                selected_pathway_genes = sum(1 for gene in pathway_genes if gene in selected_indices)
                coverage_scores.append(selected_pathway_genes / pathway_size)

        return np.mean(coverage_scores) if coverage_scores else 0.0
    
    def get_centrality_reward(self, selected_features):
        '''
        get reward based on centrality of selected features
        '''
        if not any(selected_features):
            return 0.0
        selected_indices = np.where(selected_features == 1)[0]
        return np.mean(self.centrality_scores[selected_indices])


def calculate_rewards(surrogate, action_list, current_state, improvement, pathway_metrics=None):
    n_features = len(action_list)
    possible_changes = np.zeros((n_features, n_features))

    #generate possible changes for each feature
    for i in range(n_features):
        possible_changes[i,i] = 1 if action_list[i] == 0 else -1

    #get predictions from surrogate model
    predicted_deltas, uncertainties = surrogate.predict_changes(action_list, possible_changes)

    #calculate base reward using uncertainty aware method
    base_rewards = predicted_deltas.flatten() - 0.1 * uncertainties.flatten() + improvement

    if pathway_metrics is not None:
        rewards = calculate_combined_rewards(base_rewards, action_list, pathway_metrics)
    else:
        rewards = base_rewards
        
    return rewards

def calculate_combined_rewards(base_rewards, action_list, pathway_metrics):
    """Calculate rewards while preserving gene connectivity information"""
    r_list = np.zeros_like(base_rewards)
    
    # Get the indices of selected genes
    selected_genes = np.where(action_list == 1)[0]

    # Amplification factors for positive rewards
    performance_boost = 1.5  # Boost factor for positive base rewards
    centrality_boost = 1.5  # Boost factor for positive centrality differences
    
    # Calculate pathway metrics for the full selection
    full_pathway_coverage = pathway_metrics.calculate_pathway_coverage(action_list)
    full_centrality_reward = pathway_metrics.get_centrality_reward(action_list)
    
    for i in range(len(base_rewards)):
        if i in selected_genes:
            # For selected genes, calculate their contribution by removing them
            temp_action_list = action_list.copy()
            temp_action_list[i] = 0
            
            # Calculate the difference in pathway metrics when this gene is removed
            diff_coverage = full_pathway_coverage - pathway_metrics.calculate_pathway_coverage(temp_action_list)
            diff_centrality = full_centrality_reward - pathway_metrics.get_centrality_reward(temp_action_list)

            #amplify or dampen the positive and negative rewards
            base_rewards[i] = performance_boost * base_rewards[i] if base_rewards[i] > 0 else 0.8 * base_rewards[i]
            diff_centrality = centrality_boost * diff_centrality if diff_centrality > 0 else 0.8 * diff_centrality

            # print(f'diff centrality is: {diff_centrality}')

            # print(f' base rewards is: {base_rewards}')
            # print(f'diff coverage is: {diff_coverage}')
            # print(f' diff centraility is: {diff_centrality}')
            
            # Reward is based on the gene's marginal contribution
            r_list[i] = (0.6 * base_rewards[i] + 
                        0.2 * diff_coverage + 
                        0.2 * diff_centrality)
        else:
            # For unselected genes, calculate their potential contribution by adding them
            temp_action_list = action_list.copy()
            temp_action_list[i] = 1
            
            # Calculate the improvement in pathway metrics when this gene is added
            diff_coverage = pathway_metrics.calculate_pathway_coverage(temp_action_list) - full_pathway_coverage
            diff_centrality = pathway_metrics.get_centrality_reward(temp_action_list) - full_centrality_reward

            #amplify or dampen the positive and negative rewards
            base_rewards[i] = performance_boost * base_rewards[i] if base_rewards[i] > 0 else 0.8 * base_rewards[i]
            diff_centrality = centrality_boost * diff_centrality if diff_centrality > 0 else 0.8 * diff_centrality

            # print(f'diff centrality is: {diff_centrality}')
            
            r_list[i] = (0.6 * base_rewards[i] + 
                        0.2 * diff_coverage + 
                        0.2 * diff_centrality)
    
    return r_list

def gen_marlfs(feature_env, double_limits = False, N_STATES=64, N_ACTIONS=2, EPISODE=-1, n_samples=1500,
                EXPLORE_STEPS=30, max_selected_features = 100, window = 20, seq_freq = 3, pathway_embeddings=None, pathway_metrics = None, filtered_pathways = None, use_meta = None, shared_memory=None, critic=None):
    # np.random.seed(0)
    N_feature = feature_env.ds_size
    best_test_max_accuracy = -float('inf')
    best_optimal_set = None
    best_action_values = []
    improvements = []
    all_states = []
    past_performances_roc = []
    past_performances_pr_auc = []
    BUFFER_SIZE = 50
    UPDATE_FREQUENCY = 50
    experience_buffer = []

    if shared_memory:
        shared_memory = SharedMemory(N_feature)
    else: 
        shared_memory = None

    if use_meta:
        print(f'using meta: {use_meta}')
        # surrogate = (n_features = feature_env.ds_size, device=device)
        # meta_learner = train_ensemble_meta_learner(feature_env)
        meta_learner = train_ensemble_meta_learner(feature_env, n_samples=n_samples)

        # Generate initial training data
        # initial_samples = generate_training_samples(feature_env)
        # for current_state, change, delta_roc in initial_samples:
        #     surrogate.update(current_state, change, delta_roc)
    else:
        meta_learner = None


    if pathway_metrics is None and pathway_embeddings is not None:
        pathway_metrics = PathwayMetrics(filtered_pathways, N_feature)

    selection_patterns = {agent_id:[] for agent_id in range(N_feature)}

    # Build the gene-graph state encoder (gene graph + GNN in eval mode) used to encode the state.
    gene_graph_edges = build_gene_graph(feature_env.train.iloc[:, :-1].values, pathway_metrics=pathway_metrics)
    gnn_input_dim = 1 + (pathway_embeddings.shape[1] if pathway_embeddings is not None else 0)
    gnn_encoder = FeatureGNN(gnn_input_dim, hidden_dim=128, output_dim=N_STATES).to(device)
    gnn_encoder.eval()  # never trained; eval() disables dropout so the encoding is deterministic

    while True:
        MAX_SELECTED_FEATURES = max_selected_features
        dqn_list = [DQN(N_STATES=N_STATES, N_ACTIONS=N_ACTIONS, shared_memory=shared_memory, agent_id=i,  epsilon=EPSILON, epsilon_decay=0.99,
                         min_epsilon=0.1, explore_steps=EXPLORE_STEPS) for i in range(N_feature)]
        results = []
        all_action_values = []
        all_plot_action_values = []
        all_rewards = []
        pathway_performance = {pathway: [] for pathway in filtered_pathways.keys()}

        if critic:
            critic = CentralizedCritic(N_STATES = N_STATES, N_AGENTS = N_feature).to(device)
            critic_optimizer = torch.optim.Adam(critic.parameters(), lr=LR)
        else:
            critic = None
            critic_optimizer = None


        # Initialize action_list with maximum selected features equal to or less than MAX_SELECTED_FEATURES
        selected_indices = np.random.choice(N_feature, int(0.5 * N_feature), replace=False)
        # selected_indices = np.random.choice(N_feature, np.random.randint(1, int(0.5 * N_feature) + 1), replace=False)
        action_list = np.zeros(N_feature)
        action_list[selected_indices] = 1
        # action_list = np.random.randint(0, 2, size=N_feature)

        while sum(action_list) < 2:
            action_list = np.array([random.randint(0, 1) for _ in range(N_feature)])
            # action_list = np.random.randint(0, 2, size=N_feature)


        # result, og_result= feature_env.report_performance(action_list, flag='train', rp=False)
        result_roc, result_pr_auc, og_roc, og_pr_auc= feature_env.report_performance(action_list, flag='train', rp=False)

        print(f'very first result is: {result_roc, result_pr_auc} and first og result is: {og_roc, og_pr_auc}')
        # info(f'current selection is {action_list}')
        info(f'initial number of selected features is: {int(action_list.sum())}')

        results.append([result_roc, result_pr_auc, action_list])
        all_action_values.append([])

        state = get_gnn_representation(action_list, gene_graph_edges, gnn_encoder, pathway_embeddings=pathway_embeddings)

        best_result_roc = result_roc
        best_result_pr_auc = result_pr_auc
        re_state = state.reshape(1, -1)
        # print(f'shape of state is:{re_state.shape}')
        all_states.append(re_state)

        q_values = {agent:{'action_0': [], 'action_1': []} for agent in range(N_feature)}

        for i in tqdm.tqdm(range(EXPLORE_STEPS), desc='selecting features'):
            r_action_values = []
            if i > 0 and i % seq_freq == 0:
                action_list, action_values, plot_action_values = seq_decision(feature_env, dqn_list, shared_memory, state)
            else:
                action_list, action_values, plot_action_values = sim_decisions(feature_env, dqn_list, shared_memory, state, current_selections=action_list)

            # plot_action_values = [(agent, dqn.get_q_values(state)[action]) for agent, dqn in enumerate(dqn_list)]
            if critic:
                global_value = critic(torch.FloatTensor(state).to(device))
            else:
                global_value = 0
            # print(f'global value is: {global_value}')

            #sort action values for selecting agents
            action_values.sort(key=lambda x: x[1], reverse = True)
            #count the number of selected features
            selected_count = int(action_list.sum())

            '''this block starts here and it tracks selection history dict to track how each agent behaves'''
            # for agent_id, action in enumerate(action_list):
            #     selection_patterns[agent_id].append(action)

            # if i % 500 == 0 and i > 0:
            #     for agent_id in range(N_feature):
            #         last_500_selections = selection_patterns[agent_id][-500:]
            #         # print(last_500_selections)
            #         selection_rate = np.mean(last_500_selections)
            #         if selection_rate > 0.8 or selection_rate < 0.2:
            #             print(f'agent {agent_id} shows strong bias: {selection_rate: .3f}')

            '''this block ends here'''

            #remove features with lower action values
            if selected_count > 0.5 * MAX_SELECTED_FEATURES:
                for j in range(selected_count - MAX_SELECTED_FEATURES):
                    agent_to_remove = action_values [-(j+1)][0] #get agents with the lowest action values
                    action_list[agent_to_remove] = 0

            while sum(action_list) < 2:
                # np.random.seed(i)
                action_list = np.array([random.randint(0, 1) for _ in range(N_feature)])

            result_roc, result_pr_auc, og_roc, og_pr_auc = feature_env.report_performance(action_list, flag='train', rp=False)
             # info(f'current selection is {action_list}')
            state_ = get_gnn_representation(action_list, gene_graph_edges, gnn_encoder, pathway_embeddings=pathway_embeddings)
            re_state_ = state_.reshape(1, -1)
            all_states.append(re_state_)

            experience_buffer.append((action_list, result_roc))
            if len(experience_buffer) > BUFFER_SIZE:
                experience_buffer.pop(0)

            if use_meta and meta_learner is not None and i > 0 and i % UPDATE_FREQUENCY == 0 and len(experience_buffer) > 0:
                #prepare batch data
                batch_X = torch.FloatTensor([exp[0] for exp in experience_buffer]).to(device)
                batch_y = torch.FloatTensor([exp[1] for exp in experience_buffer]).to(device)

                #perform batch update to meta learner
                meta_learner.online_update(batch_X, batch_y, update_base_models=True)

            if len(past_performances_roc) >= window:
                moving_avg_roc = np.mean(past_performances_roc[-window:])
                # moving_avg_pr_auc = np.mean(past_performances_pr_auc[-window:])
                # improvement = (result - og_performance) + (result - moving_avg)
                improvement_roc = (result_roc - best_result_roc) + (result_roc - moving_avg_roc)
                # improvement_pr_auc = (result_pr_auc - best_result_pr_auc) + (result_pr_auc - moving_avg_pr_auc)
            else:
                # improvement = result - og_performance  # Initial improvement calculation without moving average
                improvement_roc = result_roc - best_result_roc  # Initial improvement calculation without moving average
                # improvement_pr_auc = result_pr_auc - best_result_pr_auc

            if use_meta:
                estimated_changes, uncertainties = estimate_performance_change(meta_learner, action_list, current_performance=result_roc)
                r_list = calculate_uncertainty_aware_rewards(estimated_changes, uncertainties, improvement_roc)

            else:
                r_list_roc = action_list * improvement_roc
                # r_list_pr = action_list * improvement_pr_auc
                r_list = r_list_roc
                r_list = np.array(r_list)

            # print(r_list[0:25])

            #validation code:
            # if i % 50 == 0:  # Every 50 steps
            #     # Sample 5 random agents to validate
            #     print("\nValidating surrogate predictions:")
            #     validation_agents = list(range(0, N_feature, 50))
                
            #     for agent in validation_agents:
            #         # Get current prediction from surrogate
            #         pred_delta = predicted_deltas[agent].item()
                    
            #         # Get true delta by actually flipping the feature
            #         temp_action_list = action_list.copy()
            #         temp_action_list[agent] = 1 - temp_action_list[agent]  # Flip selection
            #         new_perf, _ = feature_env.report_performance(temp_action_list, flag='train', rp=False)
            #         true_delta = new_perf - result
                    
            #         print(f"Agent {agent}: Predicted delta: {pred_delta:.4f}, True delta: {true_delta:.4f}, "
            #             f"Error: {abs(pred_delta - true_delta):.4f}")

            # print(action_list.shape, r_list.shape)

            if pathway_metrics is not None:
                r_list = calculate_combined_rewards(r_list, action_list, pathway_metrics)

            '''another random block to check performance in terms of some pathway stuff'''
            # if i % 500 == 0 and i > 0:
            #     for pathway, gene_idx in filtered_pathways.items():
            #         #calculate what percentage of genes in this pathway are being selected
            #         pathway_selection_rate = np.mean([action_list[idx] for idx in gene_idx])

            #         #get the avg perfromance contribution of selected genes in this pathway
            #         selected_genes = [idx for idx in gene_idx if action_list[idx] == 1]
            #         if selected_genes:
            #             pathway_perf = sum(r_list[idx] for idx in selected_genes) / len(selected_genes)
            #             pathway_performance[pathway].append(pathway_perf)

            #             #print meaningful pathways
            #             if pathway_selection_rate > 0.15 and pathway_perf > 0: #thresholds set randomly 
            #                 print(f'\npathway {pathway}')
            #                 print(f"Selection rate: {pathway_selection_rate:.2f}")
            #                 print(f"Average performance contribution: {pathway_perf:.4f}")
            #                 print(f"Number of genes selected: {len(selected_genes)}")
            #             else:
            #                 print('No important pathway related results found here, moving on')
            '''the pathway test block ends here'''

            # '''block to see if surrogate is kind aorking'''
            # if i % 10 == 0:  # Every 50 steps
            #     print("\nValidating surrogate predictions:")
            #     validation_agents = np.random.choice(N_feature, 5, replace=False)
                
            #     for agent in validation_agents:
            #         # Create change for this agent
            #         change = np.zeros(N_feature)
            #         change[agent] = 1 if action_list[agent] == 0 else -1
                    
            #         # Get predictions from surrogate
            #         pred_deltas_roc, uncertainties_roc = surrogate.predict_changes(action_list, [change])
                    
            #         # Get true performance change
            #         temp_action_list = action_list.copy()
            #         temp_action_list[agent] = 1 - temp_action_list[agent]
            #         new_roc, new_pr_auc, _, _ = feature_env.report_performance(temp_action_list, flag='train', rp=False)
            #         true_delta_roc = new_roc - result_roc
            #         # true_delta_pr = new_pr_auc - result_pr_auc
                    
            #         print(f"Agent {agent}:")
            #         print(pred_deltas_roc)
            #         print(true_delta_roc)
            #         print(uncertainties_roc)
            #         # print(f"  ROC - Predicted: {pred_deltas_roc:.4f}, True: {true_delta_roc:.4f}, Uncertainty: {uncertainties_roc:.4f}")
            #         # print(f"  PR-AUC - Predicted: {pred_deltas_pr[0]:.4f}, True: {true_delta_pr:.4f}, Uncertainty: {uncertainties_pr[0]:.4f}")

            # '''block ends here'''

            # print(action_list.shape, r_list.shape)
            improvement = improvement_roc
            # improvements_roc.append(improvement_roc)
            # improvements_pr_auc.append(improvement_pr_auc)
            improvements.append(improvement)
            past_performances_roc.append(result_roc)
            past_performances_pr_auc.append(result_pr_auc)
            all_rewards.append(r_list)
            # if i % 50 == 0:
            #     print(f'new performance is: {result}')
            #     print(f'imporvement is: {improvement}')
            #     print(f'reward for 25 agents is:{r_list[0:25]}')


            if len(past_performances_roc) > window:
                past_performances_roc.pop(0)
                past_performances_pr_auc.pop(0)

            if result_roc > best_result_roc:
                best_result_roc = result_roc

            if result_pr_auc > best_result_pr_auc:
                best_result_pr_auc = result_pr_auc

                
            if shared_memory is not None and use_meta:
                # value-of-presence per gene = -(predicted change on removal): higher => more beneficial to keep
                contribution = -np.asarray(estimated_changes, dtype=float)
                shared_memory.update(np.where(action_list == 1)[0], contribution, improvement)

            #update critic
            if critic:
                critic_loss = F.mse_loss(global_value, torch.FloatTensor([improvement]).to(device))
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

            for agent, dqn in enumerate(dqn_list):
                dqn.store_transition(state, action_list[agent], r_list[agent], state_, False)
                q_vals = dqn.get_q_values(state)
                # print(q_vals)
                q_values[agent]['action_0'].append(q_vals[0])
                q_values[agent]['action_1'].append(q_vals[1])
                state_value = dqn.eval_net(torch.FloatTensor(state).to(device)).mean().item()
                dqn.state_values.append(state_value)

            # print(MEMORY_CAPACITY)
            # print(dqn_list[0].memory.size)
            if dqn_list[0].memory_counter > MEMORY_CAPACITY:
                # print('DQN learning!')
                if critic is not None:
                    global_value_for_dqn = critic(torch.FloatTensor(state).to(device)).detach()
                else:
                    global_value_for_dqn = None  # no-critic ablation: learn() skips the global blend
                # info('DQN learning!')
                for dqn in dqn_list:
                    dqn.learn(global_value_for_dqn)
                if i % 50 == 0:
                    _losses = [d.loss_history[-1] for d in dqn_list if d.loss_history]
                    if _losses:
                        print(f'[step {i}] DQN learning active ({len(_losses)} agents) | mean loss {np.mean(_losses):.5f}', flush=True)
            state = state_
            re_state = state.reshape(1, -1)
            # all_states.append(re_state)
            results.append([result_roc, result_pr_auc, action_list])
            all_action_values.append(action_values)
            all_plot_action_values.append(plot_action_values)

            if shared_memory is not None and i % 25 == 0:
                shared_memory.decay()

        max_accuracy = 0
        # optimal_set = []
        optimal_action_values = []

        # for i, (result, action_list) in enumerate(results[1:], start=1):
        #     if result > max_accuracy:
        #         max_accuracy = result
        #         optimal_set = action_list
        #         optimal_action_values = all_action_values[i] if i < len(all_action_values) else None

        feature_scores = []
        for agent in range(N_feature):
            weighted_diff = calculate_weighted_average(q_values[agent]['action_0'],
                                                    q_values[agent]['action_1'])
            feature_scores.append((agent, weighted_diff))

        ranked_features_and_scores = sorted(feature_scores, key=lambda x:x[1], reverse=True)

        #create new optimal set based on importance score
        optimal_set = [0] * N_feature
        for idx, _ in ranked_features_and_scores[:max_selected_features]:
            optimal_set[idx] = 1

        optimal_set = np.array(optimal_set)  # Convert to numpy array before returning

        # Get test performance for optimal set
        test_max_roc, test_max_pr, original_auc_roc, original_pr_auc = feature_env.report_performance(optimal_set, flag='test', store=False)
        ranked_features = [idx for idx, _ in ranked_features_and_scores]
        weighted_avg_values = [val for _, val in ranked_features_and_scores]


        # Method 2 - Find best performing set during training
        max_roc_2 = 0
        optimal_set_2 = np.zeros(N_feature)

        for i, (result_roc, _, action_list) in enumerate(results[1:], start=1):
            if result_roc > max_roc_2:
                max_roc_2 = result_roc
                optimal_set_2 = action_list.copy()

        # Method 2 - Get weighted ranking and select top features
        weighted_avg_ranking = get_weighted_average_ranking_II(all_plot_action_values, optimal_set_2)
        ranked_features_2 = [idx for idx, _ in weighted_avg_ranking[:max_selected_features]]
        weighted_avg_values_2 = [val for _, val in weighted_avg_ranking[:max_selected_features]]

        # Create new optimal_set_2 with only top max_selected_features
        optimal_set_2 = np.zeros(N_feature)
        optimal_set_2[ranked_features_2] = 1

        print('='* 150)


        # Get test performance for optimal set 2
        test_max_roc_2, test_max_pr_2, original_auc_roc_2, original_pr_auc_2 = feature_env.report_performance(optimal_set_2, flag='test', store=False)

        print(f'ranked features way 1 is: {ranked_features}')
        print(f'optimal number of features to be selected is: {int(optimal_set.sum())}')
        print('='* 150)

        # print(f'ranked features way 2 is: {ranked_features_2}')
        # print(f'optimal number of features to be selected is: {int(optimal_set_2.sum())}')

        return (original_auc_roc, original_pr_auc, 
                test_max_roc, test_max_pr, optimal_set, int(optimal_set.sum()),
                original_auc_roc_2, original_pr_auc_2,test_max_roc_2, test_max_pr_2, 
                ranked_features, ranked_features_2, weighted_avg_values, improvements, all_states, 
                all_plot_action_values, dqn_list, all_rewards, q_values)
