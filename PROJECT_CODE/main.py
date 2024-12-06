import os
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GATv2Conv
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    ndcg_score,
)
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import logging 
import optuna
import plotly

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s', 
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)  

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# ====================================
# 1. Data Loading and Preprocessing
# ====================================

DATA_PATH = "data/scotus_data_2.csv"
logger.info("Loading data...")
data = pd.read_csv(DATA_PATH)

logger.info("Processing date columns...")
data["dest_date"] = pd.to_datetime(data["dest_date"], errors="coerce")
data["source_date"] = pd.to_datetime(data["source_date"], errors="coerce")

data["dest_date"] = data["dest_date"].fillna(pd.Timestamp("2000-01-01"))
data["source_date"] = data["source_date"].fillna(pd.Timestamp("2000-01-01"))

logger.info("Sorting data by destination date...")
data = data.sort_values(by="dest_date").reset_index(drop=True)

# According to the paper:
# Train: up to 2015
# Val: 2015 - 2017
# Test: after 2017
train_cutoff = pd.Timestamp("2015-12-31")
val_cutoff = pd.Timestamp("2017-12-31")

train_mask = data["dest_date"] <= train_cutoff
val_mask = (data["dest_date"] > train_cutoff) & (data["dest_date"] <= val_cutoff)
test_mask = data["dest_date"] > val_cutoff

# We'll use these masks later for splitting edges.

# ====================================
# 2. Feature Extraction
# ====================================

logger.info("Initializing LegalBERT tokenizer and model...")
tokenizer = BertTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
legalbert_model = BertModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
legalbert_model.to(device)
legalbert_model.eval()  # Set to evaluation mode

def get_cls_embedding(text, tokenizer, model, device, max_length=512):
    """
    Generates a CLS embedding for a given text using LegalBERT.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:,0,:].squeeze(0).cpu().numpy()
    return cls_embedding

EMBEDDINGS_DIR = "embeddings"
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

def load_or_compute_embeddings(text_series, column_name, tokenizer, model, device):
    embedding_file = os.path.join(EMBEDDINGS_DIR, f"{column_name}_cls_embeddings.npy")
    if os.path.exists(embedding_file):
        logger.info(f"Loading cached embeddings for {column_name}...")
        embeddings = np.load(embedding_file)
    else:
        logger.info(f"Computing CLS embeddings for {column_name}...")
        embeddings = []
        for text in tqdm(text_series, desc=f"Embedding {column_name}"):
            embedding = get_cls_embedding(text, tokenizer, model, device)
            embeddings.append(embedding)
        embeddings = np.array(embeddings)
        np.save(embedding_file, embeddings)
        logger.info(f"Saved embeddings to {embedding_file}")
    return embeddings

logger.info("Generating/loading text embeddings for source and destination opinions...")
dest_embeddings = load_or_compute_embeddings(data["dest_name"], "dest_name", tokenizer, legalbert_model, device)
source_embeddings = load_or_compute_embeddings(data["source_name"], "source_name", tokenizer, legalbert_model, device)

logger.info("Generating/loading text embeddings for quote and context...")
quote_embeddings = load_or_compute_embeddings(data["quote"], "quote", tokenizer, legalbert_model, device)
context_embeddings = load_or_compute_embeddings(data["destination_context"], "context", tokenizer, legalbert_model, device)

if device.type == "cuda":
    torch.cuda.empty_cache()

# ====================================
# 3. Encoding Categorical and Numerical Features
# ====================================

logger.info("Encoding categorical metadata...")
categorical_features = ["dest_court", "source_court"]
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
ohe.fit(pd.concat([data["dest_court"], data["source_court"]])
        .astype(str).values.reshape(-1, 1))

dest_court_ohe = ohe.transform(data["dest_court"].astype(str).values.reshape(-1, 1))
source_court_ohe = ohe.transform(data["source_court"].astype(str).values.reshape(-1, 1))

logger.info("Encoding date features...")
reference_date = pd.Timestamp("1970-01-01")
data["dest_elapsed_days"] = (data["dest_date"] - reference_date).dt.days
data["source_elapsed_days"] = (data["source_date"] - reference_date).dt.days

logger.info("Scaling numerical features...")
scaler = StandardScaler()
data[["dest_elapsed_days", "source_elapsed_days"]] = scaler.fit_transform(
    data[["dest_elapsed_days", "source_elapsed_days"]]
)

# ====================================
# 4. Constructing the Graph Based on Unique Nodes
# ====================================

logger.info("Constructing the node set...")
unique_ids = pd.unique(data[["dest_id", "source_id"]].values.ravel())
num_nodes = len(unique_ids)
id_to_index = {id_: idx for idx, id_ in enumerate(unique_ids)}
index_to_id = {idx: id_ for id_, idx in id_to_index.items()}

data["source_idx"] = data["source_id"].map(id_to_index).astype(int)
data["dest_idx"] = data["dest_id"].map(id_to_index).astype(int)

text_feature_dim = dest_embeddings.shape[1]  
court_feature_dim = dest_court_ohe.shape[1]
date_feature_dim = 1

# We will accumulate features per node.
node_text_embeddings = torch.zeros((num_nodes, text_feature_dim), dtype=torch.float)
node_court_ohe = torch.zeros((num_nodes, court_feature_dim), dtype=torch.float)
node_elapsed_days = torch.zeros((num_nodes, date_feature_dim), dtype=torch.float)
node_degree = torch.zeros((num_nodes, 1), dtype=torch.float)

logger.info("Aggregating node features...")
for idx, row in tqdm(data.iterrows(), total=data.shape[0], desc="Aggregating node features"):
    src_idx = row["source_idx"]
    dst_idx = row["dest_idx"]

    # For source node
    node_text_embeddings[src_idx] += torch.tensor(source_embeddings[idx], dtype=torch.float)
    node_court_ohe[src_idx] += torch.tensor(source_court_ohe[idx], dtype=torch.float)
    node_elapsed_days[src_idx] += torch.tensor(data["source_elapsed_days"].iloc[idx], dtype=torch.float).unsqueeze(0)
    node_degree[src_idx] += 1

    # For destination node
    node_text_embeddings[dst_idx] += torch.tensor(dest_embeddings[idx], dtype=torch.float)
    node_court_ohe[dst_idx] += torch.tensor(dest_court_ohe[idx], dtype=torch.float)
    node_elapsed_days[dst_idx] += torch.tensor(data["dest_elapsed_days"].iloc[idx], dtype=torch.float).unsqueeze(0)
    node_degree[dst_idx] += 1

node_degree = torch.clamp(node_degree, min=1)
node_text_embeddings = node_text_embeddings / node_degree
node_court_ohe = node_court_ohe / node_degree
node_elapsed_days = node_elapsed_days / node_degree

# Construct the main HeteroData object
hetero_data = HeteroData()
hetero_data["case"].x = torch.cat([node_text_embeddings, node_court_ohe, node_elapsed_days], dim=1).to(device)

logger.info(f"Node feature dimension: {hetero_data['case'].x.size(1)}")

# ====================================
# 5. Edge Construction According to Temporal Splits
# ====================================

logger.info("Constructing edge sets for train/val/test based on temporal splits...")

relation = ('case', 'cites', 'case')

# Edges
edge_index_all = torch.stack([
    torch.tensor(data["source_idx"].values, dtype=torch.long),
    torch.tensor(data["dest_idx"].values, dtype=torch.long)
], dim=0).to(device)

edge_text_features = torch.tensor(np.concatenate([quote_embeddings, context_embeddings], axis=1), dtype=torch.float).to(device)

# Now, we split edges according to the temporal criteria:
train_edges = data[train_mask]
val_edges = data[val_mask]
test_edges = data[test_mask]

train_edge_index = torch.stack([
    torch.tensor(train_edges["source_idx"].values, dtype=torch.long),
    torch.tensor(train_edges["dest_idx"].values, dtype=torch.long)
], dim=0).to(device)

val_edge_index = torch.stack([
    torch.tensor(val_edges["source_idx"].values, dtype=torch.long),
    torch.tensor(val_edges["dest_idx"].values, dtype=torch.long)
], dim=0).to(device)

test_edge_index = torch.stack([
    torch.tensor(test_edges["source_idx"].values, dtype=torch.long),
    torch.tensor(test_edges["dest_idx"].values, dtype=torch.long)
], dim=0).to(device)

train_edge_attr = edge_text_features[train_mask.values]
val_edge_attr = edge_text_features[val_mask.values]
test_edge_attr = edge_text_features[test_mask.values]

# The paper describes negative sampling on train/val/test sets:
def generate_negative_edges(num_neg, edge_index, num_nodes):
    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_neg,
        method='sparse',
    ).to(device)
    return neg_edge_index

train_neg_edge_index = generate_negative_edges(train_edge_index.size(1), train_edge_index, num_nodes)
val_neg_edge_index = generate_negative_edges(val_edge_index.size(1), val_edge_index, num_nodes)
test_neg_edge_index = generate_negative_edges(test_edge_index.size(1), test_edge_index, num_nodes)

# Store them in hetero_data for easy access
hetero_data[relation].edge_index = train_edge_index
hetero_data[relation].edge_attr = train_edge_attr
hetero_data["train_neg_edge_index"] = train_neg_edge_index
hetero_data["val_pos_edge_index"] = val_edge_index
hetero_data["val_neg_edge_index"] = val_neg_edge_index
hetero_data["test_pos_edge_index"] = test_edge_index
hetero_data["test_neg_edge_index"] = test_neg_edge_index

logger.info(f"Train edges: {train_edge_index.size(1)}, Val edges: {val_edge_index.size(1)}, Test edges: {test_edge_index.size(1)}")

# ====================================
# 7. GAIN Model Implementation
# ====================================

logger.info("Implementing the GAIN model...")

class GAIN(nn.Module):
    """
    Graph Attention Inductive Network for link prediction.
    Using GATv2Conv as a stand-in for GAIN layers.
    This model operates in an inductive setting (nodes unseen at train time)
    due to how we split the graph by time.
    """
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        heads=4,
        dropout=0.5,
    ):
        super(GAIN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(
            GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout, add_self_loops=False)
        )
        for _ in range(num_layers - 1):
            self.convs.append(
                GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout, add_self_loops=False)
            )
        self.convs.append(
            GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout, add_self_loops=False)
        )
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        # edge_attr can be optionally used if desired, but not strictly required for vanilla GAIN
        for conv in self.convs[:-1]:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(conv(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

class LinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels=128):
        super(LinkPredictor, self).__init__()
        self.fc1 = nn.Linear(in_channels * 2, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, 1)
    
    def forward(self, x_i, x_j):
        x = torch.cat([x_i, x_j], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze()

in_channels = hetero_data["case"].x.size(1)
hidden_channels = 128
out_channels = 64
heads = 4
num_layers = 3

model = GAIN(in_channels, hidden_channels, out_channels, num_layers, heads).to(device)
link_predictor = LinkPredictor(out_channels).to(device)

logger.info(model)

optimizer = torch.optim.Adam(list(model.parameters()) + list(link_predictor.parameters()), lr=0.001, weight_decay=5e-4)
criterion = nn.BCEWithLogitsLoss()

# ====================================
# 9. Training and Evaluation
# ====================================

train_losses = []

def compute_loss(pos_scores, neg_scores):
    pos_labels = torch.ones(pos_scores.size(0)).to(device)
    neg_labels = torch.zeros(neg_scores.size(0)).to(device)
    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([pos_labels, neg_labels])
    loss = criterion(scores, labels)
    return loss

def train_epoch(model, link_predictor, data, optimizer, criterion):
    model.train()
    link_predictor.train()
    optimizer.zero_grad()

    out = model(
        data["case"].x,
        data[relation].edge_index,
        data[relation].edge_attr,
    )

    pos_src = data[relation].edge_index[0]
    pos_dst = data[relation].edge_index[1]
    pos_scores = link_predictor(out[pos_src], out[pos_dst])

    neg_edge_index = data["train_neg_edge_index"]
    neg_src = neg_edge_index[0]
    neg_dst = neg_edge_index[1]
    neg_scores = link_predictor(out[neg_src], out[neg_dst])

    loss = compute_loss(pos_scores, neg_scores)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, link_predictor, data, split="val"):
    model.eval()
    link_predictor.eval()
    with torch.no_grad():
        out = model(
            data["case"].x,
            data[relation].edge_index,
            data[relation].edge_attr,
        )

    if split == "val":
        pos_edge_index = data["val_pos_edge_index"]
        neg_edge_index = data["val_neg_edge_index"]
    elif split == "test":
        pos_edge_index = data["test_pos_edge_index"]
        neg_edge_index = data["test_neg_edge_index"]
    else:
        raise ValueError("Split must be 'val' or 'test'")

    pos_src = pos_edge_index[0]
    pos_dst = pos_edge_index[1]
    pos_scores = link_predictor(out[pos_src], out[pos_dst]).cpu().detach().numpy()

    neg_src = neg_edge_index[0]
    neg_dst = neg_edge_index[1]
    neg_scores = link_predictor(out[neg_src], out[neg_dst]).cpu().detach().numpy()

    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])

    scores_prob = 1 / (1 + np.exp(-scores))

    average_precision = average_precision_score(labels, scores_prob)
    roc_auc = roc_auc_score(labels, scores_prob)
    preds_binary = (scores_prob >= 0.5).astype(int)
    precision = precision_score(labels, preds_binary, zero_division=0)
    recall = recall_score(labels, preds_binary)
    f1 = f1_score(labels, preds_binary)

    sorted_indices = np.argsort(-scores_prob)
    sorted_labels = labels[sorted_indices]

    def recall_at_k(sorted_labels, k):
        return np.sum(sorted_labels[:k]) / np.sum(sorted_labels)

    rc1 = recall_at_k(sorted_labels, 1)
    rc10 = recall_at_k(sorted_labels, 10)
    ndcg10 = ndcg_score([labels], [scores_prob], k=10)
    map_score = average_precision

    metrics = {
        "average_precision": average_precision,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "rc@1": rc1,
        "rc@10": rc10,
        "NDCG@10": ndcg10,
        "MAP": map_score,
    }

    return metrics, scores_prob, labels

def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

num_epochs = 200
patience = 20
best_recall_10 = -np.inf
patience_counter = 0
best_model_state = None

logger.info("Starting training...")

for epoch in range(1, num_epochs + 1):
    loss = train_epoch(model, link_predictor, hetero_data, optimizer, criterion)
    train_losses.append(loss)

    grad_norm_model = get_grad_norm(model)
    grad_norm_predictor = get_grad_norm(link_predictor)

    val_metrics, val_scores, val_labels = evaluate(model, link_predictor, hetero_data, split="val")

    logger.info(
        f"Epoch {epoch:03d}: Loss={loss:.4f}, Grad Norm M={grad_norm_model:.4f}, "
        f"Grad Norm P={grad_norm_predictor:.4f}, Val AP={val_metrics['average_precision']:.4f}, "
        f"Val ROC AUC={val_metrics['roc_auc']:.4f}, Precision={val_metrics['precision']:.4f}, "
        f"Recall={val_metrics['recall']:.4f}, F1={val_metrics['f1']:.4f}, "
        f"Recall@1={val_metrics['rc@1']:.4f}, Recall@10={val_metrics['rc@10']:.4f}, "
        f"NDCG@10={val_metrics['NDCG@10']:.4f}, MAP={val_metrics['MAP']:.4f}"
    )

    if val_metrics["rc@10"] > best_recall_10:
        best_recall_10 = val_metrics["rc@10"]
        patience_counter = 0
        best_model_state = {
            'model': model.state_dict(),
            'predictor': link_predictor.state_dict()
        }
    else:
        patience_counter += 1

    if patience_counter >= patience:
        logger.info("Early stopping triggered.")
        break

if best_model_state is not None:
    model.load_state_dict(best_model_state['model'])
    link_predictor.load_state_dict(best_model_state['predictor'])
    logger.info("Loaded best model and predictor states.")

logger.info("Evaluating the model on the test set...")
test_metrics, test_scores, test_labels = evaluate(model, link_predictor, hetero_data, split="test")

logger.info(f"Test Average Precision: {test_metrics['average_precision']:.4f}")
logger.info(f"Test ROC AUC: {test_metrics['roc_auc']:.4f}")
logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
logger.info(f"Test F1 Score: {test_metrics['f1']:.4f}")
logger.info(f"Test Recall@1: {test_metrics['rc@1']:.4f}")
logger.info(f"Test Recall@10: {test_metrics['rc@10']:.4f}")
logger.info(f"Test NDCG@10: {test_metrics['NDCG@10']:.4f}")
logger.info(f"Test MAP: {test_metrics['MAP']:.4f}")

# ====================================
# Hyperparameter Tuning with Optuna (as in the code)
# ====================================

logger.info("Starting hyperparameter tuning with Optuna...")

def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 2, 4)
    dropout = trial.suggest_float("dropout", 0.2, 0.8)
    heads = trial.suggest_categorical("heads", [4, 8, 16])

    temp_model = GAIN(
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        heads,
        dropout,
    ).to(device)
    temp_predictor = LinkPredictor(out_channels).to(device)
    temp_optimizer = torch.optim.Adam(
        list(temp_model.parameters()) + list(temp_predictor.parameters()),
        lr=lr, weight_decay=5e-4
    )
    temp_best_val_ap = -np.inf
    temp_patience_counter = 0

    for ep in range(1, num_epochs + 1):
        loss = train_epoch(temp_model, temp_predictor, hetero_data, temp_optimizer, criterion)
        val_metrics, _, _ = evaluate(temp_model, temp_predictor, hetero_data, split="val")
        val_ap = val_metrics["average_precision"]
        trial.report(val_ap, ep)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_ap > temp_best_val_ap:
            temp_best_val_ap = val_ap
            temp_patience_counter = 0
        else:
            temp_patience_counter += 1

        if temp_patience_counter >= patience:
            break

    return temp_best_val_ap

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50) 
logger.info("Best trial:")
trial = study.best_trial
logger.info(f"  Value (Best Val AP): {trial.value}")
logger.info("  Params:")
for key, value in trial.params.items():
    logger.info(f"    {key}: {value}")

# ====================================
# 12. Plot Generation
# ====================================

logger.info("Generating and saving plots...")

os.makedirs("plots", exist_ok=True)

# ROC Curve for Test Set
fpr, tpr, _ = roc_curve(test_labels, test_scores)
plt.figure(figsize=(8, 6))
plt.plot(
    fpr,
    tpr,
    label=f"GAIN (AUC = {test_metrics['roc_auc']:.2f})",
)
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Citation Prediction Models")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("plots/roc_curve.png")
plt.close()
logger.info("Saved ROC curve as 'plots/roc_curve.png'.")

# Training Loss Over Epochs
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("plots/training_loss.png")
plt.close()
logger.info("Saved training loss plot as 'plots/training_loss.png'.")

# Validation Average Precision Over Epochs
val_average_precisions = [m["average_precision"] for m in study.trials_dataframe().params]
val_roc_aucs = [m["roc_auc"] for m in study.trials_dataframe().params]
plt.figure(figsize=(8, 6))
plt.plot(
    range(1, len(val_average_precisions) + 1),
    val_average_precisions,
    label="Validation Average Precision",
)
plt.xlabel("Epoch")
plt.ylabel("Average Precision")
plt.title("Validation Average Precision Over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("plots/validation_average_precision.png")
plt.close()
logger.info("Saved validation average precision plot as 'plots/validation_average_precision.png'.")

# Validation ROC AUC Over Epochs
plt.figure(figsize=(8, 6))
plt.plot(
    range(1, len(val_roc_aucs) + 1),
    val_roc_aucs,
    label="Validation ROC AUC",
    color="orange",
)
plt.xlabel("Epoch")
plt.ylabel("ROC AUC")
plt.title("Validation ROC AUC Over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("plots/validation_roc_auc.png")
plt.close()
logger.info("Saved validation ROC AUC plot as 'plots/validation_roc_auc.png'.")

# Precision-Recall Curve for Test Set
precision, recall, _ = precision_recall_curve(test_labels, test_scores)
plt.figure(figsize=(8, 6))
plt.plot(
    recall,
    precision,
    label=f"GAIN (AP = {test_metrics['average_precision']:.2f})",
)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for Citation Prediction Models")
plt.legend(loc="upper right")
plt.grid(True)
plt.savefig("plots/precision_recall_curve.png")
plt.close()
logger.info("Saved Precision-Recall curve as 'plots/precision_recall_curve.png'.")

# optuna results
logger.info("Generating and saving hyperparameter tuning plots...")

try:
    optuna_plot_history = optuna.visualization.plot_optimization_history(study)
    optuna_plot_history.write_image("plots/hyperparameter_tuning_history.png")
    plt.close()
    logger.info("Saved hyperparameter tuning history as 'plots/hyperparameter_tuning_history.png'.")
    
    optuna_plot_importances = optuna.visualization.plot_param_importances(study)
    optuna_plot_importances.write_image("plots/hyperparameter_importances.png")
    plt.close()
    logger.info("Saved hyperparameter importances as 'plots/hyperparameter_importances.png'.")
except Exception as e:
    logger.error(f"An error occurred while generating plots: {e}")
    logger.info("Ensure that 'plotly' and 'kaleido' are correctly installed.")

# ====================================
# 13. Retraining the Model with Best Hyperparameters
# ====================================

best_params = trial.params
logger.info("Retraining the model with the best hyperparameters...")

best_model = GAIN(
    in_channels,
    best_params["hidden_channels"],
    out_channels,
    best_params["num_layers"],
    best_params["heads"],
    best_params["dropout"],
).to(device)

best_predictor = LinkPredictor(out_channels).to(device)

best_optimizer = torch.optim.Adam(
    list(best_model.parameters()) + list(best_predictor.parameters()),
    lr=best_params["lr"],
    weight_decay=5e-4
)

best_train_losses = []
best_val_average_precisions = []
best_val_roc_aucs = []
best_val_rc1_list = []
best_val_rc10_list = []
best_val_ndcg10_list = []
best_val_map_list = []
best_patience = patience
best_val_ap = -np.inf
best_patience_counter = 0
best_model_state_final = None

logger.info("Starting retraining with best hyperparameters...")

for epoch in range(1, num_epochs + 1):
    loss = train_epoch(best_model, best_predictor, hetero_data, best_optimizer, criterion)
    best_train_losses.append(loss)

    grad_norm_model = get_grad_norm(best_model)
    grad_norm_predictor = get_grad_norm(best_predictor)

    val_metrics, val_scores, val_labels = evaluate(best_model, best_predictor, hetero_data, split="val")
    best_val_average_precisions.append(val_metrics["average_precision"])
    best_val_roc_aucs.append(val_metrics["roc_auc"])
    best_val_rc1_list.append(val_metrics["rc@1"])
    best_val_rc10_list.append(val_metrics["rc@10"])
    best_val_ndcg10_list.append(val_metrics["NDCG@10"])
    best_val_map_list.append(val_metrics["MAP"])

    logger.info(
        f"Epoch {epoch:03d}: Loss={loss:.4f}, Grad Norm Model={grad_norm_model:.4f}, "
        f"Grad Norm Predictor={grad_norm_predictor:.4f}, Val AP={val_metrics['average_precision']:.4f}, "
        f"Val ROC AUC={val_metrics['roc_auc']:.4f}, Precision={val_metrics['precision']:.4f}, "
        f"Recall={val_metrics['recall']:.4f}, F1={val_metrics['f1']:.4f}, "
        f"Recall@1={val_metrics['rc@1']:.4f}, Recall@10={val_metrics['rc@10']:.4f}, "
        f"NDCG@10={val_metrics['NDCG@10']:.4f}, MAP={val_metrics['MAP']:.4f}"
    )

    # check for improvement..
    if val_metrics["average_precision"] > best_val_ap:
        best_val_ap = val_metrics["average_precision"]
        best_patience_counter = 0
        best_model_state_final = {
            'model': best_model.state_dict(),
            'predictor': best_predictor.state_dict()
        }
    else:
        best_patience_counter += 1

    if best_patience_counter >= best_patience:
        logger.info("Early stopping triggered during retraining.")
        break

if best_model_state_final is not None:
    best_model.load_state_dict(best_model_state_final['model'])
    best_predictor.load_state_dict(best_model_state_final['predictor'])
    logger.info("Loaded best retrained model and predictor states.")

# ====================================
# 14. Saving the Best Model and Metrics
# ====================================

logger.info("Saving the best model and evaluation metrics...")

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

model_save_path = "models/best_gain_model.pth"
torch.save({
    'model_state_dict': best_model.state_dict(),
    'predictor_state_dict': best_predictor.state_dict(),
}, model_save_path)
logger.info(f"Saved best model state to '{model_save_path}'.")

metrics_df = pd.DataFrame({
    "Metric": [
        "Average Precision",
        "ROC AUC",
        "Precision",
        "Recall",
        "F1 Score",
        "Recall@1",
        "Recall@10",
        "NDCG@10",
        "MAP",
    ],
    "Test Score": [
        test_metrics["average_precision"],
        test_metrics["roc_auc"],
        test_metrics["precision"],
        test_metrics["recall"],
        test_metrics["f1"],
        test_metrics["rc@1"],
        test_metrics["rc@10"],
        test_metrics["NDCG@10"],
        test_metrics["MAP"],
    ],
})
metrics_save_path = "results/test_metrics.csv"
metrics_df.to_csv(metrics_save_path, index=False)
logger.info(f"Saved test metrics to '{metrics_save_path}'.")

history_df = pd.DataFrame({
    "Epoch": range(1, len(best_train_losses) + 1),
    "Training Loss": best_train_losses,
    "Validation Average Precision": best_val_average_precisions,
    "Validation ROC AUC": best_val_roc_aucs,
    "Validation Recall@1": best_val_rc1_list,
    "Validation Recall@10": best_val_rc10_list,
    "Validation NDCG@10": best_val_ndcg10_list,
    "Validation MAP": best_val_map_list,
})
history_save_path = "results/training_history_retrained.csv"
history_df.to_csv(history_save_path, index=False)
logger.info(f"Saved retrained training history to '{history_save_path}'.")

logger.info("All plots and metrics have been generated and saved successfully.")
