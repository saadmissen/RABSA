
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import networkx as nx
import sense_finder  # keep in same directory @ASMA

# Keep changing and keep saving in results along with these. 
LEARNING_RATE = 0.001
BATCH_SIZE = 32  # Using a value within the range 16-64
GCN_LAYERS = 2  # Assuming 2 layers
GCN_HIDDEN_DIM = 128  # A middle value within the range 64-256
DROPOUT_RATE = 0.5
BILSTM_UNITS = 128  # A middle value within the range 64-256
#SENTIMENT_CLASSES = 3  # Output units for sentiment classification
ACTIVATION_FUNCTION = nn.Softmax(dim=1)  # Softmax activation function

# Step 2: BiLSTM Model
class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout_rate):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.dropout(lstm_out)

# Step 3: Dependency Parsing
def parse_dependencies(sentence):
    dependencies = nltk.dependency_graph.DependencyGraph.fromstring(sentence)
    G = nx.DiGraph()
    for head, rel, dep in dependencies.triples():
        G.add_edge(head[0], dep[0])
    return from_networkx(G)

# Step 4: Graph Convolutional Network (GCN)
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.final_conv = GCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)
        x = self.final_conv(x, edge_index)
        return x

# Step 5: Final Prediction
class SentimentClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, activation_function):
        super(SentimentClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = activation_function

    def forward(self, x):
        return self.activation(self.fc(x))

# Function to process a file and perform sentiment analysis
def process_file_for_sentiment_analysis(file_path):
    bilstm = BiLSTM(embedding_dim=768, hidden_dim=BILSTM_UNITS, dropout_rate=DROPOUT_RATE)
    gcn = GCN(input_dim=512, hidden_dim=GCN_HIDDEN_DIM, output_dim=128, num_layers=GCN_LAYERS, dropout_rate=DROPOUT_RATE)
    classifier = SentimentClassifier(input_dim=128, output_dim=3, activation_function=ACTIVATION_FUNCTION)

    with open(file_path, 'r') as file:
        for line in file:
            sentence = line.strip()
            
            # Step 1: WSD & Embedding from sense_finder
            embeddings = sense_finder.filter_and_process_words(sentence)  # Importing function from sense_finder
            embeddings_tensor = torch.stack(embeddings).unsqueeze(0)  # Shape (1, seq_len, embedding_dim)
            
            # Step 2: BiLSTM
            bilstm_output = bilstm(embeddings_tensor)
            
            # Step 3: Dependency Parsing
            dependency_tree = parse_dependencies(sentence)
            data = Data(x=bilstm_output.squeeze(0), edge_index=dependency_tree.edge_index)
            
            # Step 4: GCN
            gcn_output = gcn(data)
            
            # Step 5: Final Prediction
            final_prediction = classifier(gcn_output.mean(dim=0))
            print(f"Predicted Sentiment: {final_prediction}")

# File path to the positive reviews file # This data collection is modified from the one as prepared.
file_path = 'D:/PhD_Work/Data/positive_reviews.txt'

# Process the file and perform sentiment analysis
process_file_for_sentiment_analysis(file_path)
