# Importing sense_finder.py. @Asma update it when you are importing. Use its version 2

from sense_finder import filter_and_process_words, create_context_embedding


import networkx as nx
from torch import nn

# Define the BiLSTM and GCN models
class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(BiLSTM, self).__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True)

    def forward(self, x):
        outputs, _ = self.bilstm(x)
        return outputs

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, adj, x):
        x = torch.matmul(adj, x)
        x = self.fc(x)
        return x

class GCN(nn.Module):
    def __init__(self, in_features, out_features, num_layers):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList([GCNLayer(in_features if i == 0 else out_features, out_features) for i in range(num_layers)])

    def forward(self, adj, x):
        for layer in self.layers:
            x = layer(adj, x)
        return x

class SentimentClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SentimentClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc(x)
        x = self.softmax(x)
        return x

# Initialize models
bilstm = BiLSTM(input_dim=768, hidden_dim=128, num_layers=2)
gcn = GCN(in_features=256, out_features=128, num_layers=2)
classifier = SentimentClassifier(input_dim=128, output_dim=3)

# Function to create adjacency matrix from dependency tree
def create_adjacency_matrix(graph, num_nodes):
    adj = np.zeros((num_nodes, num_nodes))
    for (i, j) in graph.edges():
        adj[i, j] = 1
        adj[j, i] = 1  # Assuming undirected graph
    return torch.tensor(adj, dtype=torch.float32)

# Function to get the index of the target aspect in the sentence
def get_aspect_index(sentence, aspect):
    words = nltk.word_tokenize(sentence)
    try:
        return words.index(aspect)
    except ValueError:
        raise ValueError(f"Aspect '{aspect}' not found in sentence")

# function to predict the sentiment orientation of the aspect
## Always write this way @ASMA
## clean code so that easier to understand 

def predict_aspect_sentiment(sentence, aspect):
    # Step 1: Process the sentence and get relevant words
    relevant_words = filter_and_process_words(sentence)

    # Step 2: Get context embeddings for relevant words
    embeddings = torch.stack([create_context_embedding(sentence, word) for word in relevant_words])

    # Step 3: Get dependency parse tree and create adjacency matrix
    dep_tree = nltk.dependency_graph.DependencyGraph(nltk.pos_tag(nltk.word_tokenize(sentence)))
    graph = nx.Graph(dep_tree.tree().edges())
    adj = create_adjacency_matrix(graph, len(relevant_words))

    # Step 4: Pass through BiLSTM
    bilstm_out = bilstm(embeddings.unsqueeze(0))  # Add batch dimension

    # Step 5: Apply GCN
    gcn_out = gcn(adj, bilstm_out.squeeze(0))

    # Step 6: Mean pooling of GCN output
    pooled_output = torch.mean(gcn_out, dim=0)

    # Step 7: Predict sentiment orientation using the classifier
    sentiment_logits = classifier(pooled_output)

    return sentiment_logits

# Checker - we are having problems here sir, please check. 
## Updated - similar check - updated - 
sentence = "The hotel staff was incredibly helpful and friendly."
aspect = "staff"
sentiment_logits = predict_aspect_sentiment(sentence, aspect)
predicted_sentiment = torch.argmax(sentiment_logits)

sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
print(f"Predicted sentiment for the aspect '{aspect}': {sentiment_mapping[predicted_sentiment.item()]}")
