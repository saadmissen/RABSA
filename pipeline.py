import nltk
import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertModel
from sqlalchemy import create_engine, Column, String, Float, Integer, MetaData, Table
from sqlalchemy.orm import sessionmaker
from nltk.corpus import wordnet as wn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import networkx as nx

# Database connection
DATABASE_URL = "postgresql://asma786:aspectbasedwork@localhost/wordsenses"
engine = create_engine(DATABASE_URL)
metadata = MetaData()

# Define the word_senses table
word_senses_table = Table(
    'word_senses', metadata,
    Column('id', Integer, primary_key=True),
    Column('word', String),
    Column('best_sense', String),
    Column('similarity', Float)
)

metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Word Sense Disambiguation and Context Embedding
def create_context_embedding(sentence, target_word):
    inputs = tokenizer(sentence, return_tensors='pt')
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    word_index = tokenizer.convert_ids_to_tokens(inputs.input_ids[0]).index(target_word)
    return embeddings[0, word_index, :]

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def find_best_sense(sentence, word):
    context_embedding = create_context_embedding(sentence, word)
    best_sense = None
    highest_similarity = -1
    
    for synset in wn.synsets(word):
        definition = synset.definition()
        gloss_embedding = create_context_embedding(definition, word)
        similarity = cosine_similarity(context_embedding.detach().numpy(), gloss_embedding.detach().numpy())
        
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_sense = synset

    return best_sense, highest_similarity

def filter_and_process_words(sentence):
    words = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(words)
    relevant_words = [word for word, pos in pos_tags if pos.startswith('VB') or pos.startswith('RB') or pos.startswith('JJ')]
    
    embeddings = []
    for word in relevant_words:
        best_sense, similarity = find_best_sense(sentence, word)
        if best_sense:
            save_to_db(word, best_sense.name(), similarity)
            embeddings.append(create_context_embedding(sentence, word))

    return embeddings

def save_to_db(word, best_sense, similarity):
    insert_stmt = word_senses_table.insert().values(word=word, best_sense=best_sense, similarity=similarity)
    session.execute(insert_stmt)
    session.commit()

# BiLSTM Model
class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return lstm_out

# Dependency Parsing - error removed - that function was deprecated @sir
def parse_dependencies(sentence):
    dependencies = nltk.dependency_graph.DependencyGraph.fromstring(sentence)
    G = nx.DiGraph()
    for head, rel, dep in dependencies.triples():
        G.add_edge(head[0], dep[0])
    return from_networkx(G)

# Graph Convolutional Network (GCN) 
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Final Prediction
class SentimentClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SentimentClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=1)

def process_file(file_path):
    bilstm = BiLSTM(embedding_dim=768, hidden_dim=256)
    gcn = GCN(input_dim=512, hidden_dim=256, output_dim=128)
    classifier = SentimentClassifier(input_dim=128, output_dim=3)  # Assuming 3 classes: positive, negative, neutral

    with open(file_path, 'r') as file:
        for line in file:
            sentence = line.strip()
            
            # call WSD & Embedding
            embeddings = filter_and_process_words(sentence)
            embeddings_tensor = torch.stack(embeddings).unsqueeze(0)  # Shape (1, seq_len, embedding_dim)
            
            # call BiLSTM
            bilstm_output = bilstm(embeddings_tensor)
            
            # call Dependency Parsing
            dependency_tree = parse_dependencies(sentence)
            data = Data(x=bilstm_output.squeeze(0), edge_index=dependency_tree.edge_index)
            
            # call GCN
            gcn_output = gcn(data)
            
            # call Final Prediction
            final_prediction = classifier(gcn_output.mean(dim=0))
            print(f"Predicted Sentiment: {final_prediction}")

# File path to the positive reviews file
file_path = 'D:/PhD_Work/Data/positive_reviews.txt'

# Process the file and save results to the database
process_file(file_path)

# Close the session
session.close()
