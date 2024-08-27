import nltk
from nltk.corpus import wordnet as wn
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sqlalchemy import create_engine, Column, String, Float, Integer, MetaData, Table
from sqlalchemy.orm import sessionmaker

# Database connection (adjust the connection string as needed)
# ASK sir how to do this when wrong

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

# Function to create context embeddings using BERT
def create_context_embedding(sentence, target_word):
    inputs = tokenizer(sentence, return_tensors='pt')
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    word_index = tokenizer.convert_ids_to_tokens(inputs.input_ids[0]).index(target_word)
    return embeddings[0, word_index, :]

# Function to calculate cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Function to find the most suitable sense of a word using WordNet and BERT
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
           # print best_sense
    return best_sense, highest_similarity

# Function to filter and process only verbs, adverbs, and adjectives
def filter_and_process_words(sentence):
    words = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(words)
   # print pos_tags
    
    relevant_words = [word for word, pos in pos_tags if pos.startswith('VB') or pos.startswith('RB') or pos.startswith('JJ')]
    
    for word in relevant_words:
        best_sense, similarity = find_best_sense(sentence, word)
        if best_sense:
            save_to_db(word, best_sense.name(), similarity)

# Function to save the results to the database
def save_to_db(word, best_sense, similarity):
    insert_stmt = word_senses_table.insert().values(word=word, best_sense=best_sense, similarity=similarity)
    session.execute(insert_stmt)
    session.commit()

# @ASMa- avoid using original data- this is on updated - Function to process a file and save results to the database
def process_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            sentence = line.strip()
            filter_and_process_words(sentence)

# Data folder contains updated dataset- File path to the positive reviews file
file_path = 'D:/PhD_Work/Data/positive_reviews.txt'

# Process the file and save results to the database
process_file(file_path)

# Close the session
session.close()



