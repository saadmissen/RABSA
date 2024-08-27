This repository contains code for ABSA. Python code is saved in different files. It may contain files with different version of same work. 
We have tried our best to keep the code clean and remove unnecessary lines used during coding, debugging and testing
The code uses a more cleaner code of the data collection present at https://github.com/saadmissen/RABSA-Data-Collection.git
aspect_SO_finder.py is accessing WordNet dictionary to find the real sense of the word in a sentence. It uses BERt model to create context embeddings
All *pipeline*.py modules the whole pipeline of the proposed system in different ways. Different versions exist
Baseline_1.py is used to check the results of baselines on the cleaned data collection. 
We have tried to code more readable by using constants on different  places i.e. hyperparametrs, database connections, file handles etc
We did not use the same repository for data collection and code because code is being shared only for reviewers and also we had already shared the data collection DOI so we do not want any change

