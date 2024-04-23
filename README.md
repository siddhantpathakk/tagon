# TAGON: Temporal Attention Graph-Optimized Networks


As the digital world evolves, sequential recommendation systems are becoming crucial for enhancing user engagement by personalizing the digital experience. These systems analyze a user's interaction history and context, predicting future actions with high relevance. Utilizing deep learning techniques, such as Graph Neural Networks, these models handle complex user-item relationships effectively, especially when integrated with temporal data.

This research introduces Temporal Attention Graph-Optimized Networks, a robust architecture that incorporates temporal dynamics into graph-based models. By capturing the sequence and timing of interactions, the proposed model aims to provide more accurate predictions, significantly improving the performance of recommendation systems in various domains like e-commerce and social networking.

This is the official repository for Temporal Attention Graph-Optimized Networks. Submitted as part of the Final Year Project - SCSE23-0406 by Pathak Siddhant (U2023715K).

## Citation

Please cite our paper if using this code.

Pathak, S. (2024). Temporal attention graph-optimized networks for sequential recommendation. Final Year Project (FYP), Nanyang Technological University, Singapore. https://hdl.handle.net/10356/175250
## Run Locally

Download and preprocess the dataset
```bash
  cd data

  # Amazon Reviews datasets

  wget https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Baby_5.json.gz 
  wget https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Digital_Music_5.json.gz
  wget https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_5.json.gz
  wget https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz
  wget https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Tools_and_Home_Improvement_5.json.gz

  python process_dataset/process_amazon.py

  # Movielens-100K dataset

  wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
  unzip ml-100k

  python process_dataset/process_ml100k.py
```

Clone the project

```bash
  git clone https://github.com/siddhantpathakk/tagon.git
```

Go to the project directory

```bash
  cd tagon
```


Install dependencies

```bash
  conda create --name tagon_venv python=3.10
  conda activate tagon_venv
  pip install -r src/requirements.txt
```

Start the training (modify the parameters in the shell script accordingly)

```bash
  sh run_training.sh
```


## Authors

- [@siddhantpathakk](https://www.github.com/siddhantpathakk)


## Feedback

If you have any feedback, please reach out to us at fake@fake.com

