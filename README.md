# Temporal GNN Recommender

This project explores next-item recommendation from time-ordered user-item interactions using graph-based models.

## Project goal

The goal is to model recommendation as a temporal graph problem:

- users and items form a bipartite graph
- interactions happen over time
- the task is to predict the next item a user will interact with

The project is designed to stay lightweight enough for local development while still being easy to run in Google Colab for larger experiments.

## Planned workflow

1. Load the MovieLens 100K dataset
2. Sort interactions by timestamp
3. Create train, validation, and test splits by time
4. Build simple recommendation baselines
5. Build a graph-based recommendation model
6. Compare results with ranking metrics such as Hit@K and MRR

## Starter structure

```text
temporal-gnn-recommender/
|- data/
|- notebooks/
|- src/
|  |- __init__.py
|  |- baselines.py
|  |- data_loader.py
|  |- evaluate.py
|  |- graph_builder.py
|  |- models.py
|  |- preprocess.py
|  |- train.py
|- .gitignore
|- README.md
|- requirements.txt
```

## First milestone

The first milestone is getting a clean preprocessing pipeline working for MovieLens 100K:

- load raw interactions
- rename columns clearly
- sort by timestamp
- split by time

## Notes

- Local debugging can be done on small samples
- Full experiments can be run in Colab
- The first version will focus on correctness and clarity before model complexity
