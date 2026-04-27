# Temporal GNN Recommender

This project explores next-item recommendation from time-ordered user-item interactions using graph-based models.

## Project goal

The goal is to model recommendation as a temporal graph problem:

- users and items form a bipartite graph
- interactions happen over time
- the task is to predict the next item a user will interact with
- the first graph version maps users and items into one shared node-id space for a lightweight GNN-friendly representation

The project is designed to stay lightweight enough for local development while still being easy to run in Google Colab for larger experiments.

## Planned workflow

1. Load the MovieLens 100K dataset
2. Sort interactions by timestamp
3. Create train, validation, and test splits by time
4. Visualize a readable sample of the bipartite user-item graph
5. Build simple recommendation baselines
6. Build a graph-based recommendation model
7. Compare results with ranking metrics such as Recall@K and NDCG@K

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

## Graph visualization

After copying MovieLens `u.data` into `data/raw/u.data`, run:

```bash
python visualize_graph.py
```

This saves a small sampled bipartite graph to `outputs/bipartite_graph_sample.png`. The full graph is too dense to view directly, so the script samples active users and items to create a readable picture.

## Simple GNN baseline

The first graph model is a lightweight LightGCN-style recommender implemented with PyTorch sparse matrix operations. It uses the train split to build the bipartite graph, learns user/item embeddings, and evaluates future interactions with Hit@10.
The main metrics are Recall@10 and NDCG@10.

```bash
python train_gnn.py
```

This model is intentionally small so it can run locally on CPU before moving larger experiments to Colab.

The training script also supports simple experiment settings:

```bash
python train_gnn.py --epochs 10 --embedding-dim 16 --num-layers 1 --lr 0.01
```

By default, graph edges are weighted with both rating and recency information. The time component uses approximate one-month buckets from the start of the training split.

```bash
python train_gnn.py --edge-weighting uniform
python train_gnn.py --edge-weighting rating
python train_gnn.py --edge-weighting time
python train_gnn.py --edge-weighting rating_time
```

By default, the recommender keeps ratings `>= 4` so the task is to recommend future movies that users liked. You can change the threshold with:

```bash
python train_gnn.py --min-rating 4
```

Each run appends metrics to `outputs/gnn_results.csv` for comparison.

## GraphSAGE comparison

The repo also includes a small GraphSAGE-style recommender using weighted mean neighborhood aggregation:

```bash
python train_graphsage.py --epochs 10 --embedding-dim 16 --num-layers 2 --lr 0.01 --edge-weighting rating_time --min-rating 4
```

This GraphSAGE version uses:
- learned node ID embeddings
- node-type indicators for user vs item
- simple node statistics such as interaction count, average rating, and average recency

GraphSAGE experiment runs are saved to `outputs/graphsage_results.csv`.

## Classic GCN comparison

The repo also includes a simpler classic GCN-style recommender that uses:
- learned node ID embeddings
- fixed node features
- normalized graph propagation
- linear layers + ReLU

```bash
python train_gcn.py --epochs 10 --embedding-dim 16 --num-layers 2 --lr 0.01 --edge-weighting rating_time --min-rating 4 --similarity-top-k 5 --min-similarity 0.1
```

GCN experiment runs are saved to `outputs/gcn_results.csv`.

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
