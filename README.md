# DSL Active Clustering Method

## Introduction
DSL (Data Skeleton Learning) is an active clustering method designed to process large-scale data effectively. This method focuses on enhancing efficiency and scalability when applying pairwise constraint-based active clustering across various fields, including data mining, knowledge annotation, and AI model pre-training. Our approach is structured around three primary objectives:

1. **Reduce Computational Costs:** By minimizing the computational demands associated with iterative clustering updates, we aim to streamline the processing of large datasets.
2. **Enhance Constraint Impact:** We strive to maximize the utility of user-provided constraints to reduce the annotation efforts required for precise clustering outcomes.
3. **Optimize Memory Usage:** Ensuring efficient algorithm performance in resource-constrained environments is crucial for practical deployments.

## Algorithm Overview
To achieve these objectives, we propose a novel graph-based active clustering algorithm. This algorithm incorporates:
- **Data Skeleton:** A sparse graph that represents relationships between data points.
- **Iterative Updates:** Another sparse graph that updates the data skeleton to refine connected subgraphs, facilitating nested cluster formation.

## Demonstration
A video demonstration shows the DSL algorithm reconstructing a macroscopic view of 3000 instances within the ImageNet Dataset.

[![Video demo](https://img.youtube.com/vi/wXiK-TzkmQE/0.jpg)](https://www.youtube.com/watch?v=wXiK-TzkmQE)

## Development Environment
Ensure your development environment is set up with the following specifications:
- **Python:** 3.11.4
- **NetworkX:** 3.1
- **Matplotlib:** 3.7.2
- **NumPy:** 1.24.4
- **Scikit-Learn:** 1.2.2
