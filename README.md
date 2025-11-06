Hello this is the AI home decor Project that I have created a Demo For. This software does have bugs and sometimes provides innaccurate results. 
This innaccuracy can be overcome by using good quality annotated data. 

Below is the working principle of my designed system 

Frontend (React)
   │
   ├── Sends layout requests → /solve-layout
   │
Backend (FastAPI)
   ├── Layout Solver
   │     ├── Uses Genetic Algorithm (GA) or Simulated Annealing (SA)
   │     └── Applies geometry + penalty functions (no overlap, walkway, etc.)
   │
   ├── Neural Network (GNN)
   │     ├── Learns object–object spatial relationships from saved layouts
   │     └── Generates seed positions for next auto-arrange requests
   │
   └── Finalization Endpoint
         ├── Stores accepted layouts as “training data”
         └── Incrementally retrains the model



Initially, I used classical optimization — Genetic Algorithm (GA) and Simulated Annealing (SA) — to explore possible placements.
These were guided by a weighted penalty function that scores layouts based on compactness, overlap area, walkway spacing, 
and aesthetic factors like centering or wall preference.

To enable learning, I integrated a Graph Neural Network (GNN) that models pairwise spatial relationships between objects — each 
node represents an item (sofa, TV, table), and edges encode relationships like ‘facing’, ‘distance’, or ‘adjacency’

The GNN learns embeddings for item types and predicts relative placements.
When a user saves a layout as ‘final’, that data is added to the training set, and the model is retrained incrementally.

### Why the Inaccuracy and how can It be rectified?
With only 5–10 training examples, the GNN didn’t have enough diversity to generalize spatial arrangements.
It tended to minimize penalties trivially by stacking objects near the center, which technically reduces some metrics but isn’t visually appealing.
This revealed a common issue in data-driven layout learning — small data and unbalanced losses can lead to degenerate configurations

How to Improve the results ?

FIrst I would need to create thousands of synthetic data which I would use to pretrain the GNN model. These synthetic data will be user verified for fine tuining. 
In this stage I would also include doors and windows features so that the GNN can create spatial relations between these features too. 



# Backend: Python, FastAPI, Pydantic, NumPy, PyTorch
# Frontend: React, HTML5 Canvas/drag-drop
# Optimization: Genetic Algorithm, Simulated Annealing
# ML: Graph Neural Networks (PyTorch Geometric or custom)
# Persistence: JSON prototype storage, online incremental retraining
# Constraints: No-overlap, compactness, TV-view distance, wall preference
# Learning loop: Weight update (gradient + heuristics), rolling dataset of 20 samples


I have also tried to experiment by including an LLM to automatically generate hard and soft constraints and add it to the existing file. But my OpenAI api supscription 
from my university was not working and I couldnt get the key to utilize it. 
