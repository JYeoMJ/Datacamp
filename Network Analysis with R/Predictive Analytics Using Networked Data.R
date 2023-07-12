## Predictive Analytics using Networked Data

library(igraph)

# Collaboration Network

DataScienceNetwork <- data.frame( # Edgelist dataframe
  from = c('A', 'A', 'A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'D', 'E',
           'F', 'F', 'G', 'G', 'H', 'H', 'I'),
  to = c('B','C','D','E','C','D','D', 'G','E', 'F','G','F','G','I',
         'I','H','I','J','J')
)

g <- graph_from_data_frame(DataScienceNetwork, directed = FALSE)

pos <- cbind(c(2, 1, 1.5, 2.5, 4, 4.5, 3, 3.5, 5, 6),
             c(10.5, 9.5, 8, 8.5, 9, 7.5, 6, 4.5, 5.5, 4))

plot.igraph(g, edge.label = NA, edge.color = 'black', layout = pos,
            vertex.label = V(g)$name, vertex.color = 'white',
            vertex.label.color = 'black', vertex.size = 25)

# Add node attributes

V(g)$technology <-  c('R','R','?','R','R','R','P','P','P','P')

V(g)$color <- V(g)$technology

V(g)$color <- gsub('R',"blue3", V(g)$color)
V(g)$color <- gsub('P',"green4", V(g)$color)
V(g)$color <- gsub('?',"gray", V(g)$color)

## Application to Churn Prediction (Labelled Networks and network learning)

# Relational Neighbor Classifier

rNeighbors <- c(4,3,3,5,3,2,3,0,1,0) # no. neighbors using R
pNeighbors <- c(0,0,1,1,0,2,2,3,3,2) # no. neighbors using P (i.e. Python)

rRelationalNeighbor <- rNeighbors / (rNeighbors + pNeighbors) # compute ratio of neighbors
rRelationalNeighbor

# Challenges of Network-Based Inference

# 1. Splitting the Data

set.seed(1001)
sampleVertices <- sample(1:10, 6, replace=FALSE)
plot(induced_subgraph(g, V(g)[sampleVertices])) # training network
plot(induced_subgraph(g, V(g)[-sampleVertices])) # test network

# Problem: subnetworks different from original network and each other

# 2. Observations are not iid (labels of one node dependent on label of neighbors)
# 3. Collective Inference (given semi-labeled network, can we predict label for all unlabeled nodes?)

# Probabilistic relational neighbor classifier
# Nodes of neighbors have probability of belonging to each of two classes

# probability churn (C)
(0.9 + 0.2 + 0.1 + 0.4 + 0.8) / 5

# probability non-churn (NC)
(0.1 + 0.8 + 0.9 + 0.6 + 0.2) / 5

# i.e. average churn / non-churn probabilities of neighbors!

## Homophilic Networks

names <- c('A','B','C','D','E','F','G','H','I','J')
tech <- c(rep('R', 6),rep('P', 4))
DataScientists <- data.frame(name = names, technology = tech)
DataScienceNetwork <- data.frame( # Edgelist dataframe
  from = c('A', 'A', 'A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'D', 'E',
           'F', 'F', 'G', 'G', 'H', 'H', 'I'),
  to = c('B','C','D','E','C','D','D', 'G','E', 'F','G','F','G','I',
         'I','H','I','J','J'),
  label = c(rep('rr', 7),'rp','rr','rr','rp','rr','rp','rp', rep('pp', 5))
)

g <- graph_from_data_frame(DataScienceNetwork, directed = FALSE)

# Add technology as a node attribute

V(g)$label <- as.character(DataScientists$technology)
V(g)$color <- V(g)$label
V(g)$color <- gsub('R', "blue3", V(g)$color)
V(g)$color <- gsub('P', "green4", V(g)$color)

# Edge attributes

E(g)$color <- E(g)$label
E(g)$color <- gsub('rp', "red", E(g)$color)
E(g)$color <- gsub('rr', "blue3", E(g)$color)
E(g)$color <- gsub('pp', "green4", E(g)$color)

# Visualize network

pos <- cbind(c(2, 1, 1.5, 2.5, 4, 4.5, 3, 3.5, 5, 6),
             c(10.5, 9.5, 8, 8.5, 9, 7.5, 6, 4.5, 5.5, 4))

plot(g, edge.label = NA, vertex.label.color = "white",
     layout = pos, vertex.size = 25)

# Counting edge types

edge_rr <- sum(E(g)$label == 'rr') # R edges
edge_pp <- sum(E(g)$label == 'pp') # Python edges
edge_rp <- sum(E(g)$label == 'rp') # Cross label edges

# Network connectance
p <- 2 * edges / nodes*(nodes-1)

# Note: No. edges in fully connected network is `choose(nodes, 2)`

# Dyadicity (measures connectedness between nodes of same label)
p <- 2 * 19 / (10*9)
expectedREdges <- 6 * 5 / 2 * p
expectedPEdges <- 4 * 3 / 2 * p
dyadicityR <- rEdges / expectedREdges
dyadicityP <- pEdges / expectedPEdges

# Heterophilicity (connectedness between nodes of opposite labels)

# ... review this part, I zoned out

# Neighborhood features

degree(g) # no. connected nodes
neighborhood.size(g, order = 2) # second-order degree
count_triangles(g)

# Centrality features

betweenness(g)
closeness(g)
transitivity(g, type = 'local')

# Adjacency matrices

A <- get.adjacency(g)

# Link based features
preference <- c(1,1,1,1,1,1,0,0,0,0) # R preference vector
rNeighbors <- A %*% preference # no. neighbors that prefer R
as.vector(rNeighbors)

# Neighborhood features
age <- c(23,65,33,36,28,45,41,24,38,39)
degree <- degree(g) # no. neighbors
averageAge <- A %*% age / degree # average age of neighbors

# PageRank Algorithm
page.rank(g)
page.rank(g, personalized = c(1,0,0,0,0,0,0,0,0,0))






































