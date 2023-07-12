# Exercises

library(igraph)

head(friends)

# Load edgelist df as graph
friends.mat <- as.matrix(friends)
g <- graph.edgelist(friends.mat,
                    directed = FALSE)
plot(g)

# Vertices (Nodes) and Edges of graph
V(g); E(g)
gsize(g) # no. of edges
gorder(g) # no. of nodes

# Create new vertex and edge attributes

genders <- c("M", "F", "F", "M", "M", "M", "F", "M",
             "M", "F", "M", "F", "M", "F", "M", "M")

ages <- c(18, 19, 21, 20, 22, 18, 23, 21,
          22, 20, 20, 22, 21, 18, 19, 20)

g <- set_vertex_attr(g, "gender", value = genders)
g <- set_vertex_attr(g, "age", value = ages)

vertex_attr(g) # get vertex attr
V(g)[[1:5]] # view attributes of first 5 nodes

hours <- c(1, 2, 2, 1, 2, 5, 5, 1, 1,
           3, 2, 1, 1, 5, 1, 2, 4, 1,
           3, 1, 1, 1, 4, 1, 3, 3, 4)

g <- set_edge_attr(g, "hours", value = hours)
edge_attr(g)

E(g)[[.inc('Britt')]] # Find all edges that include "Britt"
E(g)[[hours>=4]]  # Find all pairs that spend 4 or more hours together per week

# Visualizing attributes

g1 <- graph_from_data_frame(d = friends1_edges,
          vertices = friends1_nodes,
          directed = FALSE)

E(g1)[[hours >= 5]]  # Subset edges greater than or equal to 5 hours
V(g1)$color <- ifelse(V(g1)$gender == "F", # Set vertex color by gender
                      "orange", "dodgerblue")

# Plot the graph
plot(g1, vertex.label.color = "black")

## Exploring igraph network layouts

plot(g1, vertex.label.color = "black", layout = layout_in_circle(g1))
plot(g1, vertex.label.color = "black", layout = layout_with_fr(g1))

m <- layout_as_tree(g1)
plot(g1, vertex.label.color = "black", layout = m)

m1 <- layout_nicely(g1)
plot(g1, vertex.label.color = "black", layout = m1)

## Visualizing edges

# Create a vector of weights based on the number of hours each pair spend together
w1 <- E(g1)$hours

# Plot the network varying edges by weights
m1 <- layout_nicely(g1)
plot(g1, 
     vertex.label.color = "black", 
     edge.color = 'black',
     edge.width = w1,
     layout = m1)

# Create a new igraph object by deleting edges that are less than 2 hours long 
g2 <- delete_edges(g1, E(g1)[hours < 2])

# Plot the new graph 
w2 <- E(g2)$hours
m2 <- layout_nicely(g2)

plot(g2, 
     vertex.label.color = "black", 
     edge.color = 'black',
     edge.width = w2,
     layout = m2)

# Initialize new directed graph
g <- graph_from_data_frame(measles, directed = TRUE)

is.directed(g); is.weighted(g) # Check directed / weighted graph
table(head_of(g, E(g))) # Check edge origins

# Exploratory plot of graph

plot(g, 
    vertex.label.color = "black", 
    edge.color = 'gray77',
    vertex.size = 0,
    edge.arrow.size = 0.1,
    layout = layout_nicely(g))

# Is there an edge going from vertex 184 to vertex 178?
g['184', '178']

# Is there an edge going from vertex 178 to vertex 184?
g['178', '184']

incident(g, '184', mode = c("all")) # Show all edges going to or from vertex 184
incident(g, '184', mode = c("out")) # Show all edges going out from vertex 184

##  Neighboring nodes

neighbors(g, '12', mode = c('all'))

# Identify other vertices that direct edges towards vertex 12
neighbors(g, '12', mode = c('in'))

# Identify any vertices that receive an edge from vertex 42 and direct an edge to vertex 124
n1 <- neighbors(g, '42', mode = c('out'))
n2 <- neighbors(g, '124', mode = c('in'))
intersection(n1, n2)

## Distances between vertices

farthest_vertices(g) # diameter / longest traversal length
get_diameter(g) # path sequence between furthest vertices

# Identify vertices that are reachable within two connections from vertex 42
ego(g, 2, '42', mode = c('out'))

# Identify vertices that can reach vertex 42 within two connections
ego(g, 2, '42', mode = c('in'))

## Identifying key vertices

g.outd <- degree(g, mode = c("out")) # Calculate the out-degree of each vertex

table(g.outd) # Summary of out-degree
hist(g.outd, breaks = 30) # out-degree histogram

which.max(g.outd) # vertex with maximum out-degree

## Betweenness Centrality Measure (information flow)

# Calculate betweenness of each vertex
g.b <- betweenness(g, directed = TRUE)

# Show histogram of vertex betweenness
hist(g.b, breaks = 80)

# Create plot with vertex size determined by betweenness score
plot(g, 
     vertex.label = NA,
     edge.color = 'black',
     vertex.size = sqrt(g.b)+1, # normalization allows all nodes to be viewed, 
     edge.arrow.size = 0.05, # ... relative importance to be identifiable
     layout = layout_nicely(g))

## Visualizing important nodes and edges

# Make an ego graph (subset of network comprised of all nodes connected to node 184)
g184 <- make_ego_graph(g, diameter(g),
                       nodes = '184',
                       mode = c("all"))[[1]]

# Get a vector of geodesic distances of all vertices from vertex 184 
dists <- distances(g184, "184")

# Create a color palette of length equal to the maximal geodesic distance plus one.
colors <- c("black", "red", "orange",
                   "blue", "dodgerblue", "cyan")

# Set color attribute to vertices of network g184.
V(g184)$color <- colors[dists+1]

# Visualize the network based on geodesic distance from vertex 184 (patient zero).
plot(g184, 
    vertex.label = dists, 
    vertex.label.color = "white",
    vertex.label.cex = .6,
    edge.color = 'black',
    vertex.size = 7,
    edge.arrow.size = .05,
    main = "Geodesic Distances from Patient Zero"
)

## Forrest Gump Network

head(gump)

# Make an undirected network
g <- graph_from_data_frame(gump, directed = FALSE)

# Identify key nodes using eigenvector centrality
# Note: high eigenvector centrality => highly connected vertex

g.ec <- eigen_centrality(g)
which.max(g.ec$vector) # identify node with highest eigen_centrality score

# Plot Forrest Gump Network
plot(g,
     vertex.label.color = "black", 
     vertex.label.cex = 0.6,
     vertex.size = 25*(g.ec$vector),
     edge.color = 'gray88',
     main = "Forrest Gump Network"
)

# Get density of a graph (proportion of current edges vs all potential edges)
gd <- edge_density(g)

# Get the diameter of the graph g (longest path distance)
diameter(g, directed = FALSE)

# Get the average path length of the graph g
g.apl <- mean_distance(g, directed = FALSE)
g.apl

## Random Graphs
# Create one random graph with the same number of nodes and edges as g
g.random <- erdos.renyi.game(n = gorder(g), p.or.m = gd, type = "gnp")
g.random

plot(g.random)

# Get density of new random graph `g.random`
edge_density(g.random)

# Get the average path length of the random graph g.random
mean_distance(g.random, directed = FALSE)

## Randomization Test

# Generate 1000 random graphs
gl <- vector('list', 1000)

for(i in 1:1000){
  gl[[i]] <- erdos.renyi.game(n = gorder(g), p.or.m = gd, type = "gnp")
}

# Calculate average path length of 1000 random graphs
gl.apls <- unlist(lapply(gl, mean_distance, directed = FALSE))

# Plot the distribution of average path lengths
hist(gl.apls, xlim = range(c(1.5, 6)))
abline(v = g.apl, col = "red", lty = 3, lwd = 2)

# Calculate the proportion of graphs with an average path length lower than our observed
mean(gl.apls < g.apl)

"""
Conclusion of Analysis:

The Forrest Gump network is far more interconnected than we would expect
by chance as zero random networks have an average path length smaller than the
Forrest Gump network's average path length.
"""

## Network Substructures (Triangles, Transitivity and Cliques)

matrix(triangles(g), nrow = 3) # matrix of all possible triangles

# Count the number of triangles that vertex "BUBBA" is in.
count_triangles(g, vids='BUBBA')

# Calculate global transitivity of the network.
g.tr <- transitivity(g)
g.tr

# Calculate the local transitivity for vertex BUBBA.
transitivity(g, vids='BUBBA', type = "local")

## Transitivity Randomizations

# Calculate average transitivity of 1000 random graphs
gl.tr <- lapply(gl, transitivity)
gl.trs <- unlist(gl.tr)

# Get summary statistics of transitivity scores
summary(gl.trs)

# Calculate the proportion of graphs with a transitivity score higher than Forrest Gump's network
mean(gl.trs > g.tr)

## Cliques

# Identify the largest cliques in the network
largest_cliques(g)

# Determine all maximal cliques in the network and assign to object 'clq'
clq <- max_cliques(g)

# Calculate the size of each maximal clique.
table(unlist(lapply(clq, length)))

# Visualizing largest cliques
# Assign largest cliques output to object 'lc'
lc <- largest_cliques(g)

# Create two new undirected subgraphs, each containing only the vertices of each largest clique.
gs1 <- as.undirected(subgraph(g, lc[[1]]))
gs2 <- as.undirected(subgraph(g, lc[[2]]))

# Plot the two largest cliques side-by-side

par(mfrow=c(1,2)) # To plot two plots side-by-side

plot(gs1,
    vertex.label.color = "black", 
    vertex.label.cex = 0.9,
    vertex.size = 0,
    edge.color = 'gray28',
    main = "Largest Clique 1",
    layout = layout.circle(gs1)
)

plot(gs2,
    vertex.label.color = "black", 
    vertex.label.cex = 0.9,
    vertex.size = 0,
    edge.color = 'gray28',
    main = "Largest Clique 2",
    layout = layout.circle(gs2)
)

## Close Relationships

# Plot the network
plot(g1)

# Convert the gender attribute into a numeric value
values <- as.numeric(factor(V(g1)$gender))

# Calculate the assortativity of the network based on gender
assortativity(g1, values)

# Calculate the assortativity degree of the network
assortativity.degree(g1, directed = FALSE)

# Calculate the observed assortativity
observed.assortativity <- assortativity(g1, values)

# Calculate the assortativity of the network randomizing the gender attribute 1000 times
results <- vector('list', 1000)
for(i in 1:1000){
  results[[i]] <- assortativity(g1, sample(values))
}

# Plot the distribution of assortativity values and add a red vertical line at the original observed value
hist(unlist(results))
abline(v = observed.assortativity, col = "red", lty = 3, lwd=2)

# Make a plot of the chimp grooming network
plot(g,
     edge.color = "black",
     edge.arrow.size = 0.3,
     edge.arrow.width = 0.5)

# Calculate the reciprocity of the graph
reciprocity(g)

## Community Detection

# Perform fast-greedy community detection on network graph
kc = fastgreedy.community(g)
sizes(kc) # Determine sizes of each community
membership(kc) # Determine which individuals belong to which community

plot(kc, g) # Plot the community structure of the network

# Perform edge-betweenness community detection on network graph
gc = edge.betweenness.community(g)
sizes(gc)

# Plot community networks determined by fast-greedy and edge-betweenness methods side-by-side
par(mfrow = c(1, 2)) 
plot(kc, g)
plot(gc, g)

## Interactive Network Visualizations with threejs
library(threejs)

# Set a vertex attribute called 'color' to 'dodgerblue' 
g <- set_vertex_attr(g, "color", value = "dodgerblue")

# Redraw the graph and make the vertex size 1
graphjs(g, vertex.size = 1)

# Create numerical vector of vertex eigenvector centralities 
ec <- as.numeric(eigen_centrality(g)$vector)

# Create new vector 'v' that is equal to the square-root of 'ec' multiplied by 5
v <- 5*sqrt(ec)

# Plot threejs plot of graph setting vertex size to v
graphjs(g, vertex.size = v)

# Create an object 'i' containin the memberships of the fast-greedy community detection
i <-  membership(kc)

# Check the number of different communities
sizes(kc)

# Add a color attribute to each vertex, setting the vertex color based on community membership
g <- set_vertex_attr(g, "color", value = c("yellow", "blue", "red")[i])

# Plot the graph using threejs
graphjs(g)
