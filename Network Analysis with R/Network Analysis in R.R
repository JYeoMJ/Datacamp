# Network Analysis in R

# Introduction to `igraph` package

library(igraph)

g <- graph.edgelist(as.matrix(df),
                    directed = FALSE)

V(g)
E(g)
gorder(g)
gsize(g)

plot(g)

# Vertex Attributes

g <- set_vertex_attr(
  g, "age", value = c(20,25,21,23,24,23,22)
)

vertex_attr(g)

# Edge Attributes

g <- set_edge_attr(
  g, "frequency", value = c(2,1,1,1,3,2,4)
)

edge_attr(g)

graph_from_data_frame(d = edges.df,
                      vertices = vertices.df,
                      directed = FALSE)

# Subsetting Networks
E(g)[[.inc('E')]]
E(g)[[frequency >= 3]]

# Network visualization
V(g)$color <- ifelse(
  V(g)$age > 22, "red", "white") # vertex colors

plot(g, vertex.label.color = "black") # label colors

plot(g, layout = layout.fruchterman.reingold(g))

# Directed Networks
is.directed(g)
is.weighted(g) # check if weighted graph

# Concept of out-degree and in-degree (outgoing and incoming edges)

g['A','E'] # check existence of edge
incident(g, 'A', mode = c('all')) # show all edges to or from node of interest
head_of(g, E(g)) # find starting vertex of all edges

# Relationship between vertices

neighbors(g, "F", mode = c("all")) # neighbors of node `F`

x <- neighbors(g, "F", mode = c("all"))
y <- neighbors(g, "D", mode = c("all"))

intersection(x,y) # common neighbors

# Concept of Paths and Ego (N-hop neighbors)

farthest_vertices(g) # returns diameter of graph and corresponding nodes (longest path traversal)
get_diameter(g) # returns exact sequence of node traversal (connections)

ego(g, 2, 'F', mode = c('out')) # vertices reachable within N steps from given vertex

# Identifying Important/Influential Vertices

# Key Measures: degree, betweenness, eigenvector centrality, closeness centrality, pagerank centrality

degree(g, # no. of adjacent edges 
       mode = c("all", "out", "in", "total")) # in-degree, out-degree or total degree

# relation to information flow, no. of shortest paths going through a node
betweenness(g, directed = TRUE, normalized = TRUE) # normalize betweenness scores

## Network Structures

# Eigenvector centrality
eigen_centrality(g)$vector # measure of how well-connected a vertex is

# Density (ratio of actual no. of edges and largest possible no. of edges in graph)
edge_density(g) # measure of how interconnected a network is

# Average path length (mean of lengths of shortest paths between all pairs)
mean_distance(g, directed = FALSE) # another measure of interconnectedness

## Network Randomizations

# Concept of Random Graphs

erdos.renyi.game(n = gorder(g), # no. vertices in graph
                 p.or.m = edge_density(g), # edge probability (edge density)
                 type = "gnp")

# Generate 1000 random graphs

gl <- vector('list', 1000)

for (i in 1:1000){
  gl[[i]] <- erdos.renyi.game(
    n = gorder(g),
    p.or.m = edge_density(g),
    type = "gnp"
  )
}

# Calculate average path length of 1000 random graphs
gl.apls <- unlist(
  lapply(gl, mean_distance, directed = FALSE)
)

# Plot distribution of average path lengths
hist(gl.apls, breaks = 20)

abline(
  v = mean_distance(g, directed = FALSE),
  col = "red", lty = 3,lwd = 2
)

## Network substructures

triangles(g) # closed triad i.e. set of 3 vertices fully connected (3 edges)

# Global transitivity (measure of interconnectedness of a group of 3 vertices)
transitivity(g) # probability that the adjacent vertices of a vertex are connected.
# Alt defn: proportion of all possible triangles in network that are closed

count_triangles(g, vids = 'A') # no. of triangles a vertex is part of

transitivity(g, vids = 'A', # local connectivity centered about given vertex
             type = 'local')

# Cliques (complete graph, every vertex connected to all other vertices, all triangles are closed)

largest_cliques(g) # all largest cliques in the input graph
# A clique is largest if there is no other clique including more vertices.

max_cliques(g) # all maximal cliques in the input graph
# A clique is maximal if it cannot be extended to a larger clique.

## Identifying Special Relationships

## (Close Relationships) Assortativity and Reciprocity
# Assortativity - how likely two vertices sharing similar attributes are linked

assortativity(g, values) # values = attributes (note cetegorical attributes need to be converted to levels)

# high degree vertices connect preferentially to other vertices with high degree
assortativity.degree(g, directed = TRUE)

# Reciprocity (for directed networks) 
# proportion of edges that are symmetrical (mutual connections)
reciprocity(g)

## Community Detection in Networks
# i.e. identifying dense clusters of nodes

# Modularity score: index of how interconnected edges are within vs between communities

x <- fastgreedy.community(g) # modularity-based method, successively add nodes

length(x) # no. clusters
sizes(x) # community sizes (no. nodes in each cluster)
membership(x) # numeric vector of community assignments to each node

edge.betweenness.community(g) 
# divides network into smaller pieces until it finds edges that act as "bridges" between communities

plot(x, g) # plot of graph partitioned by clusters

## Interactive Network Visualizations (threejs)

library(threejs)
graphjs(g)

# Adding attributes
g <- set_vertex_attr(g, "label", value = V(g)$name)
g <- set_vertex_attr(g, "color", value = "mistyrose")

graphjs(g, vertex.size = 1)

# Visualizing communities
x = edge.betweenness.community(g)
i <- membership(x)

g <- set_vertex_attr(g, "color", value = c("yellow", "blue", "red")[i])

graphjs(g)
