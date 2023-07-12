## CASE STUDIES: NETWORK ANALYSIS IN R

## EXPLORING GRAPHS THROUGH TIME -------

# Load required library and dataset
library(igraph)
library(dplyr)

amzn_raw <- read.csv(".csv")
head(amzn_raw)

amzn_g <- amzn_raw %>%
  filter(date == "2003-03-02") %>%
  select(from, to) %>%
  graph_from_data_frame(directed = TRUE)

gorder(amzn_g)
gsize(amzn_g)

# Visualzing the subgraph
sg <- induced_subgraph(amzn_g, 1:500)
sg <- delete.vertices(sg, degree(sg) == 0)
plot(sg, vertex.label = NA, edge.arrow.width = 0,
     edge.arrow.size = 0, margin = 0, vertex.size = 2)

amzn_g

# Finding Dyads and Triads
dyad_census(amzn_g)
triad_census(amzn_g)
edge_density(amzn_g)

# Clustering and Reciprocity
actual_recip <- reciprocity(amzn_g)
n_nodes <- gorder(amzn_g)
edge_dens <- edge_density(amzn_g)

# Run the simulation
simulated_recip <- rep(NA, 1000)
for(i in 1:1000) {
  # Generate an Erdos-Renyi simulated graph
  simulated_graph <- erdos.renyi.game(n_nodes, edge_dens, directed = TRUE)
  # Calculate the reciprocity of the simulated graph
  simulated_recip[i] <- reciprocity(simulated_graph)
}

# Calculate quantile of simulated reciprocity
quantile(simulated_recip , c(0.025, 0.5, 0.975))

# Identifying important products (node degrees)
out_degree <- degree(amzn_g, mode = "out")
in_degree <- degree(amzn_g, mode = "in")

# View distribution of node degrees
table(out_degree)
table(in_degree)

# Filter for important products
is_important <- out_degree > 3 & in_degree < 3
imp_prod <- V(amzn_g)[is_important]
print(imp_prod)

# Select the from and to columns from ip_df
ip_df_from_to <- ip_df[c("from","to")]

# Create a directed graph
ip_g <- graph_from_data_frame(ip_df_from_to, directed = TRUE)

# Set the edge color. If salesrank.from is less than or 
# equal to salesrank.to then blue else red.
edge_color <- ifelse(
  ip_df$salesrank.from <= ip_df$salesrank.to, 
  yes = "blue", 
  no = "red"
)

# Plot a graph of ip_g
plot(
  ip_g, 
  edge.color = edge_color,
  edge.arrow.width = 1, edge.arrow.size = 0, edge.width = 4, 
  vertex.label = NA, vertex.size = 4, vertex.color = "black"
)

legend(
  "bottomleft", 
  fill = unique(edge_color), 
  legend = c("Lower to Higher Rank", "Higher to Lower Rank"), cex = 0.7
)

# Exploring temporal structure

# Loop over time graphs calculating out degree
degree_count_list <- lapply(time_graph, degree, mode = "out")

# Flatten it
degree_count_flat <- unlist(degree_count_list)

degree_data <- data.frame(
  degree_count = degree_count_flat,
  vertex_name = names(degree_count_flat),
  date = rep(d, lengths(degree_count_list))
)

important_vertices <- c(1629, 132757, 117841)

# Filter for rows where vertex_name is in set of important vertices
important_degree_data <- degree_data %>% 
  filter(vertex_name %in% important_vertices)

# Plot degree_count vs. date, colored by vertex_name 
ggplot(important_degree_data, aes(x = date, y = degree_count, color = vertex_name)) + 
  geom_path()

# Plotting Metrics Over Time

# Transitivity by graph
transitivity_by_graph <- data.frame(
  date = d,
  metric = "transitivity",
  score = sapply(all_graphs, transitivity)
)

# Calculate reciprocity by graph
reciprocity_by_graph <- data.frame(
  date = d,
  metric = "reciprocity",
  score = sapply(all_graphs, reciprocity)
  )

# Bind datasets by row
metrics_by_graph <- bind_rows(transitivity_by_graph, reciprocity_by_graph)
metrics_by_graph

# Plot score vs date, colored by metric
ggplot(metrics_by_graph, aes(x = date, y = score, color = metric)) +
  geom_path()

## Twitter Network Analysis -------

library(stringr)
raw_tweets <- read.csv('rstatstweets.csv',
                       stringsAsFactors = FALSE)

# Building Graph
all_sn <- unique(raw_tweets$screen_name) # get all screen names
retweet_graph <- graph.empty()
retweet_graph <- retweet_graph + vertices(all_sn) # add screen names as vertices

# Extract name and add edges
for (i in 1:dim(raw_tweets)[1]){
  rt_name <- find_rt(raw_tweets$tweet_text[i]) # Extract retweet name
  
  # If there is a name, add an edge
  if(!is.null(rt_name)){
    # Check to make sure vertex exists, if not, add it
    if(!rt_name %in% all_sn){
      retweet_graph <- retweet_graph + vertices(rt_name)
    }
    # add edge
    retweet_graph <- retweet_graph +
      edges(c(raw_tweets$screen_name[i], rt_name))
  }
}

# Cleaning the graph
sum(degree(retweet_graph) == 0) # count vertices with degree 0

# Trim and simplify
retweet_graph <- simplify(retweet_graph)
retweet_graph <- delete.vertices(retweet_graph,
                                 degree(retweet_graph) == 0)

# Visualization on graph

gorder(retweet_graph) # no. nodes
gsize(retweet_graph) # no. edges
graph.density(retweet_graph) # density

plot(retweet_graph)

# Visualize nodes by degree

in_deg <- degree(retweet_graph, mode = "in")
out_deg <- degree(retweet_graph, mode = "out")

# Nodes of 3 types:
# high retweeters and highly retweeted
# users who retweeted only once (have an in-degree of 0 and an out-degree of 1).
# users who were retweeted only once (have an in-degree of 1 and an out-degree of 0).

has_tweeted_once_never_retweeted <- in_deg == 1 & out_deg == 0
has_never_tweeted_retweeted_once <- in_deg == 0 & out_deg == 1

# Set vertex colors
vertex_colors <- rep("black", gorder(retweet_graph)) # default class
vertex_colors[has_tweeted_once_never_retweeted] <- "blue"
vertex_colors[has_never_tweeted_retweeted_once] <- "green"

# Plot network
plot(
  retweet_graph, 
  vertex.color = vertex_colors
)

# Distribution of Centrality
retweet_btw <- betweenness(retweet_graph, directed = TRUE)
summary(retweet_btw)

# Calculate proportion of vertices with zero betweenness
mean(retweet_btw == 0)

# Eigen-centrality
retweet_ec <- eigen_centrality(retweet_graph, directed = TRUE)$vector
summary(retweet_ec)

# Proportion of vertices with eigen-centrality close to zero
almost_zero <- 1e-10
mean(retweet_ec < almost_zero)

# Top Ranking Vertices (Nodes of Importance)

# Metrics
betweenness_q99 <- quantile(retweet_btw, .99) # 0.99 quantile of betweenness 
top_btw <- retweet_btw[retweet_btw > betweenness_q99] # top 1% of vertices

eigen_centrality_q99 <- quantile(retweet_ec, .99) # 0.99 quantile of eigen-centrality
top_ec <- retweet_ec[retweet_ec > eigen_centrality_q99] # top 1% of vertices

# View results as data frame
data.frame(
  Rank = seq_along(top_btw), 
  Betweenness = names(sort(top_btw, decreasing = TRUE)), 
  EigenCentrality = names(sort(top_ec, decreasing = TRUE))
)

## Plotting Important Vertices

# Transform betweenness
transformed_btw <- log(retweet_btw + 2)
V(retweet_graph)$size <- transformed_btw # set as size attribute

# Plot the graph
plot(
  retweet_graph, vertex.label = NA, edge.arrow.width = 0.2,
  edge.arrow.size = 0.0, vertex.color = "red"
)

# Subset nodes for betweenness greater than 0.99 quantile
vertices_high_btw <- V(retweet_graph)[retweet_btw > betweenness_q99]
# Induce a subgraph of the vertices with high betweenness
retweet_subgraph <- induced_subgraph(retweet_graph, vertices_high_btw)

plot(retweet_subgraph) # Plot the subgraph

## Building a mentions graph

# Note: Need to avoid names in retweets
# Tweet might also mention multiple names (need to extract all names, draw links)

ment_g <- graph.empty()
ment_g <- ment_g + vertices(all_sn)

for (i in 1:dim(raw_tweets)[1]) {
  ment_name <- mention_ext(raw_tweets$tweet_text[i])
  if (length(ment_name) > 0) {
    # Add the edge(s)
    for (j in ment_name) {
      # Check if vertex exists, if not, add it
      if (!j %in% all_sn) {
        ment_g <- ment_g + vertices(j)
      ment_g <- ment_g + edges(c(raw_tweets$screen_name[i], j))
      }
    }
  }
}

ment_g <- simplify(ment_g)
ment_g <- delete.vertices(ment_g, degree(ment_g) == 0)

# Load mention data
mention_data <- data_frame(
  graph_type = "mention",
  degree_in = degree(mention_graph, mode = "in"),
  degree_out = degree(mention_graph, mode = "out"),
  io_ratio = degree_in / degree_out
)

# Create dataset of retweet ratios from the retweet_graph
retweet_data <- data_frame(
  graph_type = "retweet",
  degree_in = degree(retweet_graph, mode = "in"),
  degree_out = degree(retweet_graph, mode = "out"),
  io_ratio = degree_in / degree_out
)

# Bind the datasets by row
io_data <- bind_rows(mention_data, retweet_data) %>% 
  # Filter for finite, positive io_ratio
  filter(is.finite(io_ratio), io_ratio > 0)

# Plotting io_ratio colored by graph_type
ggplot(io_data, aes(x = io_ratio, color = graph_type)) + 
  # Add a geom_freqpoly layer
  geom_freqpoly() + 
  scale_x_continuous(breaks = 0:10, limits = c(0, 10))

# Summary statistics
io_data %>% 
  group_by(graph_type) %>% 
  summarize(
    mean_io_ratio = mean(io_ratio),
    median_io_ratio = median(io_ratio)
  )

# Assortativity and reciprocity
assortativity.degree(retweet_graph)
assortativity.degree(mention_graph)
reciprocity(retweet_graph)
reciprocity(mention_graph)

# Analysis of clique structure (who is talking to whom)
list_of_clique_vertices <- cliques(mention_graph, min = 3, max = 3) # size 3 cliques
clique_ids <- lapply(list_of_clique_vertices, as_ids) # Loop over cliques, getting IDs

# Loop over cliques, finding cases where revodavid is in the clique
has_revodavid <- sapply(
  clique_ids, 
  function(clique) {
    "revodavid" %in% clique
  }
)

# Subset cliques that have revodavid
cliques_with_revodavid <- clique_ids[has_revodavid]

# Unlist cliques_with_revodavid and get unique values
people_in_cliques_with_revodavid <- unique(unlist(cliques_with_revodavid))

# Induce subgraph of mention_graph with people_in_cliques_with_revodavid 
revodavid_cliques_graph <- induced_subgraph(mention_graph,people_in_cliques_with_revodavid)
plot(revodavid_cliques_graph) # Plot the subgraph

# Finding communities

# Make retweet_graph undirected
retweet_graph_undir <- as.undirected(retweet_graph)

# Apply clustering algorithms for comparison (fast_greedy, infomap, louvain)
communities_fast_greedy <- cluster_fast_greedy(retweet_graph_undir)
communities_infomap <- cluster_infomap(retweet_graph_undir)
communities_louvain <- cluster_louvain(retweet_graph_undir)

# Comparing distances
compare(communities_fast_greedy,communities_infomap)
compare(communities_fast_greedy, communities_louvain)
compare(communities_infomap, communities_louvain)

two_users <- c("bass_analytics", "big_data_flow")

# Subset membership of communities by two_users
membership(communities_fast_greedy)[two_users]
membership(communities_infomap)[two_users]
membership(communities_louvain)[two_users]

## Visualizing communities

# Color vertices by community membership, as a factor
V(retweet_graph)$color <- factor(membership(communities_louvain))

# Find edges that cross between commmunities
is_crossing <- crossing(communities_louvain, retweet_graph)

# Set edge linetype: solid for crossings, dotted otherwise 
E(retweet_graph)$lty <- ifelse(is_crossing, "solid", "dotted")

# Get the sizes of communities_louvain
community_size <- sizes(communities_louvain)

# Find some mid-size communities
in_mid_community <- unlist(communities_louvain[community_size > 50 & community_size < 90])

# Induce a subgraph of retweet_graph using in_mid_community
retweet_subgraph <- induced_subgraph(retweet_graph, in_mid_community)

# Plot those mid-size communities
plot(retweet_subgraph, vertex.label = NA, edge.arrow.width = 0.8, edge.arrow.size = 0.2, 
    coords = layout_with_fr(retweet_subgraph), margin = 0, vertex.size = 3)

## Chicago Bike Sharing Network Analysis -------

library(dplyr)
library(lubridate)

# Load dataset
bike_dat <- read.csv("bike.csv", stringsAsFactors = FALSE)
str(bike_data)
trip_df <- bike_dat %>%
  group_by(from_station_id, to_station_id) %>%
  summarize(weights = n()) # no. trips between stations

# Create graph
trip_g <- graph_from_data_frame(trip_df[,1:2])
# add edge weights
E(trip_g)$weight <- trip_df$weights
gsize(trip_g) # no. edges
gorder(trip_g) # no. nodes

# Explore graph
sg <- induced_subgraph(trip_g, 1:12)
plot(sg, vertex.label = NA, edge.arrow.width = 0.8,
     edge.arrow.size = 0.6,
     margin = 0,
     vertex.size = 6,
     edge.width = log(E(sg)$weight + 2))

## Constructing graphs of different user types

# Filter for rows where usertype is Subscriber
subscribers <- bike_dat %>% 
  filter(usertype == "Subscriber") 

# Number of subscriber trips
n_subscriber_trips <- nrow(subscribers)
subscriber_trip_graph <- subscribers %>% 
  group_by(from_station_id, to_station_id) %>% 
  summarize(
    # Set weights as proportion of total trips
    weights = n() / n_subscriber_trips
  ) %>%
  graph_from_data_frame()

# Filter for rows where usertype is "Customer"
customers <- bike_dat %>% 
  filter(usertype == "Customer")

# Number of customer trips
n_customer_trips <- nrow(customers)

customer_trip_graph <- customers %>% 
  group_by(from_station_id,to_station_id) %>% 
  summarize(
    # Set weights as proportion of total trips 
    weights = n() / n_customer_trips
  ) %>%
  graph_from_data_frame()

# Number of different trips by subscribers
gsize(subscriber_trip_graph)

# Number of different trips by customers
gsize(customer_trip_graph)

## Comparing Graphs of Different User Types

# Subgraph on subscriber trips
twelve_subscriber_trip_graph <- induced_subgraph(subscriber_trip_graph,1:12)
plot(
  twelve_subscriber_trip_graph, 
  main = "Subscribers"
)

# Subgraph on customer trips
twelve_customer_trip_graph <- induced_subgraph(customer_trip_graph,1:12)
plot(
  twelve_customer_trip_graph, 
  main = "Customers"
)

## Graph distance vs geographic distance

farthest_vertices(trip_g_simp) # also gives path length
get_diameter(trip_g_simp) # actual path

# Geographic distance
library(geosphere)
# Get the to stations coordinates
st_to <- bike_dat %>%
  filter(from_station_id == 336) %>%
  sample_n(1) %>%
  select(from_longitude, from_latitude)
# Get the from stations coordinates
st_from <- bike_dat %>%
  filter(from_station_id == 340) %>%
  sample_n(1) %>%
  select(from_longitude, from_latitude)
# Find the geographic distance
farthest_dist <- distm(st_from, st_to, fun = distHaversine)
farthest_dist

# Generalized function for computing geographic distance
bike_dist <- function(station_1, station_2, divy_bike_df) {
  
  st1 <- divy_bike_df %>%
    filter(from_station_id == station_1) %>%
    sample_n(1) %>%
    select(from_longitude, from_latitude)
  
  st2 <- divy_bike_df %>%
    filter(from_station_id == station_2) %>%
    sample_n(1) %>%
    select(from_longitude, from_latitude)
  
  farthest_dist <- distm(st1, st2, fun = distHaversine)
  return(farthest_dist)
}

## Comparing Subscriber vs Non-Subscriber Distances

# Get diameters of subscriber and customer graph
get_diameter(subscriber_trip_graph)
get_diameter(customer_trip_graph)

# Farthest vertices (end vertices, no. nodes in btw)
farthest_vertices(subscriber_trip_graph)
farthest_vertices(customer_trip_graph)

# Calc physical distance between end stations
calc_physical_distance_m(200,298)
calc_physical_distance_m(116,281)

## Most Traveled To and From Stations

trip_deg <- data_frame(
  trip_out = degree(trip_g_simp, mode = "out"), 
  trip_in = degree(trip_g_simp, mode = "in"),
  # Calculate the ratio of out / in
  ratio = trip_out / trip_in
)

trip_deg_filtered <- trip_deg %>%
  filter(trip_in > 10, trip_out > 10) 

# Plot histogram of filtered ratios
hist(trip_deg_filtered$ratio)

## Weighted Degree Distributions (Most Traveled Stations)

trip_strng <- data_frame(
  trip_out = strength(trip_g_simp, mode = "out"), 
  trip_in = strength(trip_g_simp, mode = "in"),
  # Calculate the ratio of out / in
  ratio = trip_out / trip_in
)

trip_strng_filtered <- trip_strng %>%
  filter(trip_in > 10, trip_out > 10) 

# Plot histogram of filtered ratios
hist(trip_strng_filtered$ratio)

## Visualizing central vertices

# Make an ego graph of the least traveled graph
g275 <- make_ego_graph(trip_g_simp, 1, nodes = "275", mode= "out")[[1]]

# Plot ego graph
plot(
  g275, 
  # Weight the edges by weight attribute 
  edge.width = E(g275)$weight,
  layout = latlong
)

## Weighted Measures of Centrality

# Eigencentrality (weighted/unweighted)
ec_weight <- eigen_centrality(trip_g_simp, directed = TRUE)$vector
ec_unweight <- eigen_centrality(trip_g_simp, directed = TRUE, weights = NA)$vector

# Weighted/unweighted closeness
close_weight <- closeness(trip_g_simp)
close_unweight <- closeness(trip_g_simp, weights = NA)

# Get vertex names
vertex_names <- names(V(trip_g_simp))

# Complete the data frame to see the results
data_frame(
  "Weighted Eigen Centrality" = vertex_names[which.min(ec_weight)],
  "Unweighted Eigen Centrality" = vertex_names[which.min(ec_unweight)],
  "Weighted Closeness" = vertex_names[which.min(close_weight)],
  "Unweighted Closeness" = vertex_names[which.min(close_unweight)]
)

## Connectivity

rand_g <- erdos.renyi.game(10, 0.4, "gnp", directed = FALSE)
vertex_connectivity(rand_g)
edge_connectivity(rand_g)

# Minimum no. of cuts
min_cut(rand_g, value.only = FALSE)

# Get parameters to simulate graph
nv <- gorder(trip_g_ud)
ed <- edge_order(trip_g_ud)

graph_vec <- rep(NA, 1000)
# Generate 1000 random graphs, find edge connectivity
for (i in 1:1000){
  w1 <- erdos.renyi.game(nv, ed, "gnp", directed = TRUE)
  graph_vec[i] <- edge_connectivity(w1)
}

# Find actual connectivity
econn <- edge_connectivity(trip_g_ud)
hist(graph_vec, xlim = c(0,140))
abline(v = edge_connectivity(trip_g_ud))

# Calculate the minimum number of cuts
ud_cut <- min_cut(trip_g_ud, value.only = FALSE)
ud_cut

# Make an ego graph from the first partition
ego_partition1 <- make_ego_graph(trip_g_ud, nodes = ud_cut$partition1)[[1]]
plot(ego_partition1)

# Find the number of cuts needed to disconnect nodes "231" and "321"
stMincuts(trip_g_simp, "231", "321")

# Find the number of cuts needed to disconnect nodes "231" and "213"
stMincuts(trip_g_simp, "231", "213")

## Unweighted Clustering Randomizations

# Calculate global transitivity
actual_global_trans <- transitivity(trip_g_simp, type = "global")
actual_global_trans

# Calculate the order
n_nodes <- gorder(trip_g_simp)

# Calculate the edge density
edge_dens <- edge_density(trip_g_simp)

# Run the simulation
simulated_global_trans <- rep(NA, 300)
for(i in 1:300) {
  simulated_graph <- erdos.renyi.game(n_nodes, edge_dens, directed = TRUE)
  # Calculate the global transitivity of the simulated graph
  simulated_global_trans[i] <- transitivity(simulated_graph, type = "global")
}

# Plot a histogram of simulated global transitivity
hist(
  simulated_global_trans, 
  xlim = c(0.35, 0.6), 
  main = "Unweighted clustering randomization"
)

# Add a vertical line at the actual global transitivity
abline(v = actual_global_trans, col = "red")

# Observe global transitivity of simulated graphs much lower than that of original graph

## Weighted Clustering Randomizations

# Find the mean local weighted clustering coeffecient using transitivity()
actual_mean_weighted_trans <- mean(transitivity(trip_g_simp, type = "weighted"))

# Calculate the order
n_nodes <- gorder(trip_g_simp)

# Calculate the edge density
edge_dens <- edge_density(trip_g_simp)

# Get edge weights
edge_weights <- E(trip_g_simp)$weight

# Run the simulation
simulated_mean_weighted_trans <- rep(NA, 100)
for(i in 1:100) {
  simulated_graph <- erdos.renyi.game(n_nodes, edge_dens, directed = TRUE)
  n_simulated_edges <- gsize(simulated_graph)
  # Sample existing weights and add them to the random graph
  E(simulated_graph)$weight <- sample(edge_weights, n_simulated_edges, replace = TRUE)
  # Get the mean transitivity of the simulated graph
  simulated_mean_weighted_trans[i] <- mean(transitivity(simulated_graph, type = "weighted"))
}

# Plot a histogram of simulated mean weighted transitivity
hist(
  simulated_mean_weighted_trans, 
  xlim = c(0.35, 0.7), 
  main = "Mean weighted clustering randomization"
)

# Add a vertical line at the actual mean weighted transitivity
abline(v = actual_mean_weighted_trans, col = "red")

## Other Graph Visualizations -------
library(ggnetwork)
library(GGally)
library(intergraph)

rand_g <- erdos.renyi.game(30, .15, "gnp", directed = FALSE)
rand_g <- simplify(rand_g)

plot(rand_g)

# Basic ggnet2
net <- asNetwork(rand_g)
ggnet2(net)

# Basic ggnetwork
gn <- ggnetwork(rand_g)
g <- ggplot(gn, aes(x = x, y = y, xend = xend, yend = yend)) +
  geom_edges() +
  geom_nodes() +
  theme_blank()

head(gn)

# Add attributes
V(rand_g)$cent <- betweenness(rand_g)
V(rand_g)$comm <- membership(cluster_walktrap(rand_g))

# Default Plot with attributes (tedious method)

plot(rand_g, vertex.label = NA, margin = 0,
     vertex.color = V(rand_g)$comm,
     vertex.size = V(rand_g)$cent / 6)

# Add legend for community membership
legend('topleft', legend = sort(unique(V(rand_g)$comm)),
       col = sort(unique(V(rand_g)$comm)), pch = 19, title = "Community")

# Add cuts and then get quantiles for size legend
cc <- cut(V(rand_g)$cent, 5)
scaled <- quantile(V(rand_g)$cent, seq(0.3, 0.9, length = 5)) / 25

# Add size legend for centrality
legend('bottomleft', legend = levels(cc),
       pt.cex = scaled, pch = 19, title = "Centrality")

# Using ggnet2 for plot with attributes (Simplified)

net <- asNetwork(rand_g)

ggnet2(net,
       node.size = "cent",
       node.color = "comm",
       edge.size = 0.8,
       color.legend = "Community Membership",
       color.palette = "Spectral",
       edge.color = c("color", "gray88"),
       size.cut = TRUE,
       size.legend = "Centrality")

# ggnetwork plot with attributes

gn <- ggnetwork(rand_g)

g <- ggplot(gn, aes(x = x, y = y, xend = xend, yend = yend)) +
  geom_edges(aes(color = as.factor(comm))) +
  geon_nodes(aes(color = as.factor(comm), size = cent)) +
  theme_blank() +
  guides(
    color = guide_legend(title = "Community"),
    size = guide_legend(title = "Centrality")
  )

plot(g)

# Exercises

# Create subgraph of retweet_graph
retweet_samp <- induced_subgraph(retweet_graph, vids = verts)
plot(retweet_samp, vertex.label = NA,
     edge.arrow.size = 0.2, edge.size = 0.5,
     vertex.color = "black", vertex.size = 1)

# Convert to a network object, plot using ggnet2
retweet_net <- asNetwork(retweet_samp)
ggnet2(retweet_net, edge.size = 0.5, node.color = "black", node.size = 1)

# Call ggplot (ggnetwork package)
ggplot(
  ggnetwork(retweet_samp), 
  aes(x = x, y = y, xend = xend, yend = yend)) +
  geom_nodes() +
  geom_edges(arrow = arrow(length = unit(6, "pt"))) +
  theme_blank()

# More ggnet plotting options
retweet_net <- asNetwork(retweet_graph)
ggnet2(
  retweet_net,
  node.size = "cent", 
  node.color = "comm", 
  color.palette = "Spectral", 
  edge.color = c("color","grey90")
  ) +
  guides(size = FALSE)

# More ggnetwork plotting options

ggplot(
  ggnetwork(retweet_graph_smaller, arrow.gap = 0.01), 
  aes(x = x, y = y, xend = xend, yend = yend)
) + 
  geom_edges(
    aes(color = comm),
    arrow = arrow(length = unit(6, "pt"), type = "closed"), 
    curvature = 0.2, color = "black"
  ) + 
  geom_nodes(aes(color = comm, size = cent), size = 4) + 
  theme_blank() +  
  guides(
    color = guide_legend(title = "Community"), 
    size = guide_legend(title = "Centrality")
  )

## Interactive Visualizations -------

library(ggiraph)
library(htmlwidgets)
library(networkD3)

# Create random graph
rand_g <- erdos.renyi.game(30, 0.12, "gnp", directed = FALSE)
rand_g <- simplify(rand_g)

V(rand_g)$cent <- betweenness(rand_g)

# Plot graph with ggplot2 and ggnetwork

gn <- ggnetwork(rand_g)
g <- ggplot(gn, aes(x = x, y = y, xend = xend, yend = yend)) +
  geom_edges(color = "black") +
  geom_nodes(aes(size = cent)) +
  theme_blank() +
  guides(size = guide_legend(title = "Centrality"))

# Create ggiraph object
my_gg <- g +
  geom_point_interactive(aes(tooltip = round(cent, 2)),
                         size = 2)

# Display ggiraph object
ggiraph(code = print(my_gg))

# Further customization using custom css (behavior on mouseover)
my_gg <- g +
  geom_point_interactive(aes(tooltip = round(cent, 2),
                             data_id = round(cent, 2)),
                         size = 2)

hover_css = "cursor:pointer;fill:red;stroke:red;r:5pt"

giraph(code = print(my_gg),
        hover_css = hover_css,
        tooltip_offx = 10,
        tooltip_offy = -10)

# Plotting with networkD3

nd3 <- igraph_to_networkD3(rand_g)
simpleNetwork(nd3$links)

# Add attributes
nd3$nodes$group = V(rand_g)$comm
nd3$nodes$cent = V(rand_g)$cent

# Plot the graph, highlights vertex and connected vertices on mouse over
# Colors each vertex by community membership

forceNetwork(Links = nd3$links,
             Nodes = nd3$nodes,
             Source = 'source',
             Target = 'target',
             NodeID = 'name',
             Group = 'group',
             Nodesize = 'cent',
             legend = T,
             fontsize = 20)

# Exercises:

static_network <- ggplot(
  ggnetwork(trip_g_simp, arrow.gap = 0.01), 
  aes(x = x, y = y, xend = xend, yend = yend)
) + 
  geom_edges() + 
  geom_nodes(aes(size = cent)) + 
  theme_blank() 

interactive_network <- static_network + 
  geom_point_interactive(
    aes(tooltip = cent, data_id = cent)
  ) 

# Print the interactive network
girafe(code = print(interactive_network)) %>%
  girafe_options(
    # Set hover options
    opts_hover(css = "cursor: pointer; fill: red; stroke: red; r: 5pt"),
    # Set tooltip options; give x-offset of 10 
    opts_tooltip(offx = 10)
  )

# Run this to see the static version of the plot
ggplot(ggnetwork(retweet_samp, arrow.gap = 0.01), 
       aes(x = x, y = y, xend = xend, yend = yend)) + 
  geom_edges(color = "black") + 
  geom_nodes(aes(color = as.factor(comm))) + 
  theme_blank()   

# Convert retweet_samp to a networkD3 object
nd3 <-igraph_to_networkD3(retweet_samp, V(retweet_samp)$comm)

# View the data structure
str(nd3)

# Render D3.js network
forceNetwork(
  Links = nd3$links, 
  Nodes = nd3$nodes, 
  Source = "source", 
  Target = "target", 
  NodeID = "name",
  Group = "group",  
  legend = TRUE, 
  fontSize = 20
)

## Alternative Visualizations: Hive plots and biofabric plots ----

library(HiveR)

# Create random graph
rand_g <- erdos.renyi.game(18, 0.3, "gnp", directed = TRUE)
plot(rand_g, vertex.size = 7)

# Convert to dataframe for hive plots and add weights
rand_g_df <- as.data.frame(get.edgelist(rand_g))
rand_g_df$weight <- 1
# Convert to hive object
rand_hive <- edge2HPD(edge_df = rand_g_df) 
# Set axis and radius of each node
rand_hive$nodes$axis <- sort(rep(1:3, 6))
rand_hive$nodes$radius <- as.double(rep(1:6, 3))

rand_hive$nodes # see how nodes are modified

# Hive plot
plotHive(rand_hive, method = 'abs', bkgnd = "white")

# Hive plot extended customization:
# Add weights to each edge
rand_hive$edges$weight <- as.double(
  rpois(length(rand_hive$edges$weight), 5)
)
# Add color based on edge origination
rand_hive$edges$color[rand_hive$edges$id1 %in% 1:6] <- 'red'
rand_hive$edges$color[rand_hive$edges$id1 %in% 7:12] <- 'blue'
rand_hive$edges$color[rand_hive$edges$id1 %in% 13:18] <- 'green'

# (Finalized) Plot
plotHive(rand_hive, method = 'abs', bkgnd = "white")

# Biofabric plots

rand_g <- erdos.renyi.game(10, 0.3, "gnp", directed = TRUE)
rand_g <- simplify(rand_g)
V(rand_g)$name <- LETTERS[1:length(V(rand_g))]

biofbc <- bioFabric(rand_g)
bioFabric_htmlwidget(biofbc)

## Extended Examples:

# Bike Sharing Hive Plot
# Convert trip_df to hive object using edge2HPD()
bike_hive <- edge2HPD(trip_df, axis.cols = rep("black", 3))

# Set edge color
bike_hive$edges$color <- dist_gradient(trip_df$geodist)
# Set node radius based on centrality
bike_hive$nodes$radius <- ifelse(bike_cent > 0, bike_cent, runif(1000, 0, 3))
# Set node axis to station axis
bike_hive$nodes$axis <- dist_stations$axis

# Plot the hive
plotHive(bike_hive, method = "norm", bkgnd = "white")

# Retweet Biofabric Plot
retweet_bf <- bioFabric(retweet_samp)
bioFabric_htmlwidget(retweet_bf) # Create HTMLwidget of retweet_bf

## Plotting Graphs on a map

weighted_trips_by_usertype <- bike_dat %>% 
  group_by(
    from_station_id, to_station_id, 
    from_latitude, from_longitude, 
    to_latitude, to_longitude, 
    usertype
  ) %>% 
  # Weight each journey by number of trips
  summarize(weight = n())

# Create a base map
ggmap(chicago) + 
  # Add a line segment layer
  geom_segment( 
    aes(
      x = from_longitude, y = from_latitude, 
      xend = to_longitude, yend = to_latitude, 
      # Color by user type
      color = usertype, 
      # Set the line width to the weight
      size = weight
    ), 
    data = weighted_trips_by_usertype,
    alpha = 0.5
  )
