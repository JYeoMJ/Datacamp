# Inspect edgeList
head(edgeList)

# Construct grapj
network <- graph_from_data_frame(edgeList, directed = FALSE)
network

# Inspect the customers dataframe
head(customers)

# Count the number of churners and non-churners
table(customers$churn)

# Add a node attribute called churn
V(network)$churn <- customers$churn

# Visualize the network
plot(network, vertex.label = NA, edge.label = NA,
    edge.color = 'black', vertex.size = 2)

# Add a node attribute called color
V(network)$color <- V(network)$churn

# Change the color of churners to red and non-churners to white
V(network)$color <- gsub("1", "red", V(network)$color) 
V(network)$color <- gsub("0", "white", V(network)$color)

# Plot the network
plot(network, vertex.label = NA, edge.label = NA,
    edge.color = "black", vertex.size = 2)

# Create a subgraph with only churners
churnerNetwork <- induced_subgraph(network, 
                      v = V(network)[which(V(network)$churn == 1)])

# Plot the churner network 
plot(churnerNetwork, vertex.label = NA, vertex.size = 2)

# Compute the churn probabilities
churnProb <- ChurnNeighbors / (ChurnNeighbors + NonChurnNeighbors)

# Find who is most likely to churn
mostLikelyChurners <- which(churnProb == max(churnProb))

# Extract the IDs of the most likely churners
customers$id[mostLikelyChurners]

# Find churn probability of the 44th customer
churnProb[44]

# Update the churn probabilties and the non-churn probabilities
churnProb_updated <- as.vector((AdjacencyMatrix %*% churnProb) / neighbors)

# Find updated churn probability of the 44th customer
churnProb_updated[44] # note increase in churn probability

## Collective Inferencing

# Load the pROC package and data
library(pROC)
load("Nex132.RData")

# Compute the AUC
auc(customers$churn, as.vector(churnProb))

# Write a for loop to update the probabilities
for (i in 1:10){
  churnProb <- as.vector((AdjacencyMatrix %*% churnProb) / neighbors)
}

# Compute the AUC again
auc(customers$churn, as.vector(churnProb))

# Extracting types of edges
# Add the column edgeList$FromLabel
edgeList$FromLabel <- customers[match(edgeList$from, customers$id), 2]

# Add the column edgeList$ToLabel
edgeList$ToLabel <- customers[match(edgeList$to, customers$id), 2]

# Add the column edgeList$edgeType
edgeList$edgeType <- edgeList$FromLabel + edgeList$ToLabel

# Count the number of each type of edge
table(edgeList$edgeType)

# Count churn edges
ChurnEdges <- sum(edgeList$edgeType == 2)

# Count non-churn edges
NonChurnEdges <- sum(edgeList$edgeType == 0)

# Count mixed edges
MixedEdges <- sum(edgeList$edgeType == 1)

# Count all edges
edges <- ChurnEdges + NonChurnEdges + MixedEdges

#Print the number of edges
edges

# Count the number of churn nodes
ChurnNodes <- sum(customers$churn == 1)

# Count the number of non-churn nodes
NonChurnNodes <- sum(customers$churn == 0)

# Count the total number of nodes
nodes <- ChurnNodes + NonChurnNodes

# Compute the network connectance
connectance <- 2 * edges / nodes / (nodes - 1)

# Print the value
connectance

# Compute the expected churn dyadicity
ExpectedDyadChurn <- ChurnNodes * (ChurnNodes-1) * connectance / 2

# Compute the churn dyadicity
DyadChurn <- ChurnEdges / ExpectedDyadChurn

# Inspect the value
DyadChurn

# Compute the expected heterophilicity
ExpectedHet <- NonChurnNodes * ChurnNodes * connectance

# Compute the heterophilicity
Het <- MixedEdges / ExpectedHet

# Inspect the heterophilicity
Het

# ... review this part... I zoned out

# Extract network degree
V(network)$degree <- degree(network, normalized=TRUE)

# Extract 2.order network degree
degree2 <- neighborhood.size(network, 2)

# Normalize 2.order network degree
V(network)$degree2 <- degree2 / (length(V(network)) - 1)

# Extract number of triangles
V(network)$triangles <- count_triangles(network)

# Extract the betweenness
V(network)$betweenness <- betweenness(network, normalized=TRUE)

# Extract the closeness
V(network)$closeness <- closeness(network, normalized=TRUE)

# Extract the eigenvector centrality
V(network)$eigenCentrality <- eigen_centrality(network, scale = TRUE)$vector

# Extract the local transitivity
V(network)$transitivity <- transitivity(network, type='local', isolates='zero')

# Compute the network's transitivity
transitivity(network)

# Extract the adjacency matrix
AdjacencyMatrix <- as_adjacency_matrix(network)

# Compute the second order matrix
SecondOrderMatrix_adj <- AdjacencyMatrix %*% AdjacencyMatrix

# Adjust the second order matrix
SecondOrderMatrix <- ((SecondOrderMatrix_adj) > 0) + 0
diag(SecondOrderMatrix) <- 0

# Inspect the second order matrix
SecondOrderMatrix[1:10, 1:10]

# Compute the number of churn neighbors
V(network)$ChurnNeighbors <- as.vector(AdjacencyMatrix %*% V(network)$Churn)

# Compute the number of non-churn neighbors
V(network)$NonChurnNeighbors <- as.vector(AdjacencyMatrix %*% (1 - V(network)$Churn))

# Compute the relational neighbor probability
V(network)$RelationalNeighbor <- as.vector(V(network)$ChurnNeighbors / 
                                             (V(network)$ChurnNeighbors + V(network)$NonChurnNeighbors))

# Compute the number of churners in the second order neighborhood
V(network)$ChurnNeighbors2 <- as.vector(SecondOrderMatrix %*% V(network)$Churn)

# Compute the number of non-churners in the second order neighborhood
V(network)$NonChurnNeighbors2 <- as.vector(SecondOrderMatrix %*% (1 - V(network)$Churn))

# Compute the relational neighbor probability in the second order neighborhood
V(network)$RelationalNeighbor2 <- as.vector(V(network)$ChurnNeighbors2 / 
                              (V(network)$ChurnNeighbors2 + V(network)$NonChurnNeighbors2))

# Extract the average degree of neighboring nodes
V(network)$averageDegree <- 
  as.vector(AdjacencyMatrix %*% V(network)$degree) / degree

# Extract the average number of triangles of neighboring nodes
V(network)$averageTriangles <- 
  as.vector(AdjacencyMatrix %*% V(network)$triangles) / degree

# Extract the average transitivity of neighboring nodes    
V(network)$averageTransitivity<-
  as.vector(AdjacencyMatrix %*% V(network)$transitivity) / degree

# Extract the average betweenness of neighboring nodes    
V(network)$averageBetweenness <- 
  as.vector(AdjacencyMatrix %*% V(network)$betweenness) / degree

# Compute one iteration of PageRank 
iter1 <- page.rank(network, algo = 'power', options = list(niter = 1))$vector

# Compute two iterations of PageRank 
iter2 <- page.rank(network, algo = 'power', options = list(niter = 2))$vector

# Inspect the change between one and two iterations
sum(abs(iter1 - iter2))

# Inspect the change between nine and ten iterations
sum(abs(iter9 - iter10))

# Create an empty vector
value <- c()

# Write a loop to compute PageRank 
for(i in 1:15){
  value <- cbind(value, page.rank(network, algo = 'power',options = list(niter = i))$vector)
}

# Compute the differences 
difference <- colSums(abs(value[,1:14] - value[,2:15]))

# Plot the differences
plot(1:14, difference)

# Look at the distribution of standard PageRank scores
boxplots(damping = 0.85)

# Inspect the distribution of personalized PageRank scores
boxplots(damping = 0.85, personalized = TRUE)

# Look at the standard PageRank with damping factor 0.2
boxplots(damping = 0.2)

# Inspect the personalized PageRank scores with a damping factor 0.99
boxplots(damping = 0.99, personalized = TRUE)

# Compute the default PageRank score
V(network)$pr_0.85 <- page.rank(network)$vector

# Compute the PageRank score with damping 0.2
V(network)$pr_0.20 <- page.rank(network, damping = 0.2)$vector

# Compute the personalized PageRank score
V(network)$perspr_0.85 <- page.rank(network, personalized = V(network)$Churn)$vector

# Compute the personalized PageRank score with damping 0.99
V(network)$perspr_0.99 <- page.rank(network, damping = 0.99, personalized = V(network)$Churn)$vector

## Putting it all together

# Extract the dataset
studentnetworkdata_full <- (network, what = ___)

# Inspect the dataset
head(studentnetworkdata_full)

# Remove customers who already churned
studentnetworkdata_filtered <- studentnetworkdata_full[-which(studentnetworkdata_full$___ == 1), ]


# Remove useless columns
studentnetworkdata <- studentnetworkdata_filtered[, -c(1, 2)]

# Missing Values

apply(studentnetworkdata, 2, function(x) sum(is.na(x)))

# Note: columnn RelationalNeighborSecond has 6 missing values

# Inspect the feature
summary(studentnetworkdata$RelationalNeighborSecond)

# Find the indices of the missing values
toReplace <- which(is.na(studentnetworkdata$RelationalNeighborSecond))

# Replace the missing values with 0
studentnetworkdata$RelationalNeighborSecond[toReplace] <- 0

# Inspect the feature again
summary(studentnetworkdata$RelationalNeighborSecond)

# Correlated variables
# Remove the Future column from studentnetworkdata 
no_future <- studentnetworkdata[,-1]

# Load the corrplot package
library(corrplot)

# Generate the correlation matrix
M <- cor(no_future)

# Plot the correlations
corrplot(M, method = "circle")

# Print the column names
colnames(studentnetworkdata)

# Create toRemove
toRemove <- c(10, 13, 19, 22)

# Remove the columns
studentnetworkdata_no_corrs <- studentnetworkdata[, -toRemove]

# Set the seed
set.seed(7)

# Creat the index vector
index_train <- sample(1:nrow(studentnetworkdata), 2 / 3 * nrow(studentnetworkdata))

# Make the training set
training_set <- studentnetworkdata[index_train,]

# Make the test set
test_set <- studentnetworkdata[-index_train,]

# Make firstModel
# Build a logistic regression model using the network features in training_set.
Call the model firstModel.
firstModel <- glm(Future ~ degree + degree2 +
                    triangles + betweenness 
                  + closeness + transitivity, family = 'binomial', data = training_set)

# Build a logistic regression model using the link based features in the dataset.
# Call it secondModel.
secondModel <- glm(Future ~ ChurnNeighbors + RelationalNeighbor + ChurnNeighborsSecond + RelationalNeighborSecond + averageDegree + averageTriangles + averageTransitivity + averageBetweenness, 
                   family = 'binomial', data = training_set)

# Build a logistic regression model using all the features.
# Call it thirdModel
thirdModel <- glm(Future~., family = 'binomial', data = training_set)

# Random Forest Model
# Load package
library(randomForest)

# Set seed
set.seed(863)

# Build model
rfModel <- randomForest(as.factor(Future)~. ,data=training_set)

# Plot variable importance
varImpPlot(rfModel)

# perspr_0.85 and eigenCentrality are the most important features

# Load the package
library(pROC)

# Predict with the first model
firstPredictions <- predict(firstModel, newdata = test_set, type = "response")

# Predict with the second model
secondPredictions <- predict(secondModel, newdata = test_set, type = "response")

# Predict with the third model
thirdPredictions <- predict(thirdModel, newdata = test_set, type = "response")

# Predict with the rfModel
rfPredictions<- predict(rfModel, newdata = test_set, type= "prob")

# Measure AUC
auc(test_set$Future, firstPredictions)
auc(test_set$Future, secondPredictions)
auc(test_set$Future, thirdPredictions)
auc(test_set$Future, rfPredictions[,2])

# Measure top decile lift
TopDecileLift(test_set$Future, firstPredictions)
TopDecileLift(test_set$Future, secondPredictions)
TopDecileLift(test_set$Future, thirdPredictions)
TopDecileLift(test_set$Future, rfPredictions[,2])


