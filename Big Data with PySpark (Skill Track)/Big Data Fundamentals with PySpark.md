PySpark is the Python API for Apache Spark, an open-source, distributed computing system used for big data processing and analytics. This notebook will cover the basics of using PySpark, including data loading, data manipulation, and basic operations on Resilient Distributed Datasets (RDDs).

## 1. Introduction to Big Data Analysis with Spark
***
The SparkContext object (`sc`) is the entry point to any spark functionality. When running Spark, you start a new Spark application by creating a SparkContext. You can then use this context to create RDDs, accumulators and broadcast variables, access Spark services and run jobs.

**Inspecting SparkContext:**

```python
sc.version # SparkContext verison
sc.pythonVer # Retrieve Python version
sc.master # Master URL to connect to
```

The above code displays the version of SparkContext. It's always a good practice to check the version of the library as some functionalities may vary between versions.

**Loading Data in PySpark:**

To load data in PySpark, we use the `parallelize` function when the data is already in your local system, and the `textFile` function when the data is in a distributed storage like HDFS (Hadoop Distributed File System).

```python
# Create RDD from a list data
rdd = sc.parallelize([1,2,3,4,5]) 

# Read text file from HDFS
textFile = sc.textFile("/my/directory/*.txt")
```

**Review of lambda functions (`map` and `filter`):**

Python's lambda functions, along with `map` and `filter`, are commonly used with PySpark. `map` applies a function to all items in an input list and `filter` creates a list of elements for which a function returns true.

```python
items = [1,2,3,4]

# The map() function applies a lambda function to all items in the list
list(map(lambda x: x+2, items))

# The filter() function creates a list of items for which the lambda function evaluates to true
list(filter(lambda x: (x%2 != 0), items))
```


## 2. Programming in PySpark RDD's
***
In this section, we will look at how data is processed in PySpark. We'll explore *Resilient Distributed Datasets* (RDDs), how to create them, and the transformations and actions we can perform on them. 

### Resilient Distributed Datasets (RDDs)

In PySpark, data is distributed across multiple nodes in a cluster, allowing for parallel processing of large datasets. The primary abstraction in Spark is an RDD. RDDs are immutable distributed collections of objects. Each dataset in RDD is divided into logical partitions, which may be computed on different nodes of the cluster.

### Partitions

A partition is a logical division of a large distributed data set. Partitioning is an important concept in distributed data processing systems like PySpark because it enables data parallelism and efficient data processing by dividing the data across multiple nodes in a cluster.

**Partitioning in PySpark:**

```python
# Create RDD with defined number of partitions
numRDD = sc.parallelize(range(10), minPartitions = 6)
fileRDD = sc.textFile("README.md", minPartitions = 6)

# Checking the number of partitions
numRDD.getNumPartitions()
```

### RDD Operations (Transformations and Actions)

RDDs support two types of operations: 
* _transformations_, which create a new dataset from an existing one (i.e. create new RDDs)
* _actions_, which return a value to the driver program after running a computation on the dataset (i.e. perform computation on the RDDs) 

For example, `map` is a transformation that passes each dataset element through a function and returns a new RDD representing the results. On the other hand, `reduce` is an action that aggregates all the elements of the RDD using some function and returns the final result to the driver program.

### Transformations

```python
# map() transformation applies function to all elements in RDD
RDD.map(lambda x: x * x)

# filter()
RDD.filter(lambda x: x > 2)

# flatMap() transformation returns multiple values for ea element in RDD
RDD.flatMap(lambda x: x.split(" "))

# union()
inputRDD = sc.textFile("logs.txt")
errorRDD = inputRDD.filter(lambda x: "error"in x.split())
warningsRDD = inputRDD.filter(lambda x: "warnings" in x.split())
combinedRDD = errorRDD.union(warningsRDD)
```

### Basic RDD Actions

```python
# Return a list of all RDD elements
RDD.collect()
# Take first 2 RDD elements
RDD.take(2)
# Take first RDD element
RDD.first()
# Count number of elements in RDD
RDD.count()
```

Note: `collect()` should only be used to retrieve results for small datasets. It should not be used for large datasets.

### PairRDDs in PySpark

PairRDDs, also known as key-value RDDs, are a specific type of Resilient Distributed Dataset (RDD) in PySpark where each element in the RDD is a key-value pair. They provide a way to work with data in a key-value format, enabling operations and transformations that are specific to this data structure.

**Creating pair RDDs:**

```python
my_tuple = [('Sam', 23), ('Mary', 34), ('Peter', 25)]
pairRDD_tuple = sc.parallelize(my_tuple)

my_list = ['Sam 23', 'Mary 34', 'Peter 25']
regularRDD = sc.parallelize(my_list)
pairRDD_RDD = regularRDD.map(lambda s: (s.split(' ')[0], s.split(' ')[1]))
```

### Transformations on pair RDDs

* `reduceByKey()` transformation

```python
# Example:
regularRDD = sc.parallelize([("Messi", 23), ("Ronaldo", 34),
							 ("Neymar", 22), ("Messi", 24)])

## reduceByKey() combines values with same key							 
pairRDD_reducebykey = regularRDD.reduceByKey(lambda x,y : x + y)
pairRDD_reducebykey.collect()
# Output: [('Neymar', 22), ('Ronaldo', 34), ('Messi', 47)]
```

* `sortByKey()` transformation

```python
## sortByKey() returns an RDD sorted by key in ascending or descending order
pairRDD_reducebykey_rev = pairRDD_reducebykey.map(lambda x: (x[1], x[0]))
pairRDD_reducebykey_rev.sortByKey(ascending=False).collect()
# Output: [(47, 'Messi'), (34, 'Ronaldo'), (22, 'Neymar')]
```

* `groupByKey()` transformation

```python
## groupByKey() groups all values with same key in pairRDD
airports = [("US", "JFK"),("UK", "LHR"),("FR", "CDG"),("US", "SFO")]
regularRDD = sc.parallelize(airports)

pairRDD_group = regularRDD.groupByKey().collect()
for cont, air in pairRDD_group:
	print(cont, list(air))
```

* `join()` transformation

```python
## join() transformation joins the two pair RDDs based on key
RDD1 = sc.parallelize([("Messi", 34),("Ronaldo", 32),("Neymar", 24)])
RDD2 = sc.parallelize([("Ronaldo", 80),("Neymar", 120),("Messi", 100)])
RDD1.join(RDD2).collect()
# Output: [('Neymar', (24, 120)), ('Ronaldo', (32, 80)), ('Messi', (34, 100))]
```

### Advanced RDD Actions

* `reduce()` action

```python
# reduce() used for aggregating elements of regular RDD
x = [1,3,4,6]
RDD = sc.parallelize(x)
RDD.reduce(lambda x, y : x + y)
```

* `saveAsTextFile()` action

```python
# Saves RDD into text file inside a directory, each partition as seperate file
RDD.saveAsTextFile("tempFile")
# coalesce() method can be used to saveRDD as a single text file
RDD.coalesce(1).saveAsTextFile("tempFile")
```

* `countByKey()` action

```python
# Only available for type (K,V) i.e. Pair RDDs
# countByKey() counts number of elements for each key
rdd = sc.parallelize([("a", 1), ("b", 1), ("a", 1)])
for key, val in rdd.countByKey().items():
	print(key, val)

# Output: ('a', 2)('b', 1)
```

* `collectAsMap()` action

```python
# collectAsMap() returns the key-value pairs in the RDD as a dictionary
sc.parallelize([(1, 2), (3, 4)]).collectAsMap()
```

### Example: Shakespeare

```python
## Create Base RDD and transform -------

# Create a baseRDD from the file path
baseRDD = sc.textFile(".../Complete_Shakespeare.txt")
# Split the lines of baseRDD into words
splitRDD = baseRDD.flatMap(lambda x: x.split())
# Count the total number of words
print("Total number of words in splitRDD:", splitRDD.count())

## Removing stop words, Reducing Dataset ------

# Convert words in lower case, remove stop words from stop_words curated list
splitRDD_no_stop = splitRDD.filter(lambda x: x.lower() not in stop_words)
# Create tuple (w, 1) where w is for each word
splitRDD_no_stop_words = splitRDD_no_stop.map(lambda w: (w, 1))
# Count number of occurences of each word
resultRDD = splitRDD_no_stop_words.reduceByKey(lambda x, y: x + y)

## Printing Word Frequencies ------

# Display the first 10 words and their frequencies from the input RDD
for word in resultRDD.take(10):
	print(word)
# Swap the keys and values from the input RDD
resultRDD_swap = resultRDD.map(lambda x: (x[1], x[____]))
# Sort the keys in descending order
resultRDD_swap_sort = resultRDD_swap.sortByKey(ascending=False)
# Show the top 10 most frequent words and their frequencies from the sorted RDD
for word in resultRDD_swap_sort.take(10):
	print("{},{}". format(word[1], word[0]))
```

## 3. PySpark SQL & DataFrames
***
In this section, we will discuss PySpark SQL and DataFrames. PySpark SQL is a module in Apache Spark that provides a programming interface for working with structured (e.g. relational database) and semi-structured data (e.g. JSON) using SQL as well as DataFrame API.

### Creating Spark DataFrame from RDD:

We can create a DataFrame in PySpark from an existing RDD or from a CSV/JSON/TXT file.

```python
# Creating a DataFrame from an RDD
iphones_RDD = sc.parallelize([
						("XS", 2018, 5.65, 2.79, 6.24),
						("XR", 2018, 5.94, 2.98, 6.84), 
						("X10", 2017, 5.65, 2.79, 6.13), 
						("8Plus", 2017, 6.23, 3.07, 7.12)
						])
						
names = ['Model', 'Year', 'Height', 'Width', 'Weight']
iphones_df = spark.createDataFrame(iphones_RDD, schema=names)
```

**Creating DataFrame from CSV/JSON/TXT**

```python
df_csv = spark.read.csv("people.csv", header=True, inferSchema=True)
df_json = spark.read.json("people.json", header=True, inferSchema=True)
df_txt = spark.read.txt("people.txt", header=True, inferSchema=True)
```

### DataFrame Operations in PySpark

Once a DataFrame is created, we can perform a variety of operations such as selecting specific columns, filtering rows based on conditions, grouping data, renaming columns, and more.

```python
# Subset columns in df
df.select('colname')
# Print first (n) rows in df
df.show(n)
# Filter rows based on condition
df.filter(...conditional...)
# Aggregation
df.groupby('colname')
# Count rows
df.count()
# Drop duplicate rows
df.dropDuplicates()
# Rename column
df.withColumnRenamed("original", "new_name")
# Print types of columns in df
df.printSchema()
# Print columns of df
df.columns
# Summary statistics
df.describe().show()
```

### Executing SQL Queries

Operations on DataFrames can also be done using SQL queries:

```python
# Create temporary table
df.createOrReplaceTempView("table1")

df2 = spark.sql("SELECT field1, field2 FROM table1")
df2.collect()

# SQL query to extract data
test_df.createOrReplaceTempView("test_table")
query = "SELECT Product_ID FROM test_table"
test_product_df = spark.sql(query)
test_product_df.show(5)

# Summarizing and grouping data using SQL queries
test_df.createOrReplaceTempView("test_table")
query = "SELECT Age, max(Purchase) FROM test_table GROUP BY Age"
spark.sql(query).show(5)

# Filtering columns using SQL queries
test_df.createOrReplaceTempView("test_table")
query = '''SELECT Age, Purchase, Gender FROM test_table WHERE Purchase > 20000 AND Gender == "F"'''
spark.sql(query).show(5)
```

### Data Visualization in PySpark

Ploing graphs using PySpark DataFrames is done using three methods:
- `pyspark_dist_explore` library,
- `toPandas()`,
- `HandySpark` library

```python
# Using Pyspark_dist_explore
test_df = spark.read.csv("test.csv", header=True, inferSchema=True)
test_df_age = test_df.select('Age')
hist(test_df_age, bins=20, color="red")

# Using Pandas for plotting DataFrames
test_df = spark.read.csv("test.csv", header=True, inferSchema=True)
test_df_sample_pandas = test_df.toPandas()
test_df_sample_pandas.hist('Age')

# HandySpark Method
test_df = spark.read.csv('test.csv', header=True, inferSchema=True)
hdf = test_df.toHandy()
hdf.cols["Age"].hist()
```

## 4. Machine Learning with PySpark MLlib
***
In this section, we will explore the Machine Learning library of PySpark, known as MLlib. MLlib provides several algorithms for machine learning, which can be applied directly to RDDs or DataFrames.

Various tools provided by MLlib include:
* ML Algorithms: collaborative filtering, classification and clustering
* Featurization: feature extraction, transformation, dimensionality reduction and selection
* Pipelines: tools for constructing, evaluating, and tuning ML pipelines
***
### PySpark MLlib Algorithms:

MLlib includes several types of machine learning algorithms, including classification, regression, clustering, and collaborative filtering, as well as supporting functionality such as model evaluation and data import.

* **Classication (Binary and Multiclass) and Regression:** Linear SVMs, logistic regression, decision trees, random forests, gradient-boosted trees, naive Bayes, linear least squares, Lasso, ridge regression, isotonic regression 
* **Collaborative Filtering:** Alternating least squares (ALS) 
* **Clustering:** K-means, Gaussian mixture, Bisecting K-means and Streaming K-Means

PySpark MLlib imports:

```python
# Recommendation
from pyspark.mllib.recommendation import ALS
# Classification
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
# Clustering
from pyspark.mllib.clustering import KMeans
```

***
### Collaborative Filtering (CF)

Use-cases in recommendation systems, for finding users that share common interests. 
* **User-User CF:** Finding users that are similar to target user
* **Item-Item CF:** Find and recommend items similar to items with the target user

```python
# Importing Required Class
from pyspark.mllib.recommendation import Rating

# Rating class is a wrapper around tuple (user, product, and rating)
# Here, we create a Rating object with the corresponding user/prod ID and rating
r = Rating(user=1, product=2, rating=5.0)
print((r.user, r.product, r.rating))

# Splitting data using randomSplit() ---------------
data = sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
training, test = data.randomSplit([0.6, 0.4]) # Specify split ratio
print(training.collect()) # 60% training split
print(test.collect()) # 40% test split

# Creating Collaborative Filtering Model ---------------
# Alternating Least Squares (ALS)
r1 = Rating(1, 1, 1.0)
r2 = Rating(1, 2, 2.0)
r3 = Rating(2, 1, 2.0)
ratings = sc.parallelize([r1, r2, r3])
print(ratings.collect())

# Set rank (no. latent factors in model) and number of iterations
model = ALS.train(ratings, rank=10, iterations=10)

# Making Predictions ---------------
# predictAll() - Returns RDD of Rating Objects
unrated_RDD = sc.parallelize([(1, 2), (1, 1)]) # create RDD of user-product pair
predictions = model.predictAll(unrated_RDD) # generate predictions
print(predictions.collect())

# Model Evaluation using MSE ----------------
# Transform actual & predicted ratings into pairs of ((user, product), rating)
rates = ratings.map(lambda x: ((x[0], x[1]), x[2]))
print(rates.collect())

preds = predictions.map(lambda x: ((x[0], x[1]), x[2]))
print(preds.collect())

# Join based on (user,product) pair 
rates_preds = rates.join(preds)
print(rates_preds.collect())

# Calculating MSE
MSE = rates_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error (MSE):", MSE)
```

***
### Classification

A note on vectors: PySpark MLlib contains specific data types `Vectors` and `LabelledPoint`.

```python
# Vectors
# Dense Vector: store all their entries in an array of foating point numbers
denseVec = Vectors.dense([1.0, 2.0, 3.0])
# Sparse Vector: store only the nonzero values and their indices
parseVec = Vectors.sparse(4, {1: 1.0, 3: 5.5})

# LabeledPoint: Wrapper for input features and predicted value
# For binary classication, label is either 0 (negative) or 1 (positive)
positive = LabeledPoint(1.0, [1.0, 0.0, 3.0])
negative = LabeledPoint(0.0, [2.0, 1.0, 1.0])
print(positive)
print(negative)

# HashingTF() used to map feature value to indices in the feature vector
from pyspark.mllib.feature import HashingTF
sentence = "hello hello world"
words = sentence.split()
tf = HashingTF(10000)
tf.transform(words) 
# obtain sparse vector with feature no. (i.e. word) and occurances of each word
```

Here, we introduce feature hashing, a technique to convert categorical variables to numerical values. The `HashingTF` class is a transformer which takes sets of terms and converts those sets into fixed-length feature vectors. In this case, it transforms the words in the sentence into numerical feature vectors. The number 10000 defines the number of features in the resulting feature vectors.

**Logistic Regression:**

```python
# Import required classes
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LabeledPoint

# Create LabeledPoints data and parallelize into RDD
data = [LabeledPoint(0.0, [0.0, 1.0]),
		LabeledPoint(1.0, [1.0, 0.0])]
RDD = sc.parallelize(data)

# Train Logistic Regression model with LBFGS optimizer
lrm = LogisticRegressionWithLBFGS.train(RDD)

# Compute predictions on new data points
prediction1 = lrm.predict([1.0, 0.0])
prediction2 = lrm.predict([0.0, 1.0])
print(prediction1)
print(prediction2)
```

***
### Clustering

K-Means Clustering Example:

```python
# Load and preprocess data ------------------------
RDD = sc.textFile("WineData.csv") \
        .map(lambda x: x.split(",")) \
        .map(lambda x: [float(x[0]), float(x[1])])

# RDD head
RDD.take(5)

# Train K-means clustering model (2 clusters) ------------------------
from pyspark.mllib.clustering import KMeans
model = KMeans.train(RDD, k=2, maxIterations=10)

# Print coordinates of cluster centers
model.clusterCenters

# Evaluating K-means model --------------------
from math import sqrt

def error(point):
    center = model.centers[model.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

# Compute total within-cluster sum of square error
WSSSE = RDD.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))

# Visualizing clusters --------------------
# First, convert RDD to DataFrame for visualization
wine_data_df = spark.createDataFrame(RDD, schema=["col1", "col2"])
wine_data_df_pandas = wine_data_df.toPandas()

# Convert the cluster centers to a pandas DataFrame for visualization
cluster_centers_pandas = pd.DataFrame(model.clusterCenters, 
									  columns=["col1", "col2"])
cluster_centers_pandas.head()

# Create a scatter plot of the data, with different colors for each cluster
# The data points are plotted in blue
plt.scatter(wine_data_df_pandas["col1"], wine_data_df_pandas["col2"])

# The cluster centers are plotted in red with an 'x' marker
plt.scatter(cluster_centers_pandas["col1"], 
			cluster_centers_pandas["col2"], 
			color="red", marker="x")
```