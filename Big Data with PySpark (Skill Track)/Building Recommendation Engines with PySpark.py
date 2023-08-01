## Building Recommendation Engines with PySpark


## 1. Introduction to Recommendation Systems -------------------------

# View TJ_ratings
TJ_ratings.show()
# Generate recommendations for users
get_ALS_recs(["Jane","Taylor"]) 

# Group the data by "Genre"
markus_ratings.groupBy("Genre").sum().show()

'''
Two Types of Recommendation Engines:

1. Content-Based Filtering - Inference based on features of items
2. Collaborative Filtering - Inference based on user similarity

Ratings can be explicit or implicit (e.g. view counts => confidence score)

// Alternating Least Squares (Non-Negative Matrix Factorization) //

Decompose Ratings Matrix (Users x Items) 
				= Product of User-Item(latent) Factor Matrices

where No. latent features = Matrix Rank

Error metric: RMSE
'''

## 2. Alternating Least Squares (ALS) -------------------------

## Data Preparation for Spark ALS

# Import monotonically_increasing_id and show R
from pyspark.sql.functions import monotonically_increasing_id
R.show()

'''
# Observe dataframe in conventional or "wide" format
+----------------+-----+----+----------+--------+
|            User|Shrek|Coco|Swing Kids|Sneakers|
+----------------+-----+----+----------+--------+
|    James Alking|    3|   4|         4|       3|
|Elvira Marroquin|    4|   5|      null|       2|
|      Jack Bauer| null|   2|         2|       5|
|     Julia James|    5|null|         2|       2|
+----------------+-----+----+----------+--------+
'''

# Use the to_long() function to convert the dataframe to the "long" format.
ratings = to_long(R)
ratings.show()

'''
    +----------------+----------+------+
    |            User|     Movie|Rating|
    +----------------+----------+------+
    |    James Alking|     Shrek|     3|
    |    James Alking|      Coco|     4|
    |    James Alking|Swing Kids|     4|
    |    James Alking|  Sneakers|     3|
    |Elvira Marroquin|     Shrek|     4|
    |Elvira Marroquin|      Coco|     5|
    |Elvira Marroquin|  Sneakers|     2|
    |      Jack Bauer|      Coco|     2|
    |      Jack Bauer|Swing Kids|     2|
    |      Jack Bauer|  Sneakers|     5|
    |     Julia James|     Shrek|     5|
    |     Julia James|Swing Kids|     2|
    |     Julia James|  Sneakers|     2|
    +----------------+----------+------+
'''

# Get unique users and repartition to 1 partition using coalesce
users = ratings.select("User").distinct().coalesce(1)

# Create a new column of unique integers called "userId" in dataframe
# .persist() method to ensure new integer IDs persist
users = users.withColumn("userId", monotonically_increasing_id()).persist()
users.show()

## Movie IDs

# Extract the distinct movie id's
movies = ratings.select("Movie").distinct() 

# Repartition the data to have only one partition.
movies = movies.coalesce(1) 

# Create a new column of movieId integers. 
movies = movies.withColumn("movieId", monotonically_increasing_id()).persist() 

# Join the ratings, users and movies dataframes
movie_ratings = ratings.join(users, "User", "left").join(movies, "Movie", "left")
movie_ratings.show()

## Building out ALS Model

# Split the ratings dataframe into training and test data
(training_data, test_data) = ratings.randomSplit([.8, .2], seed=42)

# Set the ALS hyperparameters
from pyspark.ml.recommendation import ALS
als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating",
 rank = 10, maxIter = 15, regParam = .1,
 coldStartStrategy = "drop",
 nonnegative = True,
 implicitPrefs = False)

# Fit the mdoel to the training_data
model = als.fit(training_data)

# Generate predictions on the test_data
test_predictions = model.transform(test_data)
test_predictions.show()

# Import RegressionEvaluator
from pyspark.ml.evaluation import RegressionEvaluator

# Complete the evaluator code
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

# Extract the 3 parameters
print(evaluator.getMetricName())
print(evaluator.getLabelCol())
print(evaluator.getPredictionCol())

## 3. Recommending Movies (MovieLens Dataset) -------------------------

# Inspecting MovieLens Dataset
print(ratings.columns)
print(ratings.show(5))

'''
<script.py> output:
['userId', 'movieId', 'rating', 'timestamp']
+------+-------+------+----------+
|userId|movieId|rating| timestamp|
+------+-------+------+----------+
|     1|     31|   2.5|1260759144|
|     1|   1029|   3.0|1260759179|
|     1|   1061|   3.0|1260759182|
|     1|   1129|   2.0|1260759185|
|     1|   1172|   4.0|1260759205|
+------+-------+------+----------+
only showing top 5 rows
'''

## Computing Sparsity of Ratings matrix

# Count the total number of ratings in the dataset
numerator = ratings.select("rating").count()

# Count the number of distinct userIds and distinct movieIds
num_users = ratings.select("userId").distinct().count()
num_movies = ratings.select("movieId").distinct().count()

# Set the denominator equal to the number of users multiplied by the number of movies
denominator = num_users * num_movies

# Divide the numerator by the denominator
sparsity = (1.0 - (numerator*1.0)/denominator)*100
print("The ratings dataframe is ", "%.2f" % sparsity + "% empty.")

'''
<script.py> output:
    The ratings dataframe is  98.36% empty.
'''

# Import the requisite packages
from pyspark.sql.functions import col

# View the ratings dataset
ratings.show()

# Filter to show only userIds less than 100
ratings.filter(col("userId") < 100).show()

# Group data by userId, count ratings
ratings.groupBy("userId").count().show()

## MovieLens Summary Statistics

# Min num ratings for movies
print("Movie with the fewest ratings: ")
ratings.groupBy("movieId").count().select(min("count")).show()

'''
    Movie with the fewest ratings: 
    +----------+
    |min(count)|
    +----------+
    |         1|
    +----------+
'''

# Avg num ratings per movie
print("Avg num ratings per movie: ")
ratings.groupBy("movieId").count().select(avg("count")).show()

'''
    Avg num ratings per movie: 
    +------------------+
    |        avg(count)|
    +------------------+
    |11.030664019413193|
    +------------------+
'''
# Min num ratings for user
print("User with the fewest ratings: ")
ratings.groupBy("userId").count().select(min("count")).show()

'''
    User with the fewest ratings: 
    +----------+
    |min(count)|
    +----------+
    |        20|
    +----------+
'''
# Avg num ratings per users
print("Avg num ratings per user: ")
ratings.groupBy("userId").count().select(avg("count")).show()

''' 
    Avg num ratings per user: 
    +------------------+
    |        avg(count)|
    +------------------+
    |149.03725782414307|
    +------------------+
'''

## Viewing Schema

# Check datatypes of the ratings dataset
ratings.printSchema()

# Note: Spark ALS requires movieIds and userId's to be provided as integers!
# Tell Spark to convert the columns to the proper data types
ratings = ratings.select(ratings.userId.cast("integer"), ratings.movieId.cast("integer"), ratings.rating.cast("double"))

# Call .printSchema() again to confirm the columns are now in the correct format
ratings.printSchema()

## Create train/test splits, build ALS model

# Import the required functions
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Create test and train set
(train, test) = ratings.randomSplit([0.8, 0.2], seed = 1234)

# Create ALS model
als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating",
 nonnegative = True,
 implicitPrefs = False)

# Add hyperparameters and their respective values to param_grid
param_grid = ParamGridBuilder() \
            .addGrid(als.rank, [10,50,100,150]) \
            .addGrid(als.maxIter, [5,50,100,200]) \
            .addGrid(als.regParam, [.01,.05,.1,.15]) \
            .build()
           
# Define evaluator as RMSE and print length of evaluator
evaluator = RegressionEvaluator(metricName="rmse",
							 labelCol="rating",
							 predictionCol="prediction")

print ("Num models to be tested: ", len(param_grid))

'''
<script.py> output:
    Num models to be tested:  64
'''

# Build cross validation using CrossValidator
cv = CrossValidator(estimator=als,
                estimatorParamMaps=param_grid,
                evaluator=evaluator,
                numFolds=5)

#Fit cross validator to the 'train' dataset
model = cv.fit(train)

#Extract best model from the cv model above
best_model = model.bestModel

# Print best_model
print(type(best_model))

# Extracting Optimal ALS model parameters
print("**Best Model**")
print("  Rank:", best_model.getRank())
print("  MaxIter:", best_model.getMaxIter())
print("  RegParam:", best_model.getRegParam())

# Generate model test predictions
test_predictions = best_model.transform(test)

# View the predictions 
test_predictions.show()

# Calculate and print the RMSE of test_predictions
RMSE = evaluator.evaluate(test_predictions)
print(RMSE)

# RMSE Interpretation: 
# On average, our model deviates in value by 0.633 (above or below) from the actual ratings

# Generate top n recommendations for all users
recommendForAllUsers(n) # n is an integer

# Inspect results of recommendations
ALS_recommendations.show()

# Cleaning up recommendation output
ALS_recommendations.registerTempTable("ALS_recs_temp")

clean_recs = spark.sql("""
    SELECT userID, rec.movieId, rec.rating as prediction
    FROM ALS_recs_temp
    LATERAL VIEW explode(recommendations) exploded_table as rec
""")

clean_recs.join(movie_info, ['movieId'], "left").show()

## Check: Do recommendations make sense?

# Look at user 60's ratings
print("User 60's Ratings:")
original_ratings.filter(col("userId") == 60).sort("rating", ascending = False).show()

# Look at the movies recommended to user 60
print("User 60s Recommendations:")
recommendations.filter(col("userId") == 60).show()

# Look at user 63's ratings
print("User 63's Ratings:")
original_ratings.filter(col("userId") == 63).sort("rating", ascending = False).show()

# Look at the movies recommended to user 63
print("User 63's Recommendations:")
recommendations.filter(col("userId") == 63).show()

## 4. Further Inferences (Implicit Ratings Models) -------------------------

## MSD Summary Statistics

# Look at the data
msd.show()

# Count the number of distinct userIds and songIds
user_count = msd.select("userId").distinct().count()
song_count = msd.select("songId").distinct().count()
print("Number of users: ", user_count)
print("Number of songs: ", song_count)

'''
<script.py> output:
    +------+------+---------+
    |userId|songId|num_plays|
    +------+------+---------+
    |   148|   148|        0|
    |   243|   496|        0|
    |    31|   471|        0|
    |   137|   463|        0|
    |   251|   623|        0|
    +------+------+---------+
    only showing top 5 rows
    
    Number of users:  321
    Number of songs:  729
'''

# Min num implicit ratings for a song
print("Minimum implicit ratings for a song: ")
msd.filter(col("num_plays") > 0).groupBy("songId").count().select(min("count")).show()

# Avg num implicit ratings per songs
print("Average implicit ratings per song: ")
msd.filter(col("num_plays") > 0).groupBy("songId").count().select(avg("count")).show()

# Min num implicit ratings from a user
print("Minimum implicit ratings from a user: ")
msd.filter(col("num_plays") > 0).groupBy("userId").count().select(min("count")).show()

# Avg num implicit ratings for users
print("Average implicit ratings per user: ")
msd.filter(col("num_plays") > 0).groupBy("userId").count().select(avg("count")).show()

'''
<script.py> output:
    Minimum implicit ratings for a song: 
    +----------+
    |min(count)|
    +----------+
    |         3|
    +----------+
    
    Average implicit ratings per song: 
    +------------------+
    |        avg(count)|
    +------------------+
    |35.251063829787235|
    +------------------+
    
    Minimum implicit ratings from a user: 
    +----------+
    |min(count)|
    +----------+
    |        21|
    +----------+
    
    Average implicit ratings per user: 
    +-----------------+
    |       avg(count)|
    +-----------------+
    |77.42056074766356|
    +-----------------+
'''

# View the ratings data
Z.show()

'''
+------+---------+-------------+
|userId|productId|num_purchases|
+------+---------+-------------+
|  2112|      777|            1|
|     7|       44|           23|
|  1132|      227|            9|
|   686|     1981|            2|
|    42|     2390|            5|
+------+---------+-------------+
only showing top 5 rows
'''
# Extract distinct userIds and productIds
users = Z.select("userId").distinct()
products = Z.select("productId").distinct()

# Cross join users and products (i.e. Joining each user to each song)
cj = users.crossJoin(products)

# Applying left join back with original ratings data
# Set null values to zero (every user has value for every song)
Z_expanded = cj.join(Z, ["userId", "productId"], "left").fillna(0)
Z_expanded.show()

'''
Note: In case of implicit ratings models, cannot use RMSE as a measure

Consider Rank Ordering Error Metric (ROEM)
	> Checks if songs with higher number of plays correspond to higher prediction

Values close to .5 (no better than random), want values close to 0
'''

# ALS hyperparameters
ranks = [10, 20, 30, 40]
maxIters = [10, 20, 30, 40]
regParams = [.05, .1, .15]
alphas =  [20, 40, 60, 80]

# Empty capture group for models
model_list = []

# Building Implicit Models:
# For loop will automatically create and store ALS models
for r in ranks:
    for mi in maxIters:
        for rp in regParams:
            for a in alphas:
                model_list.append(ALS(userCol= "userId", itemCol= "songId", ratingCol= "num_plays",
                 rank = r, maxIter = mi, regParam = rp, alpha = a,
                 coldStartStrategy="drop",
                 nonnegative = True, 
                 implicitPrefs = True))

# Print the model list, and the length of model_list
print (model_list, "Length of model_list: ", len(model_list))

# Validate
len(model_list) == (len(ranks)*len(maxIters)*len(regParams)*len(alphas))

## Cross-Validation

# Split the data into training and test sets
(training, test) = msd.randomSplit([0.8, 0.2])

#Building 5 folds within the training set.
train1, train2, train3, train4, train5 = training.randomSplit([0.2, 0.2, 0.2, 0.2, 0.2], seed = 1)
fold1 = train2.union(train3).union(train4).union(train5)
fold2 = train3.union(train4).union(train5).union(train1)
fold3 = train4.union(train5).union(train1).union(train2)
fold4 = train5.union(train1).union(train2).union(train3)
fold5 = train1.union(train2).union(train3).union(train4)

foldlist = [(fold1, train1), (fold2, train2), (fold3, train3), (fold4, train4), (fold5, train5)]

# Empty list to fill with ROEMs from each model
ROEMS = []

# Loops through all models and all folds
for model in model_list:
    for ft_pair in foldlist:

        # Fits model to fold within training data
        fitted_model = model.fit(ft_pair[0])

        # Generates predictions using fitted_model on respective CV test data
        predictions = fitted_model.transform(ft_pair[1])

        # Generates and prints a ROEM metric CV test data
        r = ROEM(predictions)
        print ("ROEM: ", r)

    # Fits model to all of training data and generates preds for test data
    v_fitted_model = model.fit(training)
    v_predictions = v_fitted_model.transform(test)
    v_ROEM = ROEM(v_predictions)

    # Adds validation ROEM to ROEM list
    ROEMS.append(v_ROEM)
    print ("Validation ROEM: ", v_ROEM)

## Running Cross-Validated Implicit ALS model

import numpy

# Find the index of the smallest ROEM
i = numpy.argmin(ROEMS)
print("Index of smallest ROEM:", i)

# Find ith element of ROEMS
print("Smallest ROEM: ", ROEMS[i])

'''
<script.py> output:
    Index of smallest ROEM: 38
    Smallest ROEM:  0.01980198019801982
'''

# Extract the best_model
best_model = model_list[38]

# Extract Hyperparameters
print ("Rank: ", best_model.getRank())
print ("MaxIter: ", best_model.getMaxIter())
print ("RegParam: ", best_model.getRegParam())
print ("Alpha: ", best_model.getAlpha())

'''
<script.py> output:
    Rank:  10
    MaxIter:  40
    RegParam:  0.05
    Alpha:  60.0
'''

## Binary Implicit Ratings

# Import the col function
from pyspark.sql.functions import col

# Look at the test predictions
binary_test_predictions.show()

# Evaluate ROEM on test predictions
ROEM(binary_test_predictions)

# Look at user 42's test predictions
binary_test_predictions.filter(col("userId") == 42).show()


# View user 26's original ratings
print ("User 26 Original Ratings:")
original_ratings.filter(col("userId") == 26).show()
# View user 26's recommendations
print ("User 26 Recommendations:")
binary_recs.filter(col("userId") == 26).show()
# View user 99's original ratings
print ("User 99 Original Ratings:")
original_ratings.filter(col("userId") == 99).show()
# View user 99's recommendations
print ("User 99 Recommendations:")
binary_recs.filter(col("userId") == 99).show()
