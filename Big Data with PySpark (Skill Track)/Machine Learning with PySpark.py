## Machine Learning with PySpark

## 1. Introduction  -------------------------

# Remote Cluster using Spark URL
spark://<IP address | DNS name>:<port>

# Creating a SparkSession
from pyspark.sql import SparkSession

# Create local cluster
spark = SparkSession.builder \
				.master('local[*]') \
				.appName('first_spark_application') \
				.getOrCreate()

# local - only 1 core;
# local[4] - 4 cores; or
# local[*] - all available cores

# Check version
print(spark.version)

# Terminate cluster (good practice)
spark.stop()

## Loading flights data

# Read data from CSV file
flights = spark.read.csv('flights.csv',
                         sep=',',
                         header=True,
                         inferSchema=True,
                         nullValue='NA')

# Get number of records
print("The data contain %d records." % flights.count())

# View the first five records
flights.show(5)

# Check column data types
print(flights.dtypes)

## Loading SMS spam data

from pyspark.sql.types import StructType, StructField, IntegerType, StringType

# Specify column names and types
schema = StructType([
    StructField("id", IntegerType()),
    StructField("text", StringType()),
    StructField("label", IntegerType())
])

# Load data from a delimited file
sms = spark.read.csv('sms.csv', sep=';', header=False, schema=schema)

# Print schema of DataFrame
sms.printSchema()

## 2. Classification -------------------------

## Data Preperation: Removing Columns and Rows

# Remove the 'flight' column
flights_drop_column = flights.drop('flight')

# Number of records with missing 'delay' values
flights_drop_column.filter('delay IS NULL').count()

# Remove records with missing 'delay' values
flights_valid_delay = flights_drop_column.filter('delay IS NOT NULL')

# Remove records with missing values in any column
# Get the number of remaining rows
flights_none_missing = flights_valid_delay.dropna()
print(flights_none_missing.count())

## Data Preparation: Column Manipulation

# Import the required function
from pyspark.sql.functions import round

# Convert 'mile' to 'km' and drop 'mile' column (1 mile is equivalent to 1.60934 km)
flights_km = flights.withColumn('km', round(flights.mile * 1.60934, 0)) \
                    .drop('mile')

# Create 'label' column indicating whether flight delayed (1) or not (0)
flights_km = flights_km.withColumn('label', (flights.delay >= 15).cast('integer'))

# Check first five records
flights_km.show(5)

## Data Preparation: Categorical Columns

from pyspark.ml.feature import StringIndexer

# Create an indexer
indexer = StringIndexer(inputCol='carrier', outputCol='carrier_idx')

# Indexer identifies categories in the data
indexer_model = indexer.fit(flights)

# Indexer creates a new column with numeric index values
flights_indexed = indexer_model.transform(flights)

# Repeat the process for the other categorical feature
flights_indexed = StringIndexer(inputCol='org', outputCol='org_idx') \
			.fit(flights_indexed) \
			.transform(flights_indexed)

flights_indexed.show(5)

## Data Preparation: Assembling Columns

# Import the necessary class
from pyspark.ml.feature import VectorAssembler

# Create an assembler object
assembler = VectorAssembler(inputCols=[
    'mon', 'dom', 'dow', 'carrier_idx', 'org_idx', 'km', 'depart', 'duration'
], outputCol='features')

# Consolidate predictor columns
flights_assembled = assembler.transform(flights)

# Check the resulting column
flights_assembled.select('features', 'delay').show(5, truncate=False)

## Train-Test Split

# Split into training and testing sets in a 80:20 ratio
flights_train, flights_test = flights.randomSplit([.8,.2], seed = 43)

# Check that training set has around 80% of records
training_ratio = flights_train.count() / flights.count()
print(training_ratio)

'''
<script.py> output:
    0.8035334184759472
'''

## Build Decision Tree

# Import the Decision Tree Classifier class
from pyspark.ml.classification import DecisionTreeClassifier

# Create a classifier object and fit to the training data
tree = DecisionTreeClassifier()
tree_model = tree.fit(flights_train)

# Create predictions for the testing data and take a look at the predictions
prediction = tree_model.transform(flights_test)
prediction.select('label', 'prediction', 'probability').show(5, False)

## Evaluate Decision Tree (Confusion Matrix)

# Create a confusion matrix
prediction.groupBy('label', 'prediction').count().show()

# Calculate the elements of the confusion matrix
TN = prediction.filter('prediction = 0 AND label = prediction').count()
TP = prediction.filter('prediction = 1 AND label = prediction').count()
FN = prediction.filter('prediction = 0 AND label != prediction').count()
FP = prediction.filter('prediction = 1 AND label != prediction').count()

# Accuracy measures the proportion of correct predictions
accuracy = (TN + TP) / (TN + TP + FN + FP)
print(accuracy)

'''
# Confusion Matrix
<script.py> output:
    +-----+----------+-----+
    |label|prediction|count|
    +-----+----------+-----+
    |    1|       0.0|  196| - False Negative (FN)
    |    0|       0.0|  310| - True Negative (TN)
    |    1|       1.0|  255| - True Positive (TP)
    |    0|       1.0|  162| - False Positive (FP)
    +-----+----------+-----+
    
    0.6121343445287107
'''
## Build Logistic Regression Model

# Import the logistic regression class
from pyspark.ml.classification import LogisticRegression

# Create a classifier object and train on training data
logistic = LogisticRegression().fit(flights_train)

# Create predictions for the testing data and show confusion matrix
prediction = logistic.transform(flights_test)
prediction.groupBy('label', 'prediction').count().show()

'''
<script.py> output:
    +-----+----------+-----+
    |label|prediction|count|
    +-----+----------+-----+
    |    1|       0.0|  184|
    |    0|       0.0|  266|
    |    1|       1.0|  281|
    |    0|       1.0|  192|
    +-----+----------+-----+
'''

## Evaluate Logistic Regression Model

from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

# Calculate precision and recall
precision = TP / (TP + FP)
recall = TP / (TP + FN)
print('precision = {:.2f}\nrecall    = {:.2f}'.format(precision, recall))

# Find weighted precision (Weighted Metrics)
# i.e. proportion of predictions (positive & negative) that are coorect

multi_evaluator = MulticlassClassificationEvaluator()
weighted_precision = multi_evaluator.evaluate(prediction, {multi_evaluator.metricName: "weightedPrecision"})

# Note other metrics: `weightedRecall`, `accuracy`, `f1`

# Recall:
# ROC = "Receiver Operating Characteristic"
# TP versus FP

# Find AUC (Threshold) - "Area Under Curve" (ideally AUC = 1)
binary_evaluator = BinaryClassificationEvaluator()
auc = binary_evaluator.evaluate(prediction, {binary_evaluator.metricName: 'areaUnderROC'})

## Punctuation, numbers and tokens

'''
Preprocessing data by:
1. Removing punctuations and numbers
2. Applying tokenization (spliting into individual words)
'''

# Import the necessary functions
from pyspark.sql.functions import regexp_replace
from pyspark.ml.feature import Tokenizer

# Remove punctuation (REGEX provided) and numbers
wrangled = sms.withColumn('text', regexp_replace(sms.text, '[_():;,.!?\\-]', ' '))
wrangled = wrangled.withColumn('text', regexp_replace(wrangled.text, '[0-9]', ' '))

# Merge multiple spaces
wrangled = wrangled.withColumn('text', regexp_replace(wrangled.text, ' +', ' '))

# Split the text into words
wrangled = Tokenizer(inputCol='text', outputCol="words").transform(wrangled)
wrangled.show(4, truncate=False)

## Stop words and Hashing

'''
Data preprocessing cont.
3. Removing stop words
4. Applying hashing trick
5. Coverting to TF-IDF representation
'''

from pyspark.ml.feature import StopWordsRemover, HashingTF, IDF

# Remove stop words.
wrangled = StopWordsRemover(inputCol="words", outputCol="terms")\
      .transform(sms)

# Apply the hashing trick
wrangled = HashingTF(inputCol="terms", outputCol ="hash", numFeatures=1024)\
      .transform(wrangled)

# Convert hashed symbols to TF-IDF
tf_idf = IDF(inputCol="hash", outputCol="features")\
      .fit(wrangled).transform(wrangled)
      
tf_idf.select('terms', 'features').show(4, truncate=False)

'''
A quick reminder about these concepts:

The hashing trick provides a fast and space-efficient way to map 
a very large (possibly infinite) set of items 
(in this case, all words contained in the SMS messages)
 onto a smaller, finite number of values.

The TF-IDF matrix reflects how important a word is to each document. 
It takes into account both the frequency of the word within each document
but also the frequency of the word across all 
'''

## Training a spam classifier

# Split the data into training and testing sets
sms_train, sms_test = sms.randomSplit([.8,.2], seed=13)

# Fit a Logistic Regression model to the training data
logistic = LogisticRegression(regParam=0.2).fit(sms_train)

# Make predictions on the testing data
prediction = logistic.transform(sms_test)

# Create a confusion matrix, comparing predictions to known labels
prediction.groupBy('label', 'prediction').count().show()

## 3. Regression -------------------------

## Encoding Flight Origin
# One-hot encoding categorical column variable for regression modelling

# Import the one hot encoder class
from pyspark.ml.feature import OneHotEncoder
# Create an instance of the one hot encoder
onehot = OneHotEncoder(inputCols=['org_idx'], outputCols=['org_dummy'])
# Apply the one hot encoder to the flights data
onehot = onehot.fit(flights)
flights_onehot = onehot.transform(flights)
# Check the results
flights_onehot.select('org', 'org_idx', 'org_dummy').distinct().sort('org_idx').show()

## Flight Duration Model: Just distance

from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Create a regression object and train on training data
regression = LinearRegression(labelCol = "duration").fit(flights_train)

# Create predictions for the testing data and take a look at the predictions
predictions = regression.transform(flights_test)
predictions.select('duration', 'prediction').show(5, False)

# Calculate the RMSE
RegressionEvaluator(labelCol="duration").evaluate(predictions)

## Interpreting Coefficients

# Intercept (average minutes on ground)
inter = regression.intercept
print(inter)
# Coefficients
coefs = regression.coefficients
print(coefs)

# Average minutes per km
minutes_per_km = regression.coefficients[0]
print(minutes_per_km)
# Average speed in km per hour
avg_speed = 60 / minutes_per_km
print(avg_speed)

## Flight Duration Model: Adding origin airport

'''
Subset from the flights DataFrame:

+------+-------+-------------+----------------------+
|km    |org_idx|org_dummy    |features              |
+------+-------+-------------+----------------------+
|3465.0|2.0    |(7,[2],[1.0])|(8,[0,3],[3465.0,1.0])|
|509.0 |0.0    |(7,[0],[1.0])|(8,[0,1],[509.0,1.0]) |
|542.0 |1.0    |(7,[1],[1.0])|(8,[0,2],[542.0,1.0]) |
|1989.0|0.0    |(7,[0],[1.0])|(8,[0,1],[1989.0,1.0])|
|415.0 |0.0    |(7,[0],[1.0])|(8,[0,1],[415.0,1.0]) |
+------+-------+-------------+----------------------+
only showing top 5 rows
'''

from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Create a regression object and train on training data
regression = LinearRegression(labelCol = "duration").fit(flights_train)

# Create predictions for the testing data
predictions = regression.transform(flights_test)

# Calculate the RMSE
RegressionEvaluator(labelCol="duration").evaluate(predictions)

## Bucketing Departure Time

from pyspark.ml.feature import Bucketizer, OneHotEncoder

# Create buckets at 3 hour intervals through the day
buckets = Bucketizer(splits=[0,3,6,9,12,15,18,21,24], inputCol="depart", outputCol="depart_bucket")

# Bucket the departure times
bucketed = buckets.transform(flights)
bucketed.select("depart","depart_bucket").show(5)

'''
<script.py> output:
    +------+-------------+
    |depart|depart_bucket|
    +------+-------------+
    |  9.48|          3.0|
    | 16.33|          5.0|
    |  6.17|          2.0|
    | 10.33|          3.0|
    |  8.92|          2.0|
    +------+-------------+
    only showing top 5 rows
'''

# Create a one-hot encoder
onehot = OneHotEncoder(inputCol="depart_bucket", outputCol="depart_dummy")

# One-hot encode the bucketed departure times
flights_onehot = onehot.fit(bucketed).transform(bucketed)
flights_onehot.select("depart","depart_bucket","depart_dummy").show(5)

'''
    +------+-------------+-------------+
    |depart|depart_bucket| depart_dummy|
    +------+-------------+-------------+
    |  9.48|          3.0|(7,[3],[1.0])|
    | 16.33|          5.0|(7,[5],[1.0])|
    |  6.17|          2.0|(7,[2],[1.0])|
    | 10.33|          3.0|(7,[3],[1.0])|
    |  8.92|          2.0|(7,[2],[1.0])|
    +------+-------------+-------------+
    only showing top 5 rows
'''

## Flight Duration Model: Adding departure time

# Find the RMSE on testing data
from pyspark.ml.evaluation import RegressionEvaluator
rmse = RegressionEvaluator(labelCol="duration").evaluate(predictions)
print("The test RMSE is", rmse)

# Average minutes on ground at OGG for flights departing between 21:00 and 24:00
avg_eve_ogg = regression.intercept
print(avg_eve_ogg)

# Average minutes on ground at OGG for flights departing between 03:00 and 06:00
avg_night_ogg = regression.intercept + regression.coefficients[8]
print(avg_night_ogg)

# Average minutes on ground at JFK for flights departing between 03:00 and 06:00
avg_night_jfk = regression.intercept + regression.coefficients[3] + regression.coefficients[9]
print(avg_night_jfk)

## Incorporating more features

'''
Subset from the flights DataFrame:

+--------------------------------------------+--------+
|features                                    |duration|
+--------------------------------------------+--------+
|(32,[0,3,11],[3465.0,1.0,1.0])              |351     |
|(32,[0,1,13,17,21],[509.0,1.0,1.0,1.0,1.0]) |82      |
|(32,[0,2,10,19,23],[542.0,1.0,1.0,1.0,1.0]) |82      |
|(32,[0,1,11,16,30],[1989.0,1.0,1.0,1.0,1.0])|195     |
|(32,[0,1,10,20,25],[415.0,1.0,1.0,1.0,1.0]) |65      |
+--------------------------------------------+--------+
only showing top 5 rows
'''
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Fit linear regression model to training data
regression = LinearRegression(labelCol = "duration").fit(flights_train)

# Make predictions on testing data
predictions = regression.transform(flights_test)

# Calculate the RMSE on testing data
rmse = RegressionEvaluator(labelCol = "duration").evaluate(predictions)
print("The test RMSE is", rmse)

# Look at the model coefficients
coeffs = regression.coefficients
print(coeffs)

# <script.py> output:
# The test RMSE is 11.365305439060927

# Observe large number of non-zero coefficients (impact on interpretability)

## Regularization

from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Fit Lasso model (λ = 1, α = 1) to training data
regression = LinearRegression(labelCol="duration", regParam = 1, elasticNetParam=1).fit(flights_train)

# Calculate the RMSE on testing data
rmse = RegressionEvaluator(labelCol="duration").evaluate(regression.transform(flights_test))
print("The test RMSE is", rmse)

# Look at the model coefficients
coeffs = regression.coefficients
print(coeffs)

# Number of zero coefficients
zero_coeff = sum([beta == 0 for beta in regression.coefficients])
print("Number of coefficients equal to 0:", zero_coeff)

# <script.py> output:
# The test RMSE is 11.847247620737766
# Number of coefficients equal to 0: 22

## 4. Ensembles & Pipelines -------------------------

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import LinearRegression

## Flight Duration Model: Pipeline Stages

# Convert categorical strings to index values
indexer = StringIndexer(inputCol='org',outputCol='org_idx')
# One-hot encode index values
onehot = OneHotEncoder(
    inputCols=['org_idx','dow'],
    outputCols=['org_dummy','dow_dummy']
)
# Assemble predictors into a single column
assembler = VectorAssembler(inputCols=['km','org_dummy','dow_dummy'], outputCol='features')
# A linear regression object
regression = LinearRegression(labelCol='duration')

## Flight Duration Model: Pipeline Model

# Import class for creating a pipeline
from pyspark.ml import Pipeline
# Construct a pipeline
pipeline = Pipeline(stages=[indexer,onehot,assembler,regression])
# Train the pipeline on the training data
pipeline = pipeline.fit(flights_train)
# Make predictions on the testing data
predictions = pipeline.transform(flights_test)

## SMS Spam Pipeline

from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF

# Break text into tokens at non-word characters
tokenizer = Tokenizer(inputCol='text', outputCol='words')

# Remove stop words
remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol='terms')

# Apply the hashing trick and transform to TF-IDF
hasher = HashingTF(inputCol=remover.getOutputCol(), outputCol="hash")
idf = IDF(inputCol=hasher.getOutputCol(), outputCol="features")

# Create a logistic regression object and add everything to a pipeline
logistic = LogisticRegression()
pipeline = Pipeline(stages=[tokenizer, remover, hasher, idf, logistic])

## Cross Validation (Simple Flight Duration Model)

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Create an empty parameter grid
params = ParamGridBuilder().build()

# Create objects for building and evaluating a regression model
regression = LinearRegression(labelCol='duration')
evaluator = RegressionEvaluator(labelCol='duration')

# Create a cross validator
cv = CrossValidator(estimator=regression, estimatorParamMaps=params, evaluator=evaluator, numFolds = 5)
# Train and test model on multiple folds of the training data
cv = cv.fit(flights_train)

# NOTE: Since cross-valdiation builds multiple models, 
# the fit() method can take a little while to complete.

## Cross Validation (Flight Duration Model Pipeline)

# Create an indexer for the org field
indexer = StringIndexer(inputCol='org',outputCol='org_idx')
# Create an one-hot encoder for the indexed org field
onehot = OneHotEncoder(inputCols=['org_idx'],outputCols=['org_dummy'])
# Assemble the km and one-hot encoded fields
assembler = VectorAssembler(inputCols=['km','org_dummy'], outputCol='features')

# Create a pipeline and cross-validator.
pipeline = Pipeline(stages=[indexer, onehot, assembler, regression])
cv = CrossValidator(estimator=pipeline,
          estimatorParamMaps=params,
          evaluator=evaluator)

## Optimizing Flights Linear Regression (Grid Search)
# Using cross-validation to select optimal set of model hyperparameters

# Create parameter grid
params = ParamGridBuilder()
# Add grids for two parameters
params = params.addGrid(regression.regParam, [0.01,0.1,1.0,10.0]) \
               .addGrid(regression.elasticNetParam, [0.0,0.5,1.0])
# Build the parameter grid
params = params.build()
print('Number of models to be tested: ', len(params))
# Number of models to be tested:  12

# Create cross-validator
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=params, evaluator=evaluator, numFolds = 5)

## Dissecting Best Flight Duration Model

# Get the best model from cross validation
best_model = cv.bestModel

# Look at the stages in the best model
print(best_model.stages)

# Get the parameters for the LinearRegression object in the best model
best_model.stages[3].extractParamMap()

# Generate predictions on testing data using the best model then calculate RMSE
predictions = best_model.transform(flights_test)
print("RMSE =", evaluator.evaluate(predictions))

## SMS spam optimized

# Create parameter grid
params = ParamGridBuilder()

# Add grid for hashing trick parameters
params = params.addGrid(hasher.numFeatures, [1024, 4096, 16384]) \
               .addGrid(hasher.binary, [True,False])

# Add grid for logistic regression parameters
params = params.addGrid(logistic.regParam, [0.01, 0.1, 1.0, 10.0]) \
               .addGrid(logistic.elasticNetParam, [0.0,0.5,1.0])

# Build parameter grid
params = params.build()

## Ensemble Models:
## Delayed Flights (Gradient-Boosted Trees)

'''
Recall Gradient-Boosted Trees (Theory):

Iterative boosting algorithm
1. Build a Decision Tree and add to ensemble.
2. Predict label for each training instance using ensemble.
3. Compare predictions with known labels.
4. Emphasize training instances with incorrect predictions.
5. Return to 1.

Model improves on each iteration
'''

# Import the classes required
from pyspark.ml.classification import DecisionTreeClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Create model objects and train on training data
tree = DecisionTreeClassifier().fit(flights_train)
gbt = GBTClassifier().fit(flights_train)

# Compare AUC on testing data
evaluator = BinaryClassificationEvaluator()
print(evaluator.evaluate(tree.transform(flights_test)))
print(evaluator.evaluate(gbt.transform(flights_test)))

# Find the number of trees and the relative importance of features
print(gbt.trees)
print(gbt.featureImportances)

## Delayed Flights (Random Forest - ensemble of Decision Trees)
# Recall: Each tree trained on random subset of data
# random subset of features used for splitting at each node

# Create a random forest classifier
forest = RandomForestClassifier()

# Create a parameter grid
params = ParamGridBuilder() \
            .addGrid(forest.featureSubsetStrategy, ['all', 'onethird', 'sqrt', 'log2']) \
            .addGrid(forest.maxDepth, [2, 5, 10]) \
            .build()

# Create a binary classification evaluator
evaluator = BinaryClassificationEvaluator()

# Create a cross-validator
cv = CrossValidator(estimator = forest, estimatorParamMaps = params, evaluator = evaluator, numFolds = 5)

## Evaluating Random Forest

# Average AUC for each parameter combination in grid
print(cv.avgMetrics)

# Average AUC for the best model
print(max(cv.avgMetrics))

# What's the optimal parameter value for maxDepth?
print(cv.bestModel.explainParam('maxDepth'))
# What's the optimal parameter value for featureSubsetStrategy?
print(cv.bestModel.explainParam('featureSubsetStrategy'))

# AUC for best model on testing data
print(evaluator.evaluate(cv.transform(flights_test)))

