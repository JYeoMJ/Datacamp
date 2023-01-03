# [Datacamp] Monte Carlo Simulations in Python

import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st

## 1. Introduction to Monte Carlo Simulations

### Deterministic Simulation
def deterministic_inflation(year, yearly_increase_percent):
    inflation_rate = 8.6
    inflation_rate = inflation_rate*((100+yearly_increase_percent)/100)**(year-2022)
    return(inflation_rate)

# Print the deterministic simulation results
print(deterministic_inflation(2050, 2))

### Monte Carlo Inflation

def monte_carlo_inflation(year,seed):
	random.seed(seed)
	inflation_rate = 8.6
	yearly_increase = random.randint(1,3)

	for i in range(year - 2022):
		inflation_rate = inflation_rate*((100+yearly_increase)/100)

	return(inflation_rate)

# Simulate the inflation rate for the year 2050 with a seed of 1234
print(monte_carlo_inflation(2050,1234))
# Simulate the inflation rate for the year 2050 with a seed of 34228
print(monte_carlo_inflation(2050,34228))

### Applying Law of Large Numbers

# Calculate the average of 1,000 simulation results with a seed between 0 and 20000

rates_1 = []
for i in range(1000):
    seed = random.randint(0, 20000)
    rates_1.append(monte_carlo_inflation(2050, seed))
print(np.mean(rates_1))

# Calculate the average of 10,000 simulation results with a seed between 0 and 20000

rates_2 = []
for i in range(10000):
    seed = random.randint(0, 20000)
    rates_2.append(monte_carlo_inflation(2050, seed))
print(np.mean(rates_2))

### Sampling with Replacement (Bootstrap

nba_weights = [96.7, 101.1, 97.9, 98.1, 98.1, 
               100.3, 101.0, 98.0, 97.4]

simu_weights = []

# Sample nine values from nba_weights with replacement 1000 times
for i in range(1000):
	bootstrap_sample = random.choices(nba_weights, k = 9)
	simu_weights.append(np.mean(bootstrap_sample))

# Calculate the mean and 95% confidence interval of the mean for your results
mean_weight = np.mean(simu_weights)
upper = np.quantile(simu_weights, 0.975)
lower = np.quantile(simu_weights, 0.025)
print(mean_weight, lower, upper)

# Plot the distribution of the simulated weights
sns.displot(simu_weights)

# Plot vertical lines for the 95% confidence intervals and mean
plt.axvline(lower, color="red")
plt.axvline(upper, color="red")
plt.axvline(mean_weight, color="green")
plt.show()

### Permutation

nba_weights = [96.7, 101.1, 97.9, 98.1, 98.1, 100.3, 101.0, 98.0, 97.4, 100.5, 100.3, 100.2, 100.6]
us_adult_weights = [75.1, 100.1, 95.2, 81.0, 72.0, 63.5, 80.0, 97.1, 94.3, 80.3, 93.5, 85.8, 95.1]

# Want to calculate the 95% confidence interval of the mean difference between NBA players and US adult males

all_weights = nba_weights + us_adult_weights
simu_diff = []

for i in range(1000):
	# Perform permutation on all_weights
	perm_sample = np.random.permutation(all_weights)
	# Assign the permuted samples to perm_nba and perm_adult
	perm_nba, perm_adult = perm_sample[0:13], perm_sample[13:]
	perm_diff = np.mean(perm_nba) - np.mean(perm_adult)
	simu_diff.append(perm_diff)

mean_diff = np.mean(nba_weights) - np.mean(us_adult_weights)
upper = np.quantile(simu_diff, 0.975)
lower = np.quantile(simu_diff, 0.025)
print(mean_diff, lower, upper)

# Note: Observe that mean difference lies outside the 95% CI, 
# suggesting that the mean weight of NBA players is significantly different from average
# of US adult males

### Paired Dice Simulation

# bag1 and bag2 each containing 3 biased dice
# dice in bags are paired: if second die in bag1 is picked, likewise for bag2

bag1 = [ [1, 2, 3, 6, 6, 6], [1, 2, 3, 4, 4, 6], [1, 2, 3, 3, 3, 5] ]
bag2 = [ [2, 2, 3, 4, 5, 6], [3, 3, 3, 4, 4, 5], [1, 1, 2, 4, 5, 5] ]

# Define success if points on both die add up to 8, else failure

# Want to calculate the probabilities of success for each unique combination
# of points on Dice 1 and Dice 2

def roll_paired_biased_dice(n, seed=1231):
    random.seed(seed)
    results={}

    for i in range(n):
        bag_index = random.randint(0, 2)
        # Obtain the dice indices
        dice_index1 = random.randint(0,5)
        dice_index2 = random.randint(0,5)
        # Sample a pair of dice from bag1 and bag2
        point1 = bag1[bag_index][dice_index1]
        point2 = bag2[bag_index][dice_index2]

        key = "%s_%s" % (point1,point2)
        if point1 + point2 == 8: 
            if key not in results:
                results[key] = 1
            else:
                results[key] += 1

    return(pd.DataFrame.from_dict({'dice1_dice2':results.keys(),
		'probability_of_success':np.array(list(results.values()))*100.0/n}))

# Run the simulation 10,000 times and assign the result to df_results
df_results = roll_paired_biased_dice(10000, seed = 1231)
sns.barplot(x="dice1_dice2", y="probability_of_success", data=df_results)
plt.show()

# -------------------------------------------------- #

## 2. Foundations for Monte Carlo

### Sampling from a Discrete Uniform Distribution

# Define low and high for use in rvs sampling below
low = 1
high = 7
# Sample 1,000 times from the discrete uniform distribution
samples = st.randint.rvs(low,high,size = 1000)

samples_dict = {'nums':samples}
sns.histplot(x='nums', data=samples_dict, bins=6, binwidth=0.3)
plt.show()

### Sampling from Geometric Distribution

p = 0.2 # biased coin
samples = st.geom.rvs(p, size = 10000)

samples_dict = {"nums":samples}
sns.histplot(x="nums", data=samples_dict)  
plt.show()

### Sampling from Normal Distribution

random.seed(1222)

# Sample 1,000 times from the normal distribution where the mean is 177
# standard deviation is 8
heights_177_8 = st.norm.rvs(loc = 177, scale = 8, size = 1000)
print(np.mean(heights_177_8))
upper = np.quantile(heights_177_8, 0.975)
lower = np.quantile(heights_177_8, 0.025)
print([lower, upper])

# Sample 1,000 times from the normal distribution where the mean is 185
# standard deviation is 8
heights_185_8 = st.norm.rvs(loc = 185, scale = 8, size = 1000)
print(np.mean(heights_185_8))
upper = np.quantile(heights_185_8, 0.975)
lower = np.quantile(heights_185_8, 0.025)
print([lower, upper])

### Two independent normal distributions

# Sample from normal distribution
income1 = st.norm.rvs(loc = 500, scale = 50, size = 1000)
income2 = st.norm.rvs(loc = 1000, scale = 200, size = 1000)

# Define total_income
total_income = income1 + income2
upper = np.quantile(total_income, 0.975)
lower = np.quantile(total_income, 0.025)
print([lower, upper])

### Multinomial Sampling

# Initialize probabilities (must sum to 1)
p_sunny = 300/365
p_cloudy = 35/365
p_rainy = 30/365
num_of_days_in_a_year = 365
number_of_years = 50

# Simulate results
days = st.multinomial.rvs(num_of_days_in_a_year,
    [p_sunny, p_cloudy, p_rainy], size = number_of_years)

# Complete the definition of df_days
df_days = pd.DataFrame({"sunny": days[:, 0],
     "cloudy": days[:, 1],
     "rainy":  days[:, 2]})
sns.pairplot(df_days) # uses scatterplot() for each pairing and hisplot() along diagonals
plt.show()

### Exploring a Multivariate Normal Distribution

sns.pairplot(house_price_size) 
plt.show() # Note: Observe a strong positive covariance based on scatterplots

# Covariance matrix of data
print(house_price_size.cov())

mean_value = [20, 500]
sample_size_value = 5000
cov_mat = np.array([[19, 950], [950, 50000]])

# Simulate the results using sampling
simulated_results = st.multivariate_normal.rvs(mean = mean_value, 
	size = sample_size_value, cov = cov_mat)

simulated_house_price_size = pd.DataFrame({"price":simulated_results[:,0],
                         				   "size":simulated_results[:,1]})

# Visualize the results 
sns.pairplot(simulated_house_price_size)
plt.show()

# -------------------------------------------------- #

## 3. Principled Monte Carlo (Diabetes dataset)

### Exploratory Data Analysis

# Create a pairplot of tc, ldl, and hdl
sns.pairplot(dia[["tc","ldl","hdl"]])
plt.show()
# Calculate correlation coefficients
print(dia[["tc","ldl","hdl"]].corr())

### Choosing Probability Distributions (fit using MLE)
distributions = [st.uniform, st.norm, st.expon]
mles = []

for distribution in distributions:
	# Fit the distribution and obtain the MLE value
	pars = distribution.fit(dia["age"])
	mle = distribution.nnlf(pars, dia["age"]) # negative loglikelihood function
	mles.append(mle)

print(mles) # lower nnlf MLE value is preferred (better fit)

### Another Example: Clerk Data

sns.histplot(clerk_data)
plt.show()

distributions = [st.uniform, st. norm, st.expon]
mles = []
for distribution in distributions:
	pars = distribution.fit(clerk_data)
	mle = distribution.nnlf(pars, clerk_data)
	mles.append(mle)

print(mles)

### Monte Carlo Simulation with Multivariate Normal Dist.

# Parameters for Multivar Normal Dist.
cov_dia = dia[["age", "bmi", "bp", "tc", "ldl", "hdl", "tch", "ltg", "glu"]].cov()
mean_dia = dia[["age", "bmi", "bp", "tc", "ldl", "hdl", "tch", "ltg", "glu"]].mean()

simulation_results = st.multivariate_normal.rvs(mean = mean_dia, # simulating 10000 runs
													size = 10000, cov = cov_dia)

df_results = pd.DataFrame(simulation_results,columns=["age", "bmi", "bp", "tc",
													 "ldl", "hdl", "tch", "ltg", "glu"])

# Calculate bmi and tc means for the historical and simulated results
print(dia[["bmi","tc"]].mean())
print(df_results[["bmi","tc"]].mean())
      
# Calculate bmi and tc covariances for the historical and simulated results
print(dia[["bmi","tc"]].cov())
print(df_results[["bmi","tc"]].cov())

### Going from Covariance to Correlation Matrix

# Calculate the covariance matrix of bmi and tc
cov_dia2 = dia[["bmi","tc"]].cov()

# Calculate the correlation matrix of bmi and tc
corr_dia2 = dia[["bmi","tc"]].corr()
std_dia2 = dia[["bmi","tc"]].std()

print(f'Covariance of bmi and tc from covariance matrix :{cov_dia2.iloc[0,1]}')
print(f'Covariance of bmi and tc from correlation matrix :{corr_dia2.iloc[0,1] * std_dia2[0] * std_dia2[1]}')

### Summary Statistics

cov_dia = dia[["age", "bmi", "bp", "tc", "ldl", "hdl", "tch", "ltg", "glu"]].cov()
mean_dia = dia[["age", "bmi", "bp", "tc", "ldl", "hdl", "tch", "ltg", "glu"]].mean()

simulation_results = st.multivariate_normal.rvs(mean=mean_dia, size=10000, cov=cov_dia)

df_results = pd.DataFrame(simulation_results, columns=["age", "bmi", "bp", "tc", "ldl", "hdl", "tch", "ltg", "glu"])

# Calculate the 0.1st quantile of the tc variable
print(quantile(df_results[["tc"]],0.001))

### Evaluating BMI Outcomes

simulation_results = st.multivariate_normal.rvs(mean=mean_dia, size=20000, cov=cov_dia)
df_results = pd.DataFrame(simulation_results,columns=["age", "bmi", "bp", "tc",
													 "ldl", "hdl", "tch", "ltg", "glu"])
predicted_y = regr_model.predict(df_results)
df_y = pd.DataFrame(predicted_y, columns=["predicted_y"])
df_summary = pd.concat([df_results,df_y], axis=1)

# Calculate the 10th and 90th quantile of bmi in the simulated results
bmi_q10 = np.quantile(df_summary["bmi"], 0.1)
bmi_q90 = np.quantile(df_summary["bmi"], 0.9)

# Use bmi_q10 and bmi_q90 to filter df_summary and obtain predicted y values
mean_bmi_q90_outcome = np.mean(df_summary[df_summary["bmi"] > bmi_q90]["predicted_y"]) 
mean_bmi_q10_outcome = np.mean(df_summary[df_summary["bmi"] < bmi_q10]["predicted_y"])
y_diff = mean_bmi_q90_outcome - mean_bmi_q10_outcome
print(y_diff)

# Observe that the difference in predicted y between patients in top 10% and bottom 10%
# of BMI is about 150 to 160

### Evaluating BMI and HDL Outcomes

simulation_results = st.multivariate_normal.rvs(mean=mean_dia, size=20000, cov=cov_dia)
df_results = pd.DataFrame(simulation_results,columns=["age", "bmi", "bp", "tc",
													 "ldl", "hdl", "tch", "ltg", "glu"])
predicted_y = regr_model.predict(df_results)
df_y = pd.DataFrame(predicted_y, columns=["predicted_y"])
df_summary = pd.concat([df_results,df_y], axis=1)

hdl_q25 = np.quantile(df_summary["hdl"], 0.25)
hdl_q75 = np.quantile(df_summary["hdl"], 0.75)
bmi_q10 = np.quantile(df_summary["bmi"], 0.10)
bmi_q90 = np.quantile(df_summary["bmi"], 0.90)

# Complete the mean outcome definitions
bmi_q90_hdl_q75_outcome = np.mean(df_summary[(df_summary["bmi"] > bmi_q90) & (df_summary["hdl"] > hdl_q75)]["predicted_y"]) 
bmi_q10_hdl_q15_outcome = np.mean(df_summary[(df_summary["bmi"] < bmi_q10) & (df_summary["hdl"] < hdl_q25)]["predicted_y"])
y_diff = bmi_q90_hdl_q75_outcome - bmi_q10_hdl_q15_outcome
print(y_diff)

# -------------------------------------------------- #

## 4. Model Checking and Results Validation

### Evaluating distribution fit for ldl variable 
# (Kolmogorov-Smirnov test for Goodness of Fit)

# List candidate distributions to evaluate
list_of_dists = ["laplace", "norm", "expon"]
for i in list_of_dists:
    dist = getattr(st, i)
    # Fit the data to the probability distribution
    param = dist.fit(dia["ldl"])
    # Perform the ks test to evaluate goodness-of-fit
    result = st.kstest(dia["ldl"], i, args = param)
    print(result)

# Create a pairplot of bmi and hdl
sns.pairplot(df_diffs[["bmi","hdl"]])
plt.show()

# Plot a cluster map of the correlation between bmi and hdl
sns.clustermap(df_diffs[["bmi","hdl"]].corr())
plt.show()

### Reshape Data (Wide to Long) to visualize with boxplot

# Convert the hdl and bmi columns of df_diffs from wide to long format, 
# naming the values column "y_diff"
hdl_bmi_long = df_diffs.melt(value_name="y_diff",
							 value_vars=["bmi","hdl"])
print(hdl_bmi_long.head())

# Use a boxplot to visualize the results
sns.boxplot(data = hdl_bmi_long)
plt.show()

### Sensitivity Analysis - Simulation of a Profit Problem

def profit_next_year_mc(mean_inflation, mean_volume, n):
	"""
	Performs an MC simulation returning expected profit (in thousands of dollars)
	Given mean inflation rate and mean sales volume, and n number of simulation runs.
	"""
  profits = []

  for i in range(n):
    # Generate inputs by sampling from the multivariate normal distribution
    rate_sales_volume = st.multivariate_normal.rvs(mean=[mean_inflation,mean_volume],
    														 cov=cov_matrix,size=1000)
    # Deterministic calculation of company profit
    price = 100 * (100 + rate_sales_volume[:,0])/100
    volume = rate_sales_volume[:,1]
    loan_and_cost = 50 * volume + 45 * (100 + 3 * rate_sales_volume[:,0]) * (volume/100)
    profit = (np.mean(price * volume - loan_and_cost))
    profits.append(profit)

  return profits

# Run a Monte Carlo simulation 500 times using a mean_inflation of 2 and a mean_volume of 500
profits = profit_next_year_mc(mean_inflation = 2, mean_volume = 500, n = 500)

# Create a displot of the results
sns.displot(profits)
plt.show()

### Sensitivity Analysis

x1 = []
x2 = []
y = []
for infl in [0, 1, 2, 5, 10, 15, 20, 50]:
    for vol in [100, 200, 500, 800, 1000]:
		# Run profit_next_year_mc so that it samples 100 times for each infl and vol combination
        avg_prof = np.mean(profit_next_year_mc(infl,vol,100))
        x1.append(infl)
        x2.append(vol)
        y.append(avg_prof)
df_sa = pd.concat([pd.Series(x1), pd.Series(x2), pd.Series(y)], axis=1)
df_sa.columns = ["Inflation", "Volume", "Profit"]

# Create a displot of the simulation results for "Profit"
sns.displot(df_sa["Profit"])
plt.show()

### Hexbin Plot for Visualizing Sensitivity Analysis

df_sa.plot.hexbin(x="Inflation",
     y="Volume",
     C="Profit",
     reduce_C_function=np.mean,
     gridsize=10,
     cmap="viridis",
     sharex=False)
plt.show()

# Note:
# With increasing inflation, given same sales volume, simulated profits will decrease. 
# Given same inflation rate at lower value, with increasing volumes, profits will increase. 
# Profits will decrease with high inflation over 10%, even given increasing volumes.

