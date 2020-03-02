# Decision-Tree-Entropy-Analysis
You are asked to use a decision tree model to predict the usage of a car.  The data is the claim_history.csv which has 10,302 observations.  The analysis specifications are:

Target Variable
•	CAR_USE. The usage of a car.  This variable has two categories which are Commercial and Private.  The Commercial category is the Event value.

Nominal Predictor
•	CAR_TYPE. The type of a car.  This variable has six categories which are Minivan, Panel Truck, Pickup, SUV, Sports Car, and Van.
•	OCCUPATION. The occupation of the car owner.  This variable has nine categories which are Blue Collar, Clerical, Doctor, Home Maker, Lawyer, Manager, Professional, Student, and Unknown.

Ordinal Predictor
•	EDUCATION. The education level of the car owner.  This variable has five ordered categories which are Below High School < High School < Bachelors < Masters < Doctors.

Analysis Specifications
•	Partition. Specify the target variable as the stratum variable. Use stratified simple random sampling to put 75% of the records into the Training partition, and the remaining 25% of the records into the Test partition.  The random state is 60616.
•	Decision Tree.  The maximum number of branches is two.  The maximum depth is two.  The split criterion is the Entropy metric.

Question 1 (20 points)
Please provide information about your Data Partition step.  You may call the train_test_split() function in the sklearn.model_selection module in your code.
a)	(5 points). Please provide the frequency table (i.e., counts and proportions) of the target variable in the Training partition?
b)	(5 points). Please provide the frequency table (i.e., counts and proportions) of the target variable in the Test partition?
c)	(5 points). What is the probability that an observation is in the Training partition given that CAR_USE = Commercial?
d)	(5 points). What is the probability that an observation is in the Test partition given that CAR_USE = Private?

Question 2 (40 points)
Please provide information about your decision tree.  You will need to write your own Python program to find the answers.
a)	(5 points). What is the entropy value of the root node?
b)	(5 points). What is the split criterion (i.e., predictor name and values in the two branches) of the first layer?
c)	(10 points). What is the entropy of the split of the first layer?
d)	(5 points). How many leaves?
e)	(10 points). Describe all your leaves.  Please include the decision rules and the counts of the target values.
f)	(5 points). What is the Kolmogorov-Smirnov statistic and the event probability cutoff value? 

Question 3 (40 points)
Please apply your decision tree to the Test partition and then provide the following information. You will choose whether to call sklearn functions or write your own Python program to find the answers.
a)	(5 points). Use the proportion of target Event value in the training partition as the threshold, what is the Misclassification Rate in the Test partition?
b)	(5 points). Use the Kolmogorov-Smirnov event probability cutoff value in the training partition as the threshold, what is the Misclassification Rate in the Test partition?
c)	(5 points). What is the Root Average Squared Error in the Test partition?
d)	(5 points). What is the Area Under Curve in the Test partition?
e)	(5 points). What is the Gini Coefficient in the Test partition?
f)	(5 points). What is the Goodman-Kruskal Gamma statistic in the Test partition?
g)	(10 points). Generate the Receiver Operating Characteristic curve for the Test partition.  The axes must be properly labeled.  Also, don’t forget the diagonal reference line.
