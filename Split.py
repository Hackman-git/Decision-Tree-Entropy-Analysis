'''
Name: Olugbenga Abdulai
CWID: A20447331
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

'''
1(a)
'''
claim = pd.read_csv(r"C:/Users/abdul/Desktop/CS 584/HW/HW 3/claim_history.csv")
split = StratifiedShuffleSplit(test_size=0.25, random_state=60616)
for train_index, test_index in split.split(claim, claim.CAR_USE):
    train_set = claim.loc[train_index]
    test_set = claim.loc[test_index]
train_target = pd.DataFrame(train_set.CAR_USE.value_counts())
train_target['proportion'] = train_target.apply(lambda x: x.CAR_USE / train_set.shape[0], axis=1)
train_target.rename(columns={'CAR_USE': 'counts'}, inplace=True)
print('\nproportions in training set\n', train_target)

'''
1(b)
'''
test_target = pd.DataFrame(test_set.CAR_USE.value_counts())
test_target['proportion'] = test_target.apply(lambda x: x.CAR_USE / test_set.shape[0], axis=1)
test_target.rename(columns={'CAR_USE': 'counts'}, inplace=True)
print('\nproportions in test set\n', test_target)

print(claim.CAR_USE.value_counts())
print(claim.shape)

'''
1(c)
'''
# What is the probability that an observation is in the Training partition given that CAR_USE = Commercial?
prob_comm_given_train = train_set[train_set.CAR_USE == "Commercial"].shape[0] / train_set.shape[0]
prob_training = 0.75
prob_test = 0.25
prob_comm_given_test = test_set[test_set.CAR_USE == "Commercial"].shape[0] / test_set.shape[0]
prob_train_given_comm = (prob_comm_given_train * prob_training) / ((prob_comm_given_train * prob_training)
                                                                   + prob_comm_given_test * prob_test)

print("\nProbability of Training partition given car_use is commercial:\n", prob_train_given_comm)

'''
1(d)
'''
# What is the probability that an observation is in the Test partition given that CAR_USE = Private?
prob_priv_given_test = test_set[test_set.CAR_USE == "Private"].shape[0] / test_set.shape[0]
prob_priv_given_train = train_set[train_set.CAR_USE == "Private"].shape[0] / train_set.shape[0]
prob_test_given_priv = (prob_priv_given_test * prob_test) / ((prob_priv_given_test * prob_test) \
                                                             + (prob_priv_given_train * prob_training))

print("\nProbability of test partition given car_use is private:\n", prob_test_given_priv)

'''
2(a)
'''
pd.crosstab(index=[claim.CAR_TYPE, claim.OCCUPATION, claim.EDUCATION], columns=claim.CAR_USE,
            margins=True, dropna=True, )
root_entropy = -((3789 / 10302) * np.log2(3789 / 10302) + (6513 / 10302) * np.log2(6513 / 10302))
print('\nroot entropy:\n', root_entropy)

'''
2(b)
'''
# subsetting the data to predictors and target we're concerned about
claim = claim.loc[:, ['CAR_TYPE', 'OCCUPATION', 'EDUCATION', 'CAR_USE']]


# all possible nominal variables splits
def all_possible_splits(nominal_vals):
    from itertools import combinations, chain
    subsets = [i for j in range(1, len(nominal_vals)) for i in combinations(nominal_vals, j)]
    splits = []
    count = 0
    for i in range(int(len(subsets) / 2)):
        count += 1
        splits.append(list(chain(subsets[i])))
        print(list(chain(subsets[i])), '       ', count)
    return (splits)


# Check for NAs
print(claim.OCCUPATION.isna().sum(), '\n')

'''
Precondition: data must have the first column holding the
predictor and second column as target variable
'''
def entropyNominalSplit(data, splits):
    tableEntropies = []
    for i in range(len(splits)):
        data['left_branch'] = data.apply(lambda x: x[0] in splits[i], axis=1)

        crosstab = pd.crosstab(index=data.left_branch, columns=data.iloc[:, 1], margins=True, dropna=True)
        print(crosstab)

        nrows = crosstab.shape[0]
        ncols = crosstab.shape[1]

        tableEntropy = 0
        for iRow in range(nrows - 1):
            rowEntropy = 0
            for iCol in range(ncols):
                proportion = crosstab.iloc[iRow, iCol] / crosstab.iloc[iRow, (ncols - 1)]
                if proportion > 0:
                    rowEntropy -= proportion * np.log2(proportion)
            print('Row = ', iRow, 'Entropy = ', rowEntropy, '\n')
            tableEntropy += rowEntropy * crosstab.iloc[iRow, (ncols - 1)]
        tableEntropy = tableEntropy / crosstab.iloc[(nrows - 1), (ncols - 1)]

        tableEntropies.append(tableEntropy)

    return (tableEntropies)


occupation_df = claim.iloc[:, [1, 3]]
print(occupation_df.head(), '\n')

occ = list(set(occupation_df.OCCUPATION.values))
occ_splits = all_possible_splits(occ)
occ_entropies = entropyNominalSplit(occupation_df, occ_splits)
print(occ_entropies, '\n')

ind = np.argmin(occ_entropies)
best_occ_split = occ_splits[ind]
print('\nbest occupation split', best_occ_split)

best_occ_split_entropy = occ_entropies[ind]
reduction_occ = root_entropy - best_occ_split_entropy
print('\nreduction for occupation', reduction_occ)

car_type_df = claim.iloc[:, [0, 3]]
cars = list(set(car_type_df.CAR_TYPE.values))
car_splits = all_possible_splits(cars)
car_entropies = entropyNominalSplit(car_type_df, car_splits)
print('\ncar entropies', car_entropies)

ind = np.argmin(car_entropies)
best_car_split = car_splits[ind]
print('\nbest car split:', best_car_split)

best_car_split_entropy = car_entropies[ind]
reduction_car = root_entropy - best_car_split_entropy
print('\nreduction from root for car:', reduction_car)

'''
Preconditions: data must have the first column holding the
predictor and second column as target variable. Splits must be encoded
as integer values.
'''
def entropyOrdinalSplit(data, splits):
    tableEntropies = []
    for i in range(len(splits)):
        data['left_branch'] = data.apply(lambda x: x[0] <= splits[i], axis=1)

        crosstab = pd.crosstab(index=data.left_branch, columns=data.iloc[:, 1], margins=True, dropna=True)
        print(crosstab)

        nrows = crosstab.shape[0]
        ncols = crosstab.shape[1]

        tableEntropy = 0
        for iRow in range(nrows - 1):
            rowEntropy = 0
            for iCol in range(ncols):
                proportion = crosstab.iloc[iRow, iCol] / crosstab.iloc[iRow, (ncols - 1)]
                if proportion > 0:
                    rowEntropy -= proportion * np.log2(proportion)
            print('Row = ', iRow, 'Entropy = ', rowEntropy, '\n')
            tableEntropy += rowEntropy * crosstab.iloc[iRow, (ncols - 1)]
        tableEntropy = tableEntropy / crosstab.iloc[(nrows - 1), (ncols - 1)]

        tableEntropies.append(tableEntropy)

    return (tableEntropies)


education_splits = [0, 1, 2, 3]
education_df = claim.iloc[:, [2, 3]]
# check for NAs
print(education_df.EDUCATION.isna().sum())
education_df['Edu_enc'] = education_df.EDUCATION.map({'Below High School': 0, 'High School': 1, 'Bachelors': 2,
                                                      'Masters': 3, 'Doctors': 4})
education_df = education_df[['Edu_enc', 'CAR_USE', 'EDUCATION']]
edu_entropies = entropyOrdinalSplit(education_df, education_splits)
print('\neducation entropies:\n', edu_entropies)

ind = np.argmin(edu_entropies)
print('\nbest education split:', education_splits[ind])
best_edu_split = ['Below High School']

best_edu_split_entropy = edu_entropies[ind]
reduction_edu = root_entropy - best_edu_split_entropy
print('\nreduction from root for education:', reduction_edu)

'''
The best reduction in entropy is from Occupation split. The split criterion is OCCUPATION split into
['Blue Collar', Student', 'Unknown'] and ['Clerical', 'Doctor', 'Home Maker', 'Lawyer', 'Manager', 
'Professional']
'''

'''
2(c)
'''
'''
Split entropy of the first layer as calculated above is 0.71258
'''

'''
2(d)
'''
left_branch = claim[claim.apply(lambda x: x[1] in best_occ_split, axis=1)]
print('\n', left_branch.OCCUPATION.unique())
left_branch = left_branch.reset_index(drop=True)
right_branch = claim[claim.apply(lambda x: x[1] not in best_occ_split, axis=1)]
right_branch = right_branch.reset_index(drop=True)

'''
analysis for splitting left branch
'''
car_types_left = left_branch.CAR_TYPE.unique()
occ_left = left_branch.OCCUPATION.unique()
edu_left = left_branch.EDUCATION.unique()

# split by occupation
left_occ_splits = all_possible_splits(occ_left)
left_occ_df = left_branch.iloc[:, [1, 3]]
left_occ_entropies = entropyNominalSplit(left_occ_df, left_occ_splits)
print('\nleft occupation entropies:\n', left_occ_entropies)

ind = np.argmin(left_occ_entropies)
best_left_occ_split = left_occ_splits[ind]
print("\n best left occupation split:\n", best_left_occ_split)

best_left_occ_split_entropy = left_occ_entropies[ind]
print("\nbest left occupation split entropy: ", best_left_occ_split_entropy)

# split by car type
left_car_df = left_branch.iloc[:, [0, 3]]
left_car_splits = all_possible_splits(car_types_left)
left_car_entropies = entropyNominalSplit(left_car_df, left_car_splits)

ind = np.argmin(left_car_entropies)
best_left_car_split = left_car_splits[ind]
print("\nbest left car split: ",best_left_car_split)

best_left_car_split_entropy = left_car_entropies[ind]
print("\nbest left car split entropy: ", best_left_car_split_entropy)

# split by education
left_edu_df = left_branch.iloc[:, [2, 3]]
left_edu_df['Edu_enc'] = left_edu_df.EDUCATION.map({'Below High School': 0, 'High School': 1, 'Bachelors': 2,
                                                      'Masters': 3, 'Doctors': 4})
left_edu_df = left_edu_df[['Edu_enc', 'CAR_USE', 'EDUCATION']]
left_edu_splits = [0,1,2,3]
left_edu_entropies = entropyOrdinalSplit(left_edu_df, left_edu_splits)
ind = np.argmin(left_edu_entropies)
best_left_edu_split = left_edu_splits[ind]
best_left_edu_split = ['Below High School']
print("\nbest left education split: ", best_left_edu_split)

best_left_edu_split_entropy = left_edu_entropies[ind]
print("\nbest left education split entropy: ", best_left_edu_split_entropy)

'''
education is the best split to obtain the left leaves
'''
leaf_1 = left_branch[left_branch.apply(lambda x: x[2] in best_left_edu_split, axis=1)]
leaf_1 = leaf_1.reset_index(drop=True)
print("\nLeaf 1:\n", leaf_1)

leaf_2 = left_branch[left_branch.apply(lambda x: x[2] not in best_left_edu_split, axis=1)]
leaf_2 = leaf_2.reset_index(drop=True)
print("\nLeaf 2:\n", leaf_2)

'''
Analysis for right branch
'''
car_types_right = right_branch.CAR_TYPE.unique()
occ_right = right_branch.OCCUPATION.unique()
edu_right = right_branch.EDUCATION.unique()

# split by occupation
right_occ_df = right_branch.iloc[:, [1, 3]]
right_occ_splits = all_possible_splits(occ_right)
right_occ_entropies = entropyNominalSplit(right_occ_df, right_occ_splits)

ind = np.argmin(right_occ_entropies)
best_right_occ_split = right_occ_splits[ind]
print("\n best right occupation split:\n", best_right_occ_split)

best_right_occ_split_entropy = right_occ_entropies[ind]
print("\nbest right occupation split entropy: ", best_right_occ_split_entropy)

# split by car
right_car_df = right_branch.iloc[:, [0, 3]]
right_car_splits = all_possible_splits(car_types_right)
right_car_entropies = entropyNominalSplit(right_car_df, right_car_splits)

ind = np.argmin(right_car_entropies)
best_right_car_split = right_car_splits[ind]
print("\nbest right car split: ", best_right_car_split)

best_right_car_split_entropy = right_car_entropies[ind]
print("\nbest right car split entropy: ", best_right_car_split_entropy)

# split by education
right_edu_df = right_branch.iloc[:, [2, 3]]
right_edu_df['Edu_enc'] = right_edu_df.EDUCATION.map({'Below High School': 0, 'High School': 1, 'Bachelors': 2,
                                                      'Masters': 3, 'Doctors': 4})
right_edu_df = right_edu_df[['Edu_enc', 'CAR_USE', 'EDUCATION']]
right_edu_splits = [0,1,2,3]
right_edu_entropies = entropyOrdinalSplit(right_edu_df, right_edu_splits)

ind = np.argmin(right_edu_entropies)
best_right_edu_split = right_edu_splits[ind]
print("\nbest right education split: ", best_right_edu_split)
best_right_edu_split = ['Bachelors']

best_right_edu_split_entropy = right_edu_entropies[ind]
print("\nbest right education split entropy: ", best_right_edu_split_entropy)

'''
car type is the best split to obtain the right leaves
'''
leaf_3 = right_branch[right_branch.apply(lambda x: x[0] in best_right_car_split, axis=1)]
leaf_3 = leaf_3.reset_index(drop=True)
print("\nLeaf 3:\n", leaf_3)

leaf_4 = right_branch[right_branch.apply(lambda x: x[0] not in best_right_car_split, axis=1)]
leaf_4 = leaf_4.reset_index(drop=True)
print("\nLeaf 4:\n", leaf_4)

'''
Answer: 4 leaves
'''

'''
2(e)
'''

'''
leaf 1 decision rule: Education is "Below High School"
leaf 2 decision rule: Education is one of ['High School', 'Bachelors', 'Doctors', 'Masters']
leaf 3 decision rule: Car type is one of ['Minivan', 'SUV', 'Sports Car']
leaf 4 decision rule: Car type is one of ['Van', 'Pickup', 'Panel Truck']
'''

# leaf 1 target distribution
print('\nleaf 1 target distribution\n', leaf_1.CAR_USE.value_counts())

# leaf 2 target distribution
print('\nleaf 2 target distribution\n', leaf_2.CAR_USE.value_counts())

# leaf 3 target distribution
print('\nleaf 3 target distribution\n', leaf_3.CAR_USE.value_counts())

# leaf 4 target distribution
print('\nleaf 4 target distribution\n', leaf_4.CAR_USE.value_counts())

'''
2(f)
'''
X_train = train_set.loc[:, ['CAR_TYPE', 'OCCUPATION', 'EDUCATION']]
X_train = X_train.reset_index(drop=True)
y_train = train_set.loc[:, ['CAR_USE']]
y_train = y_train.reset_index(drop=True)

LE = LabelEncoder()
X_train.OCCUPATION = LE.fit_transform(X_train.OCCUPATION)
X_train.CAR_TYPE = LE.fit_transform(X_train.CAR_TYPE)
X_train.EDUCATION = LE.fit_transform(X_train.EDUCATION)

model = DecisionTreeClassifier(random_state=60616, max_depth=2, criterion='entropy')
model.fit(X_train, y_train)

X_test = test_set.loc[:, ['CAR_TYPE', 'OCCUPATION', 'EDUCATION']]
X_test = X_test.reset_index(drop=True)
y_test = test_set.loc[:, ['CAR_USE']]
y_test = y_test.reset_index(drop=True)

X_test.OCCUPATION = LE.fit_transform(X_test.OCCUPATION)
X_test.CAR_TYPE = LE.fit_transform(X_test.CAR_TYPE)
X_test.EDUCATION = LE.fit_transform(X_test.EDUCATION)
pred = model.predict(X_test)
predict_prob = model.predict_proba(X_test)
prob = predict_prob[:, 1]

fpr, tpr, thresh = roc_curve(y_test, prob, pos_label='Private')
cutoff = np.where(thresh > 1.0, np.nan, thresh)

plt.plot(cutoff, tpr, marker='o', label='True Positive',
         color='blue', linestyle='solid', linewidth=2, markersize=6)
plt.plot(cutoff, fpr, marker='o', label='False Positive',
         color='red', linestyle='solid', linewidth=2, markersize=6)
plt.grid(True)
plt.xlabel("Probability Threshold")
plt.ylabel("Positive Rate")
plt.legend(loc='upper right', shadow=True, fontsize='large')
plt.show()

# K-S statistic and predicted probability cut-off value from the plot
kolm_stat = 0.8348 - 0.2597
print('\nK-S statistic:', kolm_stat)
prob_cutoff = 0.841
print('\ncut-off:', prob_cutoff)

'''
3(a)
'''
prop = train_set[train_set.CAR_USE == "Private"].shape[0] / train_set.shape[0]
predY = np.empty_like(y_test)
for i in range(len(y_test)):
    if prob[i] > prop:
        predY[i] = "Private"
    else:
        predY[i] = "Commercial"

accuracy = accuracy_score(y_test, predY)
misclass = 1 - accuracy
print('\nMisclassification rate:', misclass)

'''
3(b)
'''
predY = np.empty_like(y_test)
for i in range(len(y_test)):
    if prob[i] > prob_cutoff:
        predY[i] = "Private"
    else:
        predY[i] = "Commercial"

accuracy_ = accuracy_score(y_test, predY)
misclass_ = 1 - accuracy
print('\nmisclassification rate:', misclass_)

'''
3(c)
'''
RASE = 0.0
y_test_ = np.array(y_test)
for i in range(len(y_test_)):
    if y_test_[0] == 'Private':
        RASE += (1 - prob[i]) ** 2
    else:
        RASE += (0 - prob[i]) ** 2
RASE = np.sqrt(RASE / len(y_test_))
print('\nroot average squared error:', RASE)

'''
3(d)
'''
auc = roc_auc_score(y_test, prob)
print('\nAUC:', auc)

'''
3(e)
'''
x = pd.DataFrame(predict_prob)
private_prob = sorted(x.loc[:, 1])
comm_prob = sorted(x.loc[:, 0])

conc = 0
disc = 0
tie = 0
for i in range(len(private_prob)):
    if private_prob[i] > comm_prob[i]:
        conc += 1
    elif private_prob[i] < comm_prob[i]:
        disc += 1
    else:
        tie += 1

gini = (conc - disc) / (conc + disc + tie)
print('\nGini index: ', gini)

'''
3(f)
'''
gamma = (conc - disc) / (conc + disc)
print("\nGamma: ", gamma)

'''
3(f)
'''
oneMinusSpecificity = np.append([0], fpr)
sensitivity = np.append([0], tpr)
oneMinusSpecificity = np.append(oneMinusSpecificity, [1])
sensitivity = np.append(sensitivity, [1])

plt.figure(figsize=(6, 6))
plt.plot(oneMinusSpecificity, sensitivity, marker='o',
         color='blue', linestyle='solid', linewidth=2, markersize=6)
plt.plot([0, 1], [0, 1], color='red', linestyle=':')
plt.grid(True)
plt.xlabel("1 - Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
plt.axis("equal")
plt.show()
