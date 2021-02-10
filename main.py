# Eclat

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# eclat rules: using Apriori as Eclat is simplified Apriori rules (respect support)
from apyori import apriori

# Putting the results well organised into a Pandas DataFrame
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results] # change index if min_length and max_length are not 2
    rhs         = [tuple(result[2][0][1])[0] for result in results] # change index if min_length and max_length are not 2
    supports    = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))

def eclat():
    # import data: this dataset does NOT have columns -> each rows have different categorical variables
    # f column names are passed explicitly then the behavior is identical to header=None. (No columns)
    dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

    transaction = dataset.values.astype(str).tolist()

    # no need independent variables for Eclat

    # no need dependent variable for Eclat

    # no need splitting dataset

    # no need feature scaling as they are categorical variables

    # train/fit the dataset with Apriori but only respect Support
    rules = apriori(transaction, min_support=float((3 * 7) / len(dataset)), min_confidence=0.2, min_lift=3, min_length=2, max_length=2)

    # visualizing the rules
    result = list(rules)  # put the rules into the list
    print(result)

    '''
    {'chicken', 'extra dark chocolate'}
    items_base=frozenset({'extra dark chocolate'} this means when people buy dark chocolate
    items_add=frozenset({'chicken'}) then, the people buy chicken
    confidence=0.23  -> 23% buy the chicken in the situations
    '''

    resultsinDataFrame = pd.DataFrame(inspect(result), columns=['Left Hand Side', 'Right Hand Side', 'Support'])
    print(resultsinDataFrame)

    # resultsinDataFrame is object of data class

    # sorting by Support top 10
    print('\n' + str(resultsinDataFrame.nlargest(n=10, columns='Support')))


if __name__ == '__main__':
    eclat()

