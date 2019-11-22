from typing import TextIO, List, Union, Dict, Tuple
import doctest
from sentiment import *
from random import shuffle
import csv
import sys
import math

#
#
#
#
#
#
#

# Your exploration functions here
# Follow FDR
def partition_dataset(file:TextIO, file_name:str, test_size:float) -> Dict:
    """Precondition: test_size > 0.0 and < 1.0 (one decimal)
    Create two datasets sorted randomly from the original. The test_dataset has the size
    requested in test_size, and the trainin_dataset has the remaining size.
    Print a message e.g., "The files: test_data.txt and training_data.txt were created",
    and return a dictionary e.g., {'test': 'test_data.txt', 'training': 'training_data.txt'}

    >>> file_names = partition_dataset(open('full.txt', 'r'), 'data', 0.2)
    The files: test_data.txt and training_data.txt were created
    >>> file_names
    {'test': 'test_data.txt', 'training': 'training_data.txt'}
    """
    all_reviews = file.readlines()
    shuffle(all_reviews)
    rating_counts = {}
    test_reviews = []
    training_reviews = []
    for review in all_reviews:
        if review[0] in rating_counts:
            rating_counts[review[0]].append(review)
        else:
            rating_counts[review[0]] = [review]
    #print(rating_counts['4'])
    for rating, reviews in rating_counts.items():
        #print(rating, values)
        length_of_test_data = round(len(reviews) * (test_size))
        test_reviews.extend(reviews[:length_of_test_data])
        training_reviews.extend(reviews[length_of_test_data:])
    test_file_name = "test_" + file_name + ".txt"
    training_file_name = "training_" + file_name + ".txt"

    with open(test_file_name, 'w') as test_file:
        for row in test_reviews:
            test_file.write(row)

    with open(training_file_name, 'w') as training_file:
        for row in training_reviews: 
            training_file.write(row)

    print('The files: '+ test_file_name + ' and ' + training_file_name + ' were created')
    return {'test':test_file_name, 'training':training_file_name}
 

def testing_pss(test_file:TextIO, kss: Dict[str, List[int]], name_datasets ) -> Dict:
    """Create a csv dataset with the comparison of the scores given by the kss model and the original ones. Print the message "The file: reviews_comparison.csv was created" and return the dictionary {'file':'reviews_comparison.csv'}
    >>> testing_result = testing_pss(open('full.txt', 'r'), kss)
    The file: reviews_comparison.csv was created
    >>> testing_result
    {'file': 'reviews_comparison.csv'}
    """

    review_scores = []
    absolute_errors = []
    test_reviews = test_file.readlines()
    for review in test_reviews:
        statement = review[1:].strip()
        predicted_rating = statement_pss(review, kss)
        original_rating = float(review[0])

        if predicted_rating != None:
            is_close_val = math.isclose(predicted_rating, original_rating, abs_tol=0.05)
            absolute_error = round((abs(float(predicted_rating) - original_rating)), 2)
            absolute_errors.append(absolute_error)
            mean_absolute_error = round(sum(absolute_errors)/len(absolute_errors), 5)
            review_scores.append([statement, round(predicted_rating, 2), round(predicted_rating), original_rating, absolute_error, is_close_val])

    print(mean_absolute_error)
    with open('reviews_'+ name_datasets + '.csv', mode ='w') as comparison_file:
        comparison_writer = csv.writer(comparison_file, delimiter=",", quotechar='"', quoting = csv.QUOTE_MINIMAL)
        comparison_writer.writerow(["Mean Absolute Error(MAE): " + str(mean_absolute_error) ])
        comparison_writer.writerow(["-","-","-","-","-"])
        comparison_writer.writerow(["Review", "PSS Score", "Predicted Rating", "Original Rating", "Absolute Error", "Evaluation Result"])
        print(len(review_scores))
        for row in review_scores:
            comparison_writer.writerow(row)

    print('The file:' + 'reviews_'+ name_datasets + '.csv' + ' was created')
    return {'Mean Absolute Error(MAE): ', mean_absolute_error}

def execute_test(dataset, name_datasets):
    with open(dataset, 'r') as file:
        file_names = partition_dataset(file, name_datasets, 0.2)
    with open(file_names['training'], 'r') as training_file:
            kss = extract_kss(training_file)  
    with open(file_names['test'], 'r') as test:
            testing_result = testing_pss(test, kss, name_datasets)
    

if __name__ == "__main__":
    execute_test("small.txt", "small")
    execute_test("medium.txt", "medium")
    execute_test("full.txt", "full")
    # Pick a dataset  
    # dataset = 'tiny.txt'
    # dataset = 'small.txt'
    #dataset = 'medium.txt'
    # dataset = 'full.txt'

    # Test if the training and test datasets were created
    # name_datasets = 'data'
    #with open(dataset, 'r') as file:
        #file_names = partition_dataset(file, name_datasets, 0.2)

    # Training the model with the training dataset created
    #with open(file_names['training'], 'r') as training_file:
           # kss = extract_kss(training_file)  

    # Testing the results with the test dataset created
    # with open(file_names['test'], 'r') as test:

    # Use test mode

    #doctest.testmod()
