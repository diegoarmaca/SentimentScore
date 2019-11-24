from typing import TextIO, List, Union, Dict, Tuple
import doctest
from sentiment import *
from random import shuffle
import csv
import sys
import math

#
#
#  The file Sentiment_Explore.ipynb complements this code, adding graphs among others by using  
#  new libraries suchas mathplotlib and pandas
#
#
#

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
    test_set = []
    training_set = []
    for review in all_reviews:
        if review[0] in rating_counts:
            rating_counts[review[0]].append(review)
        else:
            rating_counts[review[0]] = [review]
    for rating, reviews in rating_counts.items():
        length_of_test_data = round(len(reviews) * (test_size))
        test_set.extend(reviews[:length_of_test_data])
        training_set.extend(reviews[length_of_test_data:])
    test_file_name = "test_" + file_name + ".txt"
    training_file_name = "training_" + file_name + ".txt"

    with open(test_file_name, 'w') as test_file:
        for row in test_set:
            test_file.write(row)

    with open(training_file_name, 'w') as training_file:
        for row in training_set: 
            training_file.write(row)

    print('The files: '+ test_file_name + ' and ' + training_file_name + ' were created')
    return {'test':test_file_name, 'training':training_file_name}
 
def remove_common_words(common_words_file:TextIO ,kss: Dict[str, List[int]])->Dict:
    common_words_file = common_words_file.read().splitlines()
    kss_sharpened = {}
    for word, value in kss.items():
        if (judge(value[0]/value[1]) != 'neutral') or (word not in common_words_file):
            kss_sharpened[word] = value
    return kss_sharpened    

def report_errors(review: str, kss: Dict[str, List[int]])->List:
    original_rating = float(review[0])
    absolute_errors = []
    pss_score = statement_pss(review, kss)
    review_scores = []
    if pss_score != None:
        is_close_val = math.isclose(pss_score, original_rating, abs_tol=0.05)
        absolute_error = round((abs(float(pss_score) - original_rating)), 2)
        absolute_errors.append(absolute_error)
        review_scores = [pss_score, round(pss_score), absolute_error, is_close_val]
        return review_scores

def report_mean_error(absolute_errors:List[float]):
    if len(absolute_errors) != 0:
        mean_absolute_error = round(sum(absolute_errors)/len(absolute_errors), 5)
        return mean_absolute_error
    
def compare_pss_models(test_file:TextIO, common_words_file:TextIO ,kss: Dict[str, List[int]], name_datasets) -> Dict:
    """Create a csv dataset with the comparison of the scores given by the kss model and the original ones. 
    Print the message "The file: reviews_comparison.csv was created" and return a dictionary with the 
    Mean_Absolute_Error(MAE) and Mean_Absolute_Error(MAE)_Sharpened of the dataset e.g., 
    {'Mean_Absolute_Error(MAE)': 1.00225, 'Mean_Absolute_Error(MAE)_Sharpened': 0.96186}
    >>> file1 = open('full.txt', 'r')
    >>> file2 = open('most_common_english_words.txt', 'r')
    >>> testing_result = compare_pss_models(file1, file2, kss, 'data')
    The file: reviews_data.csv was created
    >>> file1.close()
    >>> file2.close()
    """
    scores_comparison = []
    original_report_list = []
    sharpened_report_list = []
    original_absolute_errors = []
    sharpened_absolute_errors = []
    test_reviews = test_file.readlines()
    
    ### Sharpend kss by removing all common words
    kss_sharpened = remove_common_words(common_words_file ,kss)
   
    # Iterate over each review in order to get predicted rating and MAE for kss and the sharpened version of kss          
    for review in test_reviews:
        statement = review[1:].strip()
        original_report = report_errors(review, kss)
        sharpened_report = report_errors(review, kss_sharpened)
        if statement_pss(review, kss) != None and statement_pss(review, kss_sharpened):
            original_report_list.append(original_report)
            sharpened_report_list.append(sharpened_report)
            original_absolute_errors.append(original_report[2])
            sharpened_absolute_errors.append(sharpened_report[2])
            scores_comparison.append([statement, review[0], 
                                      round(original_report[0],2), original_report[1], original_report[2], original_report[3], 
                                      round(sharpened_report[0],2), sharpened_report[1],sharpened_report[2], sharpened_report[3]])
            
    # Get mean absolute errors from the original and the sharpened model        
    mean_absolute_error = report_mean_error(original_absolute_errors)
    mean_absolute_error_sharpened = report_mean_error(sharpened_absolute_errors)

    # Save all reviews with their predicted scores and MAE using kss and kss_sharpened     
    with open('reviews_'+ name_datasets + '.csv', mode ='w') as comparison_file:
        comparison_writer = csv.writer(comparison_file, delimiter=",", quotechar='"', quoting = csv.QUOTE_MINIMAL)
        comparison_writer.writerow(["Mean Absolute Error(MAE): " + str(mean_absolute_error) ])
        comparison_writer.writerow(["Mean Absolute Error(MAE)   Sharpened: " + str(mean_absolute_error_sharpened) ])
        comparison_writer.writerow(["-","-","-","-","-"])
        comparison_writer.writerow(["Review", "Original Rating", "PSS Score", "Predicted Rating", "Absolute Error", "Evaluation Result", "PSS Score Sharpened", "Predicted Rating Sharpened","Absolute Error Sharpened", "Evaluation Result Sharpened"])
        for row in scores_comparison:
            comparison_writer.writerow(row)

    print('The file: ' + 'reviews_'+ name_datasets + '.csv' + ' was created')
    return {"Mean_Absolute_Error(MAE)": mean_absolute_error,"Mean_Absolute_Error(MAE)_Sharpened":mean_absolute_error_sharpened}


def execute_test(datasets: Dict[str, str], partition_size):
    """Precondition: the dictionary should have the form {name_dataset:dataset} e.g., {"small":"small.txt","medium":"medium.txt","full":"full.txt"}  
    Run compare_pss_models function for various datasets.
    """
    for name in datasets:
        with open(datasets[name], 'r') as file:
            file_names = partition_dataset(file, name, partition_size)
        with open(file_names['training'], 'r') as training_file:
                kss = extract_kss(training_file)  
        with open(file_names['test'], 'r') as test:
            with open(most_common_words) as common_words_file:
                testing_result = compare_pss_models(test, common_words_file, kss, name)

if __name__ == "__main__":
    #Create a dictionary containing diferent datasets, in order to compare accuracies among each other.
    datasets =  {
        "small"     : "small.txt",
        "medium"    : "medium.txt",
        "full"      : "full.txt"
    }        
    most_common_words = "most_common_english_words.txt"
    
    #Test for the function execute_test 
    execute_test(datasets, 0.1)
    doctest.testmod()
