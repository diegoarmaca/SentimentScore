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
 

def testing_pss(test_file:TextIO, common_words_file:TextIO ,kss: Dict[str, List[int]], name_datasets) -> Dict:
    """Create a csv dataset with the comparison of the scores given by the kss model and the original ones. Print the message "The file: reviews_comparison.csv was created" and return a dictionary with the Mean_Absolute_Error(MAE) and Mean_Absolute_Error(MAE)_Sharpened of the dataset e.g., {'Mean_Absolute_Error(MAE)': 1.00225, 'Mean_Absolute_Error(MAE)_Sharpened': 0.96186}
    
    >>> file1 = open('full.txt', 'r')
    
    >>> file2 = open('most_common_english_words.txt', 'r')
    
    >>> testing_result = testing_pss(file1, file2, kss, 'data')
    The file: reviews_data.csv was created
    >>> file1.close()
    
    >>> file2.close()
    
    """
    common_words_file = common_words_file.read().splitlines()
    review_scores = []
    absolute_errors = []
    test_reviews = test_file.readlines()
    
    #sharpened variables
    kss_sharpened = {}
    absolute_errors_sharpened = []
    sum_of_frequencies = 0
    
    ### remove all common words
    for word, value in kss.items():
        if word not in common_words_file:
            kss_sharpened[word] = value
            sum_of_frequencies += value[1]
    
       
    #Iterate over each review in order to get predicted rating and MAE for kss and the sharpened version of kss          
    for review in test_reviews:
        statement = review[1:].strip()
        original_rating = float(review[0])
        
        #Calculate predicted rating
        predicted_rating = statement_pss(review, kss)
        if predicted_rating != None:
            is_close_val = math.isclose(predicted_rating, original_rating, abs_tol=0.05)
            absolute_error = round((abs(float(predicted_rating) - original_rating)), 2)
            absolute_errors.append(absolute_error)
            mean_absolute_error = round(sum(absolute_errors)/len(absolute_errors), 5)
            
        
        #Sharp and calculate predicted rating
        predicted_rating_sharpened = statement_pss(review, kss_sharpened)
        if predicted_rating_sharpened != None:
            is_close_val_sharpened = math.isclose(predicted_rating_sharpened, original_rating, abs_tol=0.05)
            absolute_error_sharpened = round((abs(float(predicted_rating_sharpened) - original_rating)), 2)
            absolute_errors_sharpened.append(absolute_error_sharpened)
            mean_absolute_error_sharpened = round(sum(absolute_errors_sharpened)/len(absolute_errors_sharpened), 5)
            
        #Append predicted and sharpened values in the same list 
        if predicted_rating != None and predicted_rating_sharpened != None:
            review_scores.append([statement, original_rating, round(predicted_rating, 2), round(predicted_rating), absolute_error,is_close_val, round(predicted_rating_sharpened, 2), round(predicted_rating_sharpened), absolute_error_sharpened, is_close_val_sharpened])

    #Save all reviews with their predicted scores and MAE using kss and kss_sharpened     
    with open('reviews_'+ name_datasets + '.csv', mode ='w') as comparison_file:
        comparison_writer = csv.writer(comparison_file, delimiter=",", quotechar='"', quoting = csv.QUOTE_MINIMAL)
        comparison_writer.writerow(["Mean Absolute Error(MAE): " + str(mean_absolute_error) ])
        comparison_writer.writerow(["Mean Absolute Error(MAE) Sharpened: " + str(mean_absolute_error_sharpened) ])
        comparison_writer.writerow(["-","-","-","-","-"])
        comparison_writer.writerow(["Review", "Original Rating", "PSS Score", "Predicted Rating", "Absolute Error", "Evaluation Result", "PSS Score Sharpened", "Predicted Rating Sharpened","Absolute Error Sharpened", "Evaluation Result Sharpened"])
        for row in review_scores:
            comparison_writer.writerow(row)

    print('The file: ' + 'reviews_'+ name_datasets + '.csv' + ' was created')
    return {"Mean_Absolute_Error(MAE)": mean_absolute_error,"Mean_Absolute_Error(MAE)_Sharpened":mean_absolute_error_sharpened}


def execute_test(datasets: Dict[str, str], partition_size):
    """Precondition: the dictionary should have the form {name_dataset:dataset} e.g., {"small":"small.txt","medium":"medium.txt","full":"full.txt"}  
    
    Run testing_pss function for various datasets.
    """
    for name in datasets:
        with open(datasets[name], 'r') as file:
            file_names = partition_dataset(file, name, partition_size)
        with open(file_names['training'], 'r') as training_file:
                kss = extract_kss(training_file)  
        with open(file_names['test'], 'r') as test:
            with open(most_common_words) as common_words_file:
                testing_result = testing_pss(test, common_words_file, kss, name)
        print(testing_result)
    


if __name__ == "__main__":
    
    #Create a dictionary containing diferent datasets, in order to compare accuracies among each other.
    datasets =  {
        "small"     : "small.txt",
        "medium"    : "medium.txt",
        "full"      : "full.txt"
    }        
    
    #most_common_words = "most_common_english_words.txt"
    most_common_words = "stoplist.txt"
    
    #Test for the function execute_test 
    execute_test(datasets, 0.1)

    # Pick a dataset  
    # dataset = 'tiny.txt'
    # dataset = 'small.txt'
    #dataset = 'medium.txt'
    dataset = 'full.txt'

     # Test if the training and test datasets were created
    name_datasets = 'data'
     
    with open(dataset, 'r') as file:
        file_names = partition_dataset(file, name_datasets, 0.2)

 

    # Training the model with the training dataset created
    with open(file_names['training'], 'r') as training_file:
        kss = extract_kss(training_file)


    # Testing the results with the test dataset created
    with open(file_names['test'], 'r') as test_file:
        with open("most_common_english_words.txt") as common_words_file:
            testing_result = testing_pss(test_file, common_words_file, kss, name_datasets)
    print(testing_result)
   
    # Use test mode

    doctest.testmod()