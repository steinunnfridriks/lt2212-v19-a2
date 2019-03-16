import os, sys
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# simdoc.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here

parser = argparse.ArgumentParser(description="Compute some similarity statistics.")
parser.add_argument("vectorfile", type=str,
                    help="The name of the input  file for the matrix data.")

args = parser.parse_args()

print("Reading matrix from {}.".format(args.vectorfile))

#_________________________________________________

def file_to_array(vectorfile):
    """Getting the dataframe from the outputfile from gendoc"""
    dataframe = pd.read_csv(vectorfile)
    clean = dataframe.drop(dataframe.columns[0], axis=1)
    array = np.array(clean)
    return array

def get_range(vectorfile):
    crude = []
    grain = []
    total_len = []
    dataframe = pd.read_csv(vectorfile)
    for label in dataframe[dataframe.columns[0]]:
        total_len.append(label)
        if "crude" in label:
            crude.append(label)
        elif label==int:


        else:
            grain.append(label)

    return len(crude), len(grain), len(total_len)


def cosine_similarity_topic_crude(array_to_work_with, len_crude):
    """Calculates the average cosine similarity of each vector of topic crude
    compared to every vector of the same topic, averaged over the entire topic"""
    cosine_similarity_result = []

    # nested loop for cosine similarity for the matrix
    for index in range(0,len_crude):
        # Gets the first vector to be used in the cosine similiarity calculations
        #print("- vector1 [" +str(index)+":"+str(index+1)+"]" )
        vector1 = array_to_work_with[index:index+1]
        for innerindex in range(0,len_crude):
            # Gets the second vector to be used
            #print("  *vector2 [" +str(innerindex)+":"+str(innerindex+1)+"]" )
            vector2 = array_to_work_with[innerindex:innerindex+1]
            # Computes the cosine similarity between vector 1 and vector 2
            value = cosine_similarity(vector1, vector2, dense_output=True)
            cosine_similarity_result.append(value[0][0])

    average_cosine_similarity_crude = sum(cosine_similarity_result) / len(cosine_similarity_result)
    print("The average cosine similarity of topic crude:")
    print(average_cosine_similarity_crude)
    return average_cosine_similarity_crude


def cosine_similarity_topic_grain(array_to_work_with, len_grain, total_len):
    """Calculates the average cosine similarity of each vector of topic grain
    compared to every vector of the same topic, averaged over the entire topic"""
    cosine_similarity_result = []

    # nested loop for cosine similarity for the matrix
    for index in range(len_grain, total_len):
        # Gets the first vector to be used in the cosine similiarity calculations
        #print("- vector1 [" +str(index)+":"+str(index+1)+"]" )
        vector1 = array_to_work_with[index:index+1]
        for innerindex in range(len_grain, total_len):
            # Gets the second vector to be used
            #print("  *vector2 [" +str(innerindex)+":"+str(innerindex+1)+"]" )
            vector2 = array_to_work_with[innerindex:innerindex+1]
            # Computes the cosine similarity between vector 1 and vector 2
            value = cosine_similarity(vector1, vector2, dense_output=True)
            cosine_similarity_result.append(value[0][0])

    average_cosine_similarity_grain = sum(cosine_similarity_result) / len(cosine_similarity_result)
    print("The average cosine similarity of topic grain:")
    print(average_cosine_similarity_grain)
    return average_cosine_similarity_grain



def cosine_similarity_crude_to_grain(array_to_work_with, len_crude, len_grain, total_len):
    """Calculate the average cosine similarity of each vector of a specific topic
    compared to every vector of the other topic (other folder), averaged over
    the entire topic"""
    cosine_similarity_result = []

    # nested loop for cosine similarity for the matrix
    for index in range(0,len_crude):
        # Gets the first vector to be used in the cosine similiarity calculations
        #print("- vector1 [" +str(index)+":"+str(index+1)+"]" )
        vector1 = array_to_work_with[index:index+1]
        for innerindex in range(len_grain, total_len):
            # Gets the second vector to be used
            #print("  *vector2 [" +str(innerindex)+":"+str(innerindex+1)+"]" )
            vector2 = array_to_work_with[innerindex:innerindex+1]
            # Computes the cosine similarity between vector 1 and vector 2
            value = cosine_similarity(vector1, vector2, dense_output=True)
            cosine_similarity_result.append(value[0][0])

    average_cosine_similarity_crude_to_grain = sum(cosine_similarity_result) / len(cosine_similarity_result)
    print("The average cosine similarity of topic crude compared to topic grain:")
    print(average_cosine_similarity_crude_to_grain)
    return average_cosine_similarity_crude_to_grain


def cosine_similarity_grain_to_crude(array_to_work_with, len_grain, len_crude, total_len):
    """Calculate the average cosine similarity of each vector of a specific topic
    compared to every vector of the other topic (other folder), averaged over
    the entire topic"""
    cosine_similarity_result = []

    # nested loop for cosine similarity for the matrix
    for index in range(len_grain, total_len):
        # Gets the first vector to be used in the cosine similiarity calculations
        #print("- vector1 [" +str(index)+":"+str(index+1)+"]" )
        vector1 = array_to_work_with[index:index+1]
        for innerindex in range(0,len_crude):
            # Gets the second vector to be used
            #print("  *vector2 [" +str(innerindex)+":"+str(innerindex+1)+"]" )
            vector2 = array_to_work_with[innerindex:innerindex+1]
            # Computes the cosine similarity between vector 1 and vector 2
            value = cosine_similarity(vector1, vector2, dense_output=True)
            cosine_similarity_result.append(value[0][0])

    average_cosine_similarity_grain_to_crude = sum(cosine_similarity_result) / len(cosine_similarity_result)
    print("The average cosine similarity of topic grain compared to topic crude:")
    print(average_cosine_similarity_grain_to_crude)
    return average_cosine_similarity_grain_to_crude



array_to_work_with = file_to_array(args.vectorfile)
len_crude, len_grain, total_len = get_range(args.vectorfile)
cosine_similarity_topic_crude(array_to_work_with, len_crude)
cosine_similarity_topic_grain(array_to_work_with, len_grain, total_len)
cosine_similarity_crude_to_grain(array_to_work_with, len_crude, len_grain, total_len)
cosine_similarity_grain_to_crude(array_to_work_with, len_crude, len_grain, total_len)

#Calculate the average cosine similarity of each vector of a specific topic
#compared to every vector of the same topic, averaged over the entire topic.
#Cosine similarity should be between 0 and 1 between two specific vectors.
#nested for loop.

#Calculate the average cosine similarity of each vector of a specific topic
#compared to every vector of the other topic (other folder), averaged over
#the entire topic.

#First vector cosine similarity to the second vector, first to third, first to
# fourth, etc. Then the average of all those values.
#All documents to each other in the same topic. Then all documents to each
#other in the other topic. Then all of topic 1 to all of topic 2 (and vice versa
# = same result?).

#Average = the cosine values of each comparison divided with the sum of all
#comparisons.
