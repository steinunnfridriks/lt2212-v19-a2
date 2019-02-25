import os, sys
import glob
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
import re

# gendoc.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here

parser = argparse.ArgumentParser(description="Generate term-document matrix.")
parser.add_argument("-T", "--tfidf", action="store_true", help="Apply tf-idf to the matrix.")
parser.add_argument("-S", "--svd", metavar="N", dest="svddims", type=int,
                    default=None,
                    help="Use TruncatedSVD to truncate to N dimensions")
parser.add_argument("-B", "--base-vocab", metavar="M", dest="basedims",
                    type=int, default=None,
                    help="Use the top M dims from the raw counts before further processing")
parser.add_argument("foldername", type=str,
                    help="The base folder name containing the two topic subfolders.")
parser.add_argument("outputfile", type=str,
                    help="The name of the output file for the matrix data.")

args = parser.parse_args()

print("Loading data from directory {}.".format(args.foldername))

if not args.basedims:
    print("Using full vocabulary.")
else:
    print("Using only top {} terms by raw count.".format(args.basedims))

if args.tfidf:
    print("Applying tf-idf to raw counts.")

if args.svddims:
    print("Truncating matrix to {} dimensions via singular value decomposition.".format(args.svddims))

# THERE ARE SOME ERROR CONDITIONS YOU MAY HAVE TO HANDLE WITH CONTRADICTORY
# PARAMETERS.

print("Writing matrix to {}.".format(args.outputfile))



#___________________
#Create a vocabulary (making a list of every word that occurs in every document)
#Go through and count all the words in every document. Every vector should
#have every word, even if the vector doesn't contain that word (then it has the
#count of 0, but the word is still there).
#Label each document according to its topic because we need to keep track of
#where each document comes from later in the assignment, when we use every
#vector twice, once to compare with every vector from the same topic and once
#to compare to every vector from the other topic.
#Eliminate duplicate vectors - keep track of where they came from!

def vocabulary(directory, m=None):
    """Creates a vocabulary list which contains all words from all documents"""
    vocabulary_list = []
    for topic in os.listdir(directory):
        path_to_subdirectory = os.path.join(directory, topic)
        for file in os.listdir(path_to_subdirectory):
            path_to_file = os.path.join(path_to_subdirectory, file)
            with open(path_to_file, "r", encoding="utf8") as f:
                text = f.read()
                strip_punctuation = re.sub("!@#$%^&*()-=_+|;\'\:\,\"\.<>?\d", "", text).lower()
                get_words = strip_punctuation.split(" ")
                for word in get_words:
                    if word not in vocabulary_list:
                        vocabulary_list.append(word)
    return vocabulary_list


def preprocessing_and_labeling(directory, m=None):
    """Creates a top dictionary containing the topic + document names as keys and dictionaries as values.
    Those dictionaries contain words as keys and the word counts as values."""
    label_maker = {}
    for topic in os.listdir(directory):
        path_to_subdirectory = os.path.join(directory, topic)
        vocab = vocabulary(directory, m)
        vocab_dict = dict.fromkeys(vocab,0)

        for file in os.listdir(path_to_subdirectory):
            path_to_file = os.path.join(path_to_subdirectory, file)
            word_counts = vocab_dict.copy()

            with open(path_to_file, "r", encoding="utf8") as f:
                text = f.read()
                strip_punctuation = re.sub("!@#$%^&*()-=_+|;\'\:\,\"\.<>?\d", "", text).lower()
                get_words = strip_punctuation.split(" ")
                for word in get_words:
                    if word not in word_counts:
                        word_counts[word] = 1
                    else:
                        word_counts[word] += 1
            label_maker[topic+" "+file] = word_counts
    return label_maker


def vector_creator(directory, m=None):
    """Append every value (which is a word count) from the word_counts
    dictionary into lists which are the vectors. Every list should then be
    added into an array which is then converted into a dataframe from pandas"""
    supreme_dictionary = preprocessing_and_labeling(directory, m)

    for label in supreme_dictionary.keys():
        luke_i_am_your_vectorspace = []
        for word, count in supreme_dictionary[label].items():
            luke_i_am_your_vectorspace.append(count)
        supreme_dictionary[label] = luke_i_am_your_vectorspace

    return supreme_dictionary


def matrix_builder(directory, m=None):
    """Convert darth vader into an array, and the array into a dataframe"""
    darth_vader = vector_creator(directory, m)
    #matrix_array = np.array(darth_vader)
    column_names = vocabulary(directory, m)
    supreme_dictionary = preprocessing_and_labeling(directory, m)

    matrix_dataframe = pd.DataFrame.from_dict(darth_vader, orient='index', dtype=None, columns=column_names)

    print(matrix_dataframe)
    return matrix_dataframe


def cosine_similarity_same_topic():



def cosine_similarity_other_topic():



#import pdb;pdb.set_trace()
vocabulary(args.foldername, args.basedims)
preprocessing_and_labeling(args.foldername, args.basedims)
vector_creator(args.foldername, args.basedims)
matrix_builder(args.foldername, args.basedims)

#USE PANDAS. The vectors are brics. brics.index = each document.
#Can txt be converted into csv file? brics = pd.read_csv("path_to_file.csv", index_col = 0)
#pd.DataFrame(dictionary/brics) = to get the table
#brics[["word"]] = to get the column (specific word)
#brics[1:2] = slice, row number 1 (specific document)
#brics.loc[[document1]] = to select one (or more) vector. iloc = same except with index (not labels).

#Find a way to make each document into an array
#where the numbers are the word counts. For every word in vocabulary, create an
#array with the counts from a particular document.


#Create a matrix where every word in every document is counted, even if it has
#the count of 0. This is the vector space. The rows are the documents and the
#columns are the words. Each document is a vector containing the word counts
#so document 1 [1,2,3] document 2 [2,0,5] might be two vectors where the
#first number (column) is the count for the word "and", the second number is
#the count for the word "is", etc.





#create a dataframe from the dictionaries?
#OR convert the dictionaries into vectors and every vector goes into an array
#then make a dataframe from the array


#Calculate the average cosine similarity of each vector of a specific topic
#compared to every vector of the same topic, averaged over the entire topic.
#Cosine similarity should be between 0 and 1 between two specific vectors.
#nested for loop.


#Calculate the average cosine similarity of each vector of a specific topic
#compared to every vector of the other topic (other folder), averaged over
#the entire topic.

#def consine_similarity_other_topic():
