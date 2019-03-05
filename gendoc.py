import os, sys
import glob
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
import re

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
    if m is not None:
        vocabulary = vocabulary_list[:m]
    else:
        vocabulary = vocabulary_list

    return vocabulary


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
    added into an array which is later converted into a dataframe from pandas"""
    supreme_dictionary = preprocessing_and_labeling(directory, m)

    for label in supreme_dictionary.keys():
        luke_i_am_your_vectorspace = []
        for word, count in supreme_dictionary[label].items():
            luke_i_am_your_vectorspace.append(count)
        supreme_dictionary[label] = luke_i_am_your_vectorspace

    return supreme_dictionary


def matrix_builder(directory, m=None):
    """Convert the dictionary into a padas dataframe to be written into
    the output file. Dropping duplicate vectors."""
    darth_vader = vector_creator(directory, m)
    supreme_dictionary = preprocessing_and_labeling(directory, m)
    matrix_dataframe = pd.DataFrame.from_dict(darth_vader, orient='index')
    list_of_duplicates = matrix_dataframe[matrix_dataframe.duplicated()].index.tolist()
    matrix_dataframe = matrix_dataframe.drop_duplicates()
    print("These duplicated vectors have been dropped:")
    for duplicate in list_of_duplicates:
        print(duplicate)

    return matrix_dataframe


def make_tfidf(dataframe):
    """Turns the dataframe into tf-idf (term-frequency times inverse
    document-frequency) values after filtering vocabulary by setting -Bm"""
    tfidf_values = TfidfTransformer().fit_transform(dataframe) #transforms the dataframe into tfidf
    tfidf_data = tfidf_values.toarray()
    words = dataframe.keys()
    filenames = dataframe.index.values
    tfidf_data = pd.DataFrame(tfidf_data, columns=words, index=filenames)
    return dataframe


def make_svd(output_dataframe, N):
    """Turns the dataframe into into a document matrix with a feature space of
    dimensionality n. Singular value decomposition - used to exclude the least
    significant components of a vector"""
    svd = TruncatedSVD(N)
    svd_dataframe = svd.fit_transform(dataframe)

    return dataframe


def file_creator(dataframe, directory, m=None):
    """Creating the outputfile"""
    column_names = vocabulary(directory, m)
    dataframe = dataframe.to_csv(args.outputfile, index_label=column_names)
    return dataframe


#then modify README.md in Markdown to contain:

# Your name, in case that's not obvious from your github account.
# What you chose for the vocabulary restriction in (2) above, with a short justification.
# A table containing the output values from simdoc.py for each of the files (1)-(8), organized in a meaningful way.
# In your own words, write down what you think the hypothesis of this experiment was. (1 paragraph)
# A brief discussion of any trends you may see in the data, or lack thereof (possible), in light of the hypothesis you wrote down. (1 paragraph)

#____________________________________________________________
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

vocabulary(args.foldername, args.basedims)
preprocessing_and_labeling(args.foldername, args.basedims)
vector_creator(args.foldername, args.basedims)
dataframe = matrix_builder(args.foldername, args.basedims)
file_creator(dataframe, args.foldername, args.basedims)

if not args.basedims:
    print("Using full vocabulary.")
else:
    print("Using only top {} terms by raw count.".format(args.basedims))

if args.tfidf:
    print("Applying tf-idf to raw counts.")
    output_dataframe = make_tfidf(dataframe)

if args.svddims:
    print("Truncating matrix to {} dimensions via singular value decomposition.".format(args.svddims))
    if args.tfidf:
        # Selecting both TF-IDF and SVF
        output_dataframe = pd.DataFrame(output_dataframe)
        output_dataframe = make_svd(output_dataframe, args.svddims)
    else:
        output = make_svd(dataframe, args.svddims)

if args.basedims and args.svddims:
    if args.basedims <= args.svddims:
        print("Singular value decomposition dimentionality cannot be higher than the vocabulary size")
        exit(1)


print("Writing matrix to {}.".format(args.outputfile))
