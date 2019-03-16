import os, sys
import glob
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
import re
from nltk.tokenize import RegexpTokenizer
import nltk

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
                lowercase = text.lower()
                tokenizer = RegexpTokenizer(r'\w+')
                strip_punct = tokenizer.tokenize(lowercase)
                for word in strip_punct:
                    vocabulary_list.append(word)

    vocab_dict = dict(nltk.FreqDist(vocabulary_list))
    frequency = [(word,vocab_dict[word]) for word in sorted(vocab_dict, key=vocab_dict.get,reverse=True)]

    if m is not None:
        vocabulary = frequency[:m]
    else:
        vocabulary = frequency

    final_vocabulary = []
    for word, frequency in vocabulary:
        final_vocabulary.append(word)

    return final_vocabulary


def preprocessing_and_labeling(directory, m=None):
    """Creates a top dictionary containing the topic + document names as keys and dictionaries as values.
    Those dictionaries contain words as keys and the word counts as values."""
    label_maker = {}
    vocab = vocabulary(directory, m)
    vocab_dict = dict.fromkeys(vocab,0)

    for topic in os.listdir(directory):
        path_to_subdirectory = os.path.join(directory, topic)
        for file in os.listdir(path_to_subdirectory):
            path_to_file = os.path.join(path_to_subdirectory, file)
            word_counts = vocab_dict.copy()
            with open(path_to_file, "r", encoding="utf8") as f:
                text = f.read()
                lowercase = text.lower()
                tokenizer = RegexpTokenizer(r'\w+')
                strip_punct = tokenizer.tokenize(lowercase)
                for word in strip_punct:
                    if word in vocab:
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
    column_names = vocabulary(directory, m)
    matrix_dataframe = pd.DataFrame.from_dict(darth_vader, orient='index',  dtype=None, columns=column_names) #I know this doesn't run on the server but after almost a month of trying to work around it, it's the best I can come up with
    list_of_duplicates = matrix_dataframe[matrix_dataframe.duplicated()].index.tolist()
    matrix_dataframe = matrix_dataframe.drop_duplicates()
    print("These duplicated vectors have been dropped:")
    for duplicate in list_of_duplicates:
        print(duplicate)

    return matrix_dataframe


def make_svd(directory, output_dataframe, outputfile, N, m=None):
    """Turns the dataframe into into a document matrix with a feature space of
    dimensionality n. Singular value decomposition - used to exclude the least
    significant components of a vector"""
    darth_vader = vector_creator(directory, m)
    darth_vader_array = np.array(list(darth_vader.values()),dtype=float)
    words = dataframe.keys()
    filenames = dataframe.index.values
    svd = TruncatedSVD(N)
    svd_fit = svd.fit_transform(darth_vader_array)
    svd_dataframe = pd.DataFrame(svd_fit, index=filenames, columns=words)
    svd_dataframe.to_csv(outputfile, encoding="utf-8")

    return dataframe


def file_creator(dataframe, directory, m=None):
    """Creating the outputfile"""
    dataframe = dataframe.to_csv(args.outputfile)
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
    tfidf_values = TfidfTransformer().fit_transform(dataframe) #transforms the dataframe into tfidf
    tfidf_data = tfidf_values.toarray()
    words = dataframe.keys()
    filenames = dataframe.index.values
    tfidf_data = pd.DataFrame(tfidf_data, columns=words, index=filenames)
    tfidf_data.to_csv(args.outputfile, encoding="utf8")

if args.svddims:
    print("Truncating matrix to {} dimensions via singular value decomposition.".format(args.svddims))
    if args.tfidf:
        # Selecting both TF-IDF and SVF
        tfidf_values = TfidfTransformer().fit_transform(dataframe) #transforms the dataframe into tfidf
        tfidf_data = tfidf_values.toarray()
        words = dataframe.keys()
        filenames = dataframe.index.values
        tfidf_data = pd.DataFrame(tfidf_data, columns=words, index=filenames)
        svd = TruncatedSVD(args.svddims)
        svd_fit = svd.fit_transform(tfidf_data)
        svd_dataframe = pd.DataFrame(svd_fit, index=filenames)
        svd_dataframe.to_csv(args.outputfile, encoding="utf-8")
    else:
        darth_vader = vector_creator(args.foldername, args.basedims)
        row_labels = [x for x in darth_vader.keys()]
        darth_vader_array = np.array(list(darth_vader.values()),dtype=float)
        svd = TruncatedSVD(args.svddims)
        svd_fit = svd.fit_transform(darth_vader_array)
        svd_dataframe = pd.DataFrame(svd_fit, index=row_labels)
        svd_dataframe.to_csv(args.outputfile, encoding="utf-8")

if args.basedims and args.svddims:
    if args.basedims <= args.svddims:
        print("Singular value decomposition dimentionality cannot be higher than the vocabulary size")
        exit(1)


print("Writing matrix to {}.".format(args.outputfile))
