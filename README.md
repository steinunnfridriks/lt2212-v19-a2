# LT2212 V19 Assignment 2

From Asad Sayeed's statistical NLP course at the University of Gothenburg.

My name: STEINUNN RUT FRIÐRIKSDÓTTIR

## Additional instructions

For some reason and after endless hours and an allnighter of trying to fix it
(after actually spending hours in trying to find out how to keep the column
labels to begin with, as they always disappeared), my gendoc.py generates files
that do contain the column labels (the words from the vocabulary), but also the
index of them (so 1,2,3,4...). This results in my files having around 12 thousand
columns with NaNs. I give up.  

## Results and discussion

### Vocabulary restriction.

I chose to limit my vocabulary to the top 200 most frequent words, because that's
way lower than the 12 thousand word vocabulary and should give an obvious
comparison.

### Result table

As I stated above, my results are not valid due to the NaN columns. When my file
still worked like it was supposed to (before realizing I don't have the same
version of pandas as the MLTGPU has), all my results were around 0.4.

### The hypothesis in your own words
I suppose the hypothesis of this experiment must be that when extracting word
counts from this amount of data, you really need to take into account the
frequency of words that have more meaning to the topic. For example, even though
the word "Sushi" might not be the most frequent one overall in a corpus, it
might have a lot of meaning for a subset of the corpus where the topic is
related to Japanese cuisine. By inversing the frequency of the documents,
lower-count, meaningful words can have more value in the computations.  
Then we would for example realize how much three recipes that all share the
ingredient "squid ink" have in common, as opposed to just realizing the irrelevant
fact that the ingredient "egg" is very common in recipes.

### Discussion of trends in results in light of the hypothesis
As I don't have my proper results, I'm not sure how to answer this question.
However, my earlier results showed very similar cosine similarities for every
test. Perhaps the topics weren't unrelated or variant enough to show the full
meaning of having TFIDF files.

## Bonus answers

The flaws in this experiment might be just what I pointed out above, that the
data isn't diverse enough to really generate some definitive results. By testing
data of various different topics, you might have more "shocking" results.  
