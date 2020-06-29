'''
@author: 190000096
Aiming to evaluate accuracy of a first-order HMM for POS tagging 
and determine whether use of UNK tags helps to improve accuracy.
'''
import nltk
from nltk.corpus import brown
from nltk import ConditionalFreqDist
from nltk import ConditionalProbDist
from nltk import MLEProbDist
from nltk import WittenBellProbDist
from nltk.probability import FreqDist
from texttable import Texttable
from nltk.metrics import ConfusionMatrix


# adding "START" and "END"
def add_start_end (start, end):    
    tags_words = []
    for sent in brown.tagged_sents(tagset="universal")[start:end]:
        tags_words.append(("START", "START"))
        tags_words.extend([(t, w) for (w, t) in sent])
        tags_words.append(("END", "END"))
    return tags_words


# smoothing transition probability
def smoothed_transition_probability (tag):
    return WittenBellProbDist(cfd_tags[tag], bins=1e5)   


# smoothing observation likelihood
def smoothed_observation_likelihood(tag):
    return WittenBellProbDist(cfd_tagwords[tag], bins=1e5)


# finding the tags using Viterbi
def viterbi_method (sentence, backpointer):
    backpointer = backpointer
    sentence = sentence
    viterbi_pre = {}
    viterbi_now = {}
    num_now = 0
    num_max_now = 0
    tag_max_now = ""
    
    # Initializing viterbi_now
    for tag in distinct_tags:
        num_now = smoothed_transition_probability("START").prob(tag) * smoothed_observation_likelihood(tag).prob(sentence[1])
        viterbi_now[tag] = num_now
        if num_now > num_max_now:
            num_max_now = num_now
            tag_max_now = tag
    backpointer.append("START")
    backpointer.append(tag_max_now)
    
    # iterating and finding the best sequence of tags for a sentence
    for l in range(2, len(sentence)):
        num_max_now = 0
        tag_max_now = ""
        viterbi_pre = viterbi_now
        viterbi_now = {}
        for tag in distinct_tags:
            num_pre_max = 0
            for tag_pre in distinct_tags:
                num_pre = viterbi_pre[tag_pre] * smoothed_transition_probability(tag_pre).prob(tag) * smoothed_observation_likelihood(tag).prob(sentence[l])
                if num_pre > num_pre_max:
                    num_pre_max = num_pre
            viterbi_now[tag] = num_pre_max
            if num_pre_max > num_max_now:
                num_max_now = num_pre_max
                tag_max_now = tag
        backpointer.append(tag_max_now)
       
    return backpointer


# find tags for couple sentences
def find_tag_for_sentences (sentences):
    backpointer = []
    words = []
    for l in range(len(sentences)):
        words.append(sentences[l])
        if sentences[l] == "END":
            backpointer = viterbi_method(words, backpointer)
            words = []
    return backpointer


# replace words capitalized which occur only once and not at the beginning
def replace_with_UNKCAP (tags_words):
    tags_words = tags_words
    fdist = FreqDist(words_train)
    hapaxes = fdist.hapaxes()
    for l in range(len(tags_words)):
        if tags_words[l][1] in hapaxes or tags_words[l][1] not in words_train:
            if tags_words[l][1][0].isupper() and tags_words[l - 1][1] != "START":
                tags_words[l] = list(tags_words[l])
                tags_words[l][1] = "UNK-CAP"
                tags_words[l] = tuple(tags_words[l])
    return tags_words


            
# replace words ending with "ed" which occur only once 
def replace_with_UNKED (tags_words):
    tags_words = tags_words
    fdist = FreqDist(words_train)
    hapaxes = fdist.hapaxes()
    for l in range(len(tags_words)):
        if tags_words[l][1] in hapaxes or tags_words[l][1] not in words_train:
            if tags_words[l][1].endswith("ed"):
                tags_words[l] = list(tags_words[l])
                tags_words[l][1] = "UNK-ED"
                tags_words[l] = tuple(tags_words[l])
    return tags_words


# replace words ending with "ing" which occur only once 
def replace_with_UNKING (tags_words):
    tags_words = tags_words
    fdist = FreqDist(words_train)
    hapaxes = fdist.hapaxes()
    for l in range(len(tags_words)):
        if tags_words[l][1] in hapaxes or tags_words[l][1] not in words_train:
            if tags_words[l][1].endswith("ing"):
                tags_words[l] = list(tags_words[l])
                tags_words[l][1] = "UNK-ING"
                tags_words[l] = tuple(tags_words[l])
    return tags_words
        

# calculating accuracy
def calculate_accuracy (tag, backpointer):
    right = 0
    wrong = 0
    for l in range(len(backpointer)):
        if backpointer[l] == tag[l]:
            right = right + 1
        else:
            wrong = wrong + 1
    return " {:.2%}".format(right / (right + wrong))


# train and test set
tags_words_train = add_start_end(0, 10000)
tags_words_test = add_start_end(10001, 10501)

'''
test and train without UNK tag
'''
words_train = ([w for(_, w) in tags_words_train])
words_test = ([w for(_, w) in tags_words_test])
tags_train = ([t for(t, _) in tags_words_train])
tags_test = ([t for(t, _) in tags_words_test])
distinct_tags = set(tags_train)
# calculating transition probability
cfd_tags = ConditionalFreqDist(nltk.bigrams(tags_train))
cpd_tags = ConditionalProbDist(cfd_tags, MLEProbDist)
# calculating observation likelihood
cfd_tagwords = ConditionalFreqDist(tags_words_train)
cpd_tagwords = ConditionalProbDist(cfd_tagwords, MLEProbDist)
backpointer = find_tag_for_sentences(words_test)
accuracy_without_UNK_tag = calculate_accuracy(tags_test, backpointer)
 
'''
test and train with UNK-ING tag
'''
tags_words_train = replace_with_UNKING(tags_words_train)
tags_words_test = replace_with_UNKING(tags_words_test)
words_train = ([w for(_, w) in tags_words_train])
words_test = ([w for(_, w) in tags_words_test])
tags_train = ([t for(t, _) in tags_words_train])
tags_test = ([t for(t, _) in tags_words_test])
distinct_tags = set(tags_train)
# calculating transition probability
cfd_tags = ConditionalFreqDist(nltk.bigrams(tags_train))
cpd_tags = ConditionalProbDist(cfd_tags, MLEProbDist)
# calculating observation likelihood
cfd_tagwords = ConditionalFreqDist(tags_words_train)
cpd_tagwords = ConditionalProbDist(cfd_tagwords, MLEProbDist)
backpointer = find_tag_for_sentences(words_test)
accuracy_with_UNKING_tag = calculate_accuracy(tags_test, backpointer)
 
'''
test and train with UNK-CAP tag
'''
tags_words_train = add_start_end(0, 10000)
tags_words_test = add_start_end(10001, 10501)
tags_words_train = replace_with_UNKCAP(tags_words_train)
tags_words_test = replace_with_UNKCAP(tags_words_test)
words_train = ([w for(_, w) in tags_words_train])
words_test = ([w for(_, w) in tags_words_test])
tags_train = ([t for(t, _) in tags_words_train])
tags_test = ([t for(t, _) in tags_words_test])
distinct_tags = set(tags_train)
# calculating transition probability
cfd_tags = ConditionalFreqDist(nltk.bigrams(tags_train))
cpd_tags = ConditionalProbDist(cfd_tags, MLEProbDist)
# calculating observation likelihood
cfd_tagwords = ConditionalFreqDist(tags_words_train)
cpd_tagwords = ConditionalProbDist(cfd_tagwords, MLEProbDist)
backpointer = find_tag_for_sentences(words_test)
accuracy_with_UNKCAP_tag = calculate_accuracy(tags_test, backpointer)
 
'''
test and train with UNK-ED tag
'''
tags_words_train = add_start_end(0, 10000)
tags_words_test = add_start_end(10001, 10501)
tags_words_train = replace_with_UNKED(tags_words_train)
tags_words_test = replace_with_UNKED(tags_words_test)
words_train = ([w for(_, w) in tags_words_train])
words_test = ([w for(_, w) in tags_words_test])
tags_train = ([t for(t, _) in tags_words_train])
tags_test = ([t for(t, _) in tags_words_test])
distinct_tags = set(tags_train)
# calculating transition probability
cfd_tags = ConditionalFreqDist(nltk.bigrams(tags_train))
cpd_tags = ConditionalProbDist(cfd_tags, MLEProbDist)
# calculating observation likelihood
cfd_tagwords = ConditionalFreqDist(tags_words_train)
cpd_tagwords = ConditionalProbDist(cfd_tagwords, MLEProbDist)
backpointer = find_tag_for_sentences(words_test)
accuracy_with_UNKED_tag = calculate_accuracy(tags_test, backpointer)
 
'''
test and train with UNK-ING tag and UNK-CAP tag
'''
tags_words_train = add_start_end(0, 10000)
tags_words_test = add_start_end(10001, 10501)
tags_words_train = replace_with_UNKING(tags_words_train)
tags_words_train = replace_with_UNKCAP(tags_words_train)
tags_words_test = replace_with_UNKING(tags_words_test)
tags_words_test = replace_with_UNKCAP(tags_words_test)
words_train = ([w for(_, w) in tags_words_train])
words_test = ([w for(_, w) in tags_words_test])
tags_train = ([t for(t, _) in tags_words_train])
tags_test = ([t for(t, _) in tags_words_test])
distinct_tags = set(tags_train)
# calculating transition probability
cfd_tags = ConditionalFreqDist(nltk.bigrams(tags_train))
cpd_tags = ConditionalProbDist(cfd_tags, MLEProbDist)
# calculating observation likelihood
cfd_tagwords = ConditionalFreqDist(tags_words_train)
cpd_tagwords = ConditionalProbDist(cfd_tagwords, MLEProbDist)
backpointer = find_tag_for_sentences(words_test)
accuracy_with_UNKING_UNKCAP_tag = calculate_accuracy(tags_test, backpointer)
 
'''
test and train with UNK-CAP tag and UNK-ING tag
'''
tags_words_train = add_start_end(0, 10000)
tags_words_test = add_start_end(10001, 10501)
tags_words_train = replace_with_UNKCAP(tags_words_train)
tags_words_train = replace_with_UNKING(tags_words_train)
tags_words_test = replace_with_UNKCAP(tags_words_test)
tags_words_test = replace_with_UNKING(tags_words_test)
words_train = ([w for(_, w) in tags_words_train])
words_test = ([w for(_, w) in tags_words_test])
tags_train = ([t for(t, _) in tags_words_train])
tags_test = ([t for(t, _) in tags_words_test])
distinct_tags = set(tags_train)
# calculating transition probability
cfd_tags = ConditionalFreqDist(nltk.bigrams(tags_train))
cpd_tags = ConditionalProbDist(cfd_tags, MLEProbDist)
# calculating observation likelihood
cfd_tagwords = ConditionalFreqDist(tags_words_train)
cpd_tagwords = ConditionalProbDist(cfd_tagwords, MLEProbDist)
backpointer = find_tag_for_sentences(words_test)
accuracy_with_UNKCAP_UNKING_tag = calculate_accuracy(tags_test, backpointer)
 
'''
test and train with UNK-ED tag and UNK-CAP tag
'''
tags_words_train = add_start_end(0, 10000)
tags_words_test = add_start_end(10001, 10501)
tags_words_train = replace_with_UNKED(tags_words_train)
tags_words_train = replace_with_UNKCAP(tags_words_train)
tags_words_test = replace_with_UNKED(tags_words_test)
tags_words_test = replace_with_UNKCAP(tags_words_test)
words_train = ([w for(_, w) in tags_words_train])
words_test = ([w for(_, w) in tags_words_test])
tags_train = ([t for(t, _) in tags_words_train])
tags_test = ([t for(t, _) in tags_words_test])
distinct_tags = set(tags_train)
# calculating transition probability
cfd_tags = ConditionalFreqDist(nltk.bigrams(tags_train))
cpd_tags = ConditionalProbDist(cfd_tags, MLEProbDist)
# calculating observation likelihood
cfd_tagwords = ConditionalFreqDist(tags_words_train)
cpd_tagwords = ConditionalProbDist(cfd_tagwords, MLEProbDist)
backpointer = find_tag_for_sentences(words_test)
accuracy_with_UNKED_UNKCAP_tag = calculate_accuracy(tags_test, backpointer)
'''
test and train with UNK-CAP tag and UNK-ED tag
'''
tags_words_train = add_start_end(0, 10000)
tags_words_test = add_start_end(10001, 10501)
tags_words_train = replace_with_UNKCAP(tags_words_train)
tags_words_train = replace_with_UNKED(tags_words_train)
tags_words_test = replace_with_UNKCAP(tags_words_test)
tags_words_test = replace_with_UNKED(tags_words_test)
words_train = ([w for(_, w) in tags_words_train])
words_test = ([w for(_, w) in tags_words_test])
tags_train = ([t for(t, _) in tags_words_train])
tags_test = ([t for(t, _) in tags_words_test])
distinct_tags = set(tags_train)
# calculating transition probability
cfd_tags = ConditionalFreqDist(nltk.bigrams(tags_train))
cpd_tags = ConditionalProbDist(cfd_tags, MLEProbDist)
# calculating observation likelihood
cfd_tagwords = ConditionalFreqDist(tags_words_train)
cpd_tagwords = ConditionalProbDist(cfd_tagwords, MLEProbDist)
backpointer = find_tag_for_sentences(words_test)
accuracy_with_UNKCAP_UNKED_tag = calculate_accuracy(tags_test, backpointer)
'''
test and train with UNK-ING tag and UNK-ED tag
'''
tags_words_train = add_start_end(0, 10000)
tags_words_test = add_start_end(10001, 10501)
tags_words_train = replace_with_UNKING(tags_words_train)
tags_words_train = replace_with_UNKED(tags_words_train)
tags_words_test = replace_with_UNKING(tags_words_test)
tags_words_test = replace_with_UNKED(tags_words_test)
words_train = ([w for(_, w) in tags_words_train])
words_test = ([w for(_, w) in tags_words_test])
tags_train = ([t for(t, _) in tags_words_train])
tags_test = ([t for(t, _) in tags_words_test])
distinct_tags = set(tags_train)
# calculating transition probability
cfd_tags = ConditionalFreqDist(nltk.bigrams(tags_train))
cpd_tags = ConditionalProbDist(cfd_tags, MLEProbDist)
# calculating observation likelihood
cfd_tagwords = ConditionalFreqDist(tags_words_train)
cpd_tagwords = ConditionalProbDist(cfd_tagwords, MLEProbDist)
backpointer = find_tag_for_sentences(words_test)
accuracy_with_UNKING_UNKED_tag = calculate_accuracy(tags_test, backpointer)
cm=ConfusionMatrix(tags_test,backpointer)

'''
test and train with UNK-CAP tag, UNK-ING tag and UNK-ED tag
'''
tags_words_train = add_start_end(0, 10000)
tags_words_test = add_start_end(10001, 10501)
tags_words_train = replace_with_UNKCAP(tags_words_train)
tags_words_train = replace_with_UNKING(tags_words_train)
tags_words_train = replace_with_UNKED(tags_words_train)
tags_words_test = replace_with_UNKCAP(tags_words_test)
tags_words_test = replace_with_UNKING(tags_words_test)
tags_words_test = replace_with_UNKED(tags_words_test)
words_train = ([w for(_, w) in tags_words_train])
words_test = ([w for(_, w) in tags_words_test])
tags_train = ([t for(t, _) in tags_words_train])
tags_test = ([t for(t, _) in tags_words_test])
distinct_tags = set(tags_train)
# calculating transition probability
cfd_tags = ConditionalFreqDist(nltk.bigrams(tags_train))
cpd_tags = ConditionalProbDist(cfd_tags, MLEProbDist)
# calculating observation likelihood
cfd_tagwords = ConditionalFreqDist(tags_words_train)
cpd_tagwords = ConditionalProbDist(cfd_tagwords, MLEProbDist)
backpointer = find_tag_for_sentences(words_test)
accuracy_with_UNKCAP_UNKING_UNKED_tag = calculate_accuracy(tags_test, backpointer)
 
'''
test and train with UNK-ING tag, UNK-CAP tag and UNK-ED tag
'''
tags_words_train = add_start_end(0, 10000)
tags_words_test = add_start_end(10001, 10501)
tags_words_train = replace_with_UNKING(tags_words_train)
tags_words_train = replace_with_UNKCAP(tags_words_train)
tags_words_train = replace_with_UNKED(tags_words_train)
tags_words_test = replace_with_UNKING(tags_words_test)
tags_words_test = replace_with_UNKCAP(tags_words_test)
tags_words_test = replace_with_UNKED(tags_words_test)
words_train = ([w for(_, w) in tags_words_train])
words_test = ([w for(_, w) in tags_words_test])
tags_train = ([t for(t, _) in tags_words_train])
tags_test = ([t for(t, _) in tags_words_test])
distinct_tags = set(tags_train)
# calculating transition probability
cfd_tags = ConditionalFreqDist(nltk.bigrams(tags_train))
cpd_tags = ConditionalProbDist(cfd_tags, MLEProbDist)
# calculating observation likelihood
cfd_tagwords = ConditionalFreqDist(tags_words_train)
cpd_tagwords = ConditionalProbDist(cfd_tagwords, MLEProbDist)
backpointer = find_tag_for_sentences(words_test)
accuracy_with_UNKING_UNKCAP_UNKED_tag = calculate_accuracy(tags_test, backpointer)
 
'''
test and train with UNK-ING tag, UNK-ED tag and UNK-CAP tag
'''
tags_words_train = add_start_end(0, 10000)
tags_words_test = add_start_end(10001, 10501)
tags_words_train = replace_with_UNKING(tags_words_train)
tags_words_train = replace_with_UNKED(tags_words_train)
tags_words_train = replace_with_UNKCAP(tags_words_train)
tags_words_test = replace_with_UNKING(tags_words_test)
tags_words_test = replace_with_UNKED(tags_words_test)
tags_words_test = replace_with_UNKCAP(tags_words_test)
words_train = ([w for(_, w) in tags_words_train])
words_test = ([w for(_, w) in tags_words_test])
tags_train = ([t for(t, _) in tags_words_train])
tags_test = ([t for(t, _) in tags_words_test])
distinct_tags = set(tags_train)
# calculating transition probability
cfd_tags = ConditionalFreqDist(nltk.bigrams(tags_train))
cpd_tags = ConditionalProbDist(cfd_tags, MLEProbDist)
# calculating observation likelihood
cfd_tagwords = ConditionalFreqDist(tags_words_train)
cpd_tagwords = ConditionalProbDist(cfd_tagwords, MLEProbDist)
backpointer = find_tag_for_sentences(words_test)
accuracy_with_UNKING_UNKED_UNKCAP_tag = calculate_accuracy(tags_test, backpointer)
 
# output result in table
t = Texttable()
t.add_rows([["UNK-TAG", "Accuracy"],
           ["Without UNK TAG", accuracy_without_UNK_tag],
           ["With UNK-ING", accuracy_with_UNKING_tag],
           ["With UNK-CAP", accuracy_with_UNKCAP_tag],
           ["With UNK-ED", accuracy_with_UNKED_tag],
           ["With UNK-ING and UNK-CAP", accuracy_with_UNKING_UNKCAP_tag],
           ["With UNK-CAP and UNK-ING", accuracy_with_UNKCAP_UNKING_tag],
           ["With UNK-ED and UNK-CAP", accuracy_with_UNKED_UNKCAP_tag],
           ["With UNK-CAP and UNK-ED", accuracy_with_UNKCAP_UNKED_tag],
           ["With UNK-ING and UNK-ED", accuracy_with_UNKING_UNKED_tag],
           ["With UNK-CAP, UNK-ING and UNK-ED", accuracy_with_UNKCAP_UNKING_UNKED_tag],
           ["With UNK-ING, UNK-CAP and UNK-ED", accuracy_with_UNKING_UNKCAP_UNKED_tag],
           ["With UNK-ING, UNK-ED and UNK-CAP", accuracy_with_UNKING_UNKED_UNKCAP_tag]])
 
print(t.draw())
print()
print("The confusion matrix of POS tagging (Accuracy 94.31%) : ")
print(cm)

