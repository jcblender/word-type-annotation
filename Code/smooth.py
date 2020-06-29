'''
@author: 190000096
Aiming to evaluate accuracy of a first-order HMM for POS tagging 
and determine whether use of Witten-Bell Smoothing helps to improve accuracy.
'''
import nltk
from nltk.corpus import brown
from nltk import ConditionalFreqDist
from nltk import ConditionalProbDist
from nltk import MLEProbDist
from nltk import WittenBellProbDist
from texttable import Texttable


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


def viterbi_method_without_smooth (sentence, backpointer):
    backpointer = backpointer
    sentence = sentence
    viterbi_pre = {}
    viterbi_now = {}
    num_now = 0
    num_max_now = 0
    tag_max_now = ""
    
    # Initializing viterbi_now
    for tag in distinct_tags:
        num_now = cpd_tags["START"].prob(tag) * cpd_tagwords[tag].prob(sentence[1])
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
                num_pre = viterbi_pre[tag_pre] * cpd_tags[tag_pre].prob(tag) * cpd_tagwords[tag].prob(sentence[l])
                if num_pre > num_pre_max:
                    num_pre_max = num_pre
            viterbi_now[tag] = num_pre_max
            if num_pre_max > num_max_now:
                num_max_now = num_pre_max
                tag_max_now = tag
        backpointer.append(tag_max_now)
       
    return backpointer

def viterbi_method_with_smooth_transition (sentence, backpointer):
    backpointer = backpointer
    sentence = sentence
    viterbi_pre = {}
    viterbi_now = {}
    num_now = 0
    num_max_now = 0
    tag_max_now = ""
    
    # Initializing viterbi_now
    for tag in distinct_tags:
        num_now = smoothed_transition_probability("START").prob(tag) * cpd_tagwords[tag].prob(sentence[1])
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
                num_pre = viterbi_pre[tag_pre] * smoothed_transition_probability(tag_pre).prob(tag) * cpd_tagwords[tag].prob(sentence[l])
                if num_pre > num_pre_max:
                    num_pre_max = num_pre
            viterbi_now[tag] = num_pre_max
            if num_pre_max > num_max_now:
                num_max_now = num_pre_max
                tag_max_now = tag
        backpointer.append(tag_max_now)
       
    return backpointer

def viterbi_method_with_smooth_observation (sentence, backpointer):
    backpointer = backpointer
    sentence = sentence
    viterbi_pre = {}
    viterbi_now = {}
    num_now = 0
    num_max_now = 0
    tag_max_now = ""
    
    # Initializing viterbi_now
    for tag in distinct_tags:
        num_now = cpd_tags["START"].prob(tag) * smoothed_observation_likelihood(tag).prob(sentence[1])
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
                num_pre = viterbi_pre[tag_pre] * cpd_tags[tag_pre].prob(tag) * smoothed_observation_likelihood(tag).prob(sentence[l])
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

def find_tag_for_sentences_without_smooth (sentences):
    backpointer = []
    words = []
    for l in range(len(sentences)):
        words.append(sentences[l])
        if sentences[l] == "END":
            backpointer = viterbi_method_without_smooth(words, backpointer)
            words = []
    return backpointer

def find_tag_for_sentences_with_smooth_transition (sentences):
    backpointer = []
    words = []
    for l in range(len(sentences)):
        words.append(sentences[l])
        if sentences[l] == "END":
            backpointer = viterbi_method_with_smooth_transition(words, backpointer)
            words = []
    return backpointer

def find_tag_for_sentences_with_smooth_observation (sentences):
    backpointer = []
    words = []
    for l in range(len(sentences)):
        words.append(sentences[l])
        if sentences[l] == "END":
            backpointer = viterbi_method_with_smooth_observation(words, backpointer)
            words = []
    return backpointer



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

'''
test and train with smoothed transition probability and observation likelihood
'''
backpointer = find_tag_for_sentences(words_test)
accuracy_with_all_smooth = calculate_accuracy(tags_test, backpointer)



'''
test and train with smoothed transition probability
'''

backpointer = find_tag_for_sentences_with_smooth_transition(words_test)
accuracy_with_smooth_transition = calculate_accuracy(tags_test, backpointer)

'''
test and train with smoothed observation likelihood
'''

backpointer = find_tag_for_sentences_with_smooth_observation(words_test)
accuracy_with_smooth_observation = calculate_accuracy(tags_test, backpointer)


'''
test and train without smooth
'''

backpointer = find_tag_for_sentences_without_smooth(words_test)
accuracy_without_smooth = calculate_accuracy(tags_test, backpointer)

# output result in table
t = Texttable()
t.add_rows([["Smooth or Not", "Accuracy"],
           ["smoothed transition probability and observation likelihood", accuracy_with_all_smooth],
           ["smoothed transition probability", accuracy_with_smooth_transition],
           ["smoothed observation likelihood", accuracy_with_smooth_observation],
           ["without smooth", accuracy_without_smooth]])

print(t.draw())

