'''
@author: 190000096
Aiming to evaluate accuracy of a first-order HMM for POS tagging 
and determine whether use of UNK tags helps to improve accuracy for Portuguese.
'''
from nltk.corpus import floresta
import nltk
from nltk import ConditionalFreqDist
from nltk import ConditionalProbDist
from nltk import MLEProbDist
from nltk import WittenBellProbDist
from nltk.probability import FreqDist
from texttable import Texttable


# simplify floresta tag
def simplify_tag(t):
    if "+" in t :
        return t[t.index("+")+1:]
    else:
        return t

# adding "START" and "END"
def add_start_end (start, end):    
    tags_words = []
    tsents=floresta.tagged_sents()
    for sent in tsents[start:end]:
        tags_words.append(("START", "START"))
        tags_words.extend([(simplify_tag(t),w) for (w, t) in sent])
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

# replace words ending with "ar" which occur only once 
def replace_with_UNKAR (tags_words):
    tags_words = tags_words
    fdist = FreqDist(words_train)
    hapaxes = fdist.hapaxes()
    for l in range(len(tags_words)):
        if tags_words[l][1] in hapaxes or tags_words[l][1] not in words_train:
            if tags_words[l][1].endswith("ar") and tags_words[l][1][0].islower():
                tags_words[l] = list(tags_words[l])
                tags_words[l][1] = "UNK-AR"
                tags_words[l] = tuple(tags_words[l])
    return tags_words

'''
test and train without UNK tag
'''
tags_words_train = add_start_end(0, 8700)
tags_words_test = add_start_end(8701, 9201)
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
test and train with UNK-CAP tag
'''
tags_words_train = add_start_end(0, 8700)
tags_words_test = add_start_end(8701, 9201)
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
test and train with UNK-AR tag
'''
tags_words_train = add_start_end(0, 8700)
tags_words_test = add_start_end(8701, 9201)
tags_words_train = replace_with_UNKAR(tags_words_train)
tags_words_test = replace_with_UNKAR(tags_words_test)
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
accuracy_with_UNKAR_tag = calculate_accuracy(tags_test, backpointer)


'''
test and train with UNK-AR tag and UNK-CAP tag
'''
tags_words_train = add_start_end(0, 8700)
tags_words_test = add_start_end(8701, 9201)
tags_words_train = replace_with_UNKAR(tags_words_train)
tags_words_train = replace_with_UNKCAP(tags_words_train)
tags_words_test = replace_with_UNKAR(tags_words_test)
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
accuracy_with_UNKAR_UNKCAP_tag = calculate_accuracy(tags_test, backpointer)


# output result in table
t = Texttable()
t.add_rows([["UNK-TAG", "Accuracy for Portuguese"],
           ["Without UNK TAG", accuracy_without_UNK_tag],
           ["With UNK-CAP", accuracy_with_UNKCAP_tag],
           ["With UNK-AR", accuracy_with_UNKAR_tag],
           ["With UNK-AR, UNK-CAP", accuracy_with_UNKAR_UNKCAP_tag]])
 
print(t.draw())

