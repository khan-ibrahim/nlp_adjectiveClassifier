#!/usr/bin/python3

#!/usr/bin/python3
# CSE354 Sp20; Assignment 1 Template v02
##################################################################
_version_ = 0.2

import sys

##################################################################
#1. Tokenizer

import re #python's regular expression package

def tokenize(sent):
    #input: a single sentence as a string.
    #output: a list of each "word" in the text
    # must use regular expressions

    tokens = []
    pattern = r'''(?:[A-Z]\.)+|[\.?!;]|[-@#'’1-9A-Za-z]+'''
    ## TODO: fix tokenizer to account for abbreviations ie. S.B.U.
    tokens = re.findall(pattern, sent)

    return tokens


##################################################################
#2. Pig Latinizer

v = {'A', 'a', 'E', 'e', 'I', 'i', 'O', 'o', 'U', 'u'}
c = {'B', 'b', 'C', 'c', 'D', 'd', 'F', 'f', 'G', 'g', 'H', 'h', 'J', 'j', \
'K', 'k', 'L', 'l', 'M', 'm', 'N', 'n', 'P', 'p', 'Q', 'q', 'R', 'r', 'S', \
's', 'T', 't', 'V', 'v', 'W', 'w', 'X', 'x', 'Y', 'y', 'Z', 'z'}

#if starts with consonant - append initial consecutive consonants to end
#append 'ay'
def pigLatinizeCToken(token):
    for char in token:
        if not char in c:
            return token[token.find(char):] + \
            token[:token.find(char)] + 'ay'
    return token + 'ay'

def pigLatinizer(tokens):
    #input: tokens: a list of tokens,
    #output: plTokens: tokens after transforming to pig latin

    plTokens = []

    for token in tokens:
        if len(token) < 1 or not token.isalpha():
            plTokens.append(token)

        #if starts with vowel - append 'way'
        elif token[0] in v:
            plTokens.append(token + 'way')

        elif token[0] in c:
            plTokens.append(pigLatinizeCToken(token))

        else:
            plTokens.append(token)

    return plTokens


##################################################################
#3. Feature Extractor

import numpy as np

def getNumVowelsAndNumConsonants(token):
    numV = numC = 0
    for char in token:
        if char in v:
            numV+=1
        elif char in c:
            numC+=1
    return(numV, numC)

def getFeaturesForTokens(tokens, wordToIndex):
    #input: tokens: a list of tokens,
    #wordToIndex: dict mapping 'word' to an index in the feature list.
    #output: list of lists (or np.array) of k feature values for the given target

    num_words = len(tokens)
    featuresPerTarget = list() #holds arrays of feature per word
    for targetI in range(num_words):
        currentToken = tokens[targetI].lower()
        #print('current:{}'.format(currentToken))
        #print('LEN ' + str(len(wordToIndex)))

        #count num vowels and consonants
        numV, numC = getNumVowelsAndNumConsonants(currentToken)

        currentOneHot = np.zeros(len(wordToIndex))
        prevOneHot = np.zeros(len(wordToIndex))
        nextOneHot = np.zeros(len(wordToIndex))

        #Produce one-hot encodings of current word
        if currentToken in wordToIndex:
            currentOneHot[wordToIndex[currentToken]] == 1
        else:
            print('''Current token:'{}' not found'''.format(currentToken))
            currentIndex = -1

        #Produce one-hot encodings of previous and next word
        prevI = targetI - 1
        nextI = targetI + 1

        if prevI >= 0:
            prevToken = tokens[prevI].lower()
            if prevToken in wordToIndex:
                prevOneHot[wordToIndex[prevToken]] = 1
            else:
                print('''Previous token:'{}' not found'''.format(prevToken))
                prevousIndex = -1
        else:
            previousIndex = -2

        if nextI < num_words:
            nextToken = tokens[nextI].lower()
            if nextToken in wordToIndex:
                nextOneHot[wordToIndex[nextToken]] = 1
            else:
                print('''Next token:'{}' not found'''.format(nextToken))
                nextIndex = -1
        else:
            nextIndex = -2

        #combine features of current target into one long flat vector;
        #and add target feature vector to list
        featuresPerTarget.append(np.concatenate(([numV], [numC], currentOneHot, \
        prevOneHot, nextOneHot)))

    return featuresPerTarget #a (num_words x k) matrix


##################################################################
#4. Adjective Classifier

from sklearn.linear_model import LogisticRegression

def trainAdjectiveClassifier(features, adjs):
    #inputs: features: feature vectors (i.e. X)
    #        adjs: whether adjective or not: [0, 1] (i.e. y)
    #output: model -- a trained sklearn.linear_model.LogisticRegression object

    model = LogisticRegression().fit(features, adjs)
    #<FILL IN>

    return model


##################################################################
##################################################################
## Main and provided complete methods
## Do not edit.
## If necessary, write your own main, but then make sure to replace
## and test with this before you submit.
##
## Note: Tests below will be a subset of those used to test your
##       code for grading.

def getConllTags(filename):
    #input: filename for a conll style parts of speech tagged file
    #output: a list of list of tuples
    #        representing [[[word1, tag1], [word2, tag2]]]
    wordTagsPerSent = [[]]
    sentNum = 0
    with open(filename, encoding='utf8') as f:
        for wordtag in f:
            wordtag=wordtag.strip()
            if wordtag:#still reading current sentence
                (word, tag) = wordtag.split("\t")
                wordTagsPerSent[sentNum].append((word,tag))
            else:#new sentence
                wordTagsPerSent.append([])
                sentNum+=1
    return wordTagsPerSent

# Main
if __name__== '__main__':
    print("Initiating test. Version " , _version_)
    #Data for 1 and 2
    testSents = ['I am attending NLP class 2 days a week at S.B.U. this Spring.',
                 "I don't think data-driven computational linguistics is very tough.",
                 '@mybuddy and the drill begins again. #SemStart']

    #1. Test Tokenizer:
    print("\n[ Tokenizer Test ]\n")
    tokenizedSents = []
    for s in testSents:
        tokenizedS = tokenize(s)
        print(s, tokenizedS, "\n")
        tokenizedSents.append(tokenizedS)

    #2. Test Pig Latinizer:
    print("\n[ Pig Latin Test ]\n")
    for ts in tokenizedSents:
        print(ts, pigLatinizer(ts), "\n")

    #load data for 3 and 4 the adjective classifier data:
    taggedSents = getConllTags('daily547.conll')

    #3. Test Feature Extraction:
    print("\n[ Feature Extraction Test ]\n")
    #first make word to index mapping:
    wordToIndex = set() #maps words to an index
    for sent in taggedSents:
        if sent:
            words, tags = zip(*sent) #splits [(w, t), (w, t)] into [w, w], [t, t]
            wordToIndex |= set([w.lower() for w in words]) #union of the words into the set
    print("  [Read ", len(taggedSents), " Sentences]")
    #turn set into dictionary: word: index
    wordToIndex = {w: i for i, w in enumerate(wordToIndex)}

    #Next, call Feature extraction per sentence
    sentXs = []
    sentYs = []
    print("  [Extracting Features]")
    for sent in taggedSents:
        if sent:
            words, tags = zip(*sent)
            sentXs.append(getFeaturesForTokens(words, wordToIndex))
            sentYs.append([1 if t == 'A' else 0 for t in tags])
    #test sentences
    print("\n", taggedSents[5], "\n", sentXs[5], "\n")
    print(taggedSents[192], "\n", sentXs[192], "\n")


    #4. Test Classifier Model Building
    print("\n[ Classifier Test ]\n")
    #setup train/test:
    from sklearn.model_selection import train_test_split
    #flatten by word rather than sent:
    X = [j for i in sentXs for j in i]
    y= [j for i in sentYs for j in i]
    try:
        X_train, X_test, y_train, y_test = train_test_split(np.array(X),
                                                            np.array(y),
                                                            test_size=0.20,
                                                            random_state=42)
    except ValueError:
        print("\nLooks like you haven't implemented feature extraction yet.")
        print("[Ending test early]")
        sys.exit(1)
    print("  [Broke into training/test. X_train is ", X_train.shape, "]")
    #Train the model.
    print("  [Training the model]")
    tagger = trainAdjectiveClassifier(X_train, y_train)
    print("  [Done]")


    #Test the tagger.
    from sklearn.metrics import classification_report
    #get predictions:
    y_pred = tagger.predict(X_test)
    #compute accuracy:
    leny = len(y_test)
    print("test n: ", leny)
    acc = np.sum([1 if (y_pred[i] == y_test[i]) else 0 for i in range(leny)]) / leny
    print("Accuracy: %.4f" % acc)
    #print(classification_report(y_test, y_pred, ['not_adj', 'adjective']))
