import re
from os import listdir
from struct import pack, unpack

from nltk.tokenize import sent_tokenize
from konlpy.tag import Komoran

from konlpy.tag import Komoran
import mail_extraction
import naver_extraction
import Dao_email
from math import log


class split:
    def split(self, doc):
        return doc.split()


class split_sound:
    def split(self, doc):
        doc = re.sub("\s", "", doc)
        result = []
        result.extend(doc)
        return result


class bigram:
    def split(self, doc):
        result = []
        tokens = doc.split()
        for i in range(len(tokens) - 1):
            result.append(' '.join(tokens[i:i + 2]))  # [i,i+2) 형태라는 것에 주의!
        return result


class bigram_sound:
    def split(self, doc):
        result = []
        tokens = split_sound().split(doc)
        for i in range(len(tokens) - 1):
            result.append(' '.join(tokens[i:i + 2]))
        return result


class morphs:

    def __init__(self):
        self.ma = Komoran()

    def split(self, doc):
        doc.strip()
        doc = re.sub("(?:\s)+", " ", doc)
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U0001F914"
                                   u"\U0001F9B8"
                                   "]+", flags=re.UNICODE)
        doc = emoji_pattern.sub("", doc)
        try:
            result = self.ma.morphs(doc)
        except UnicodeDecodeError:
            result = " "
        return result


class noun:
    def __init__(self):
        self.ma = Komoran()

    def split(self, doc):
        result = []
        for sentence in sent_tokenize(doc):
            sentence = sentence.strip()
            if len(sentence) > 1:
                try:
                    result.extend(self.ma.nouns(sentence))
                except UnicodeDecodeError:
                    result.extend("")
        return result


def word_extraction(doc, method):
    words = method.split(doc)
    return words


def format_maker(mail, label):
    return [i for i in zip(mail, label)]


def training(C, D, method):
    V = list()
    for row in D:
        if len(row[0][0].split()) != 0:
            for term in method.split(row[0][0]):
                V.append(term)
    V = list(set(V))
    N = len(D)

    prior = dict()

    globalcondprob = list()

    for i, c in enumerate(C):
        Dc = [d for d in D if d[-1] == c]
        Nc = len(Dc)

        prior[i] = Nc / N

        Tc = '\n'.join([str(d[0]) for d in Dc])

        Tct = dict()
        CondProb = dict()

        for t in V:
            Tct[t] = len([w for w in method.split(Tc) if w == t])
        for t in V:
            CondProb[t] = (Tct.get(t, 0) + 1) / (sum(Tct.values()) + len(Tct))

        globalcondprob.append(CondProb)

    return V, prior, globalcondprob


def testing(C, V, Prior, CondProb, d, method):
    W = list()
    for t in method.split(d):
        if t in V:
            W.append(t)
        score = list([0] * len(C))

        for i, _ in enumerate(C):
            score[i] = log(Prior[i])
            for k in W:
                score[i] += log(CondProb[i][k])
    return score


def making_list_value(b):
    result = list()
    for i in b:
        result.append(i[0][0])

    return result


def testing_all(q, w, e, b, method):
    result = list()

    d = making_list_value(b)
    target = [i[1] for i in b]

    for i, j in enumerate(d):
        if len(j.split()) != 0:
            output = testing(["True", "False"], q, w, e, j, method)
            if output[0] > output[1]:
                result.append(True)
            else:
                result.append(False)

    return result, target


def compare_result(predict, acutal):
    true_count = 0
    false_positive = 0
    false_negative = 0
    count = 0
    for i in range(len(predict)):
        if predict[i] == acutal[i]:
            true_count += 1
        elif predict[i] and not acutal[i]:
            false_positive += 1
        elif predict[i] and not acutal[i]:
            false_negative += 1

        count += 1
    precision = true_count / count
    accuracy = true_count / (true_count + false_positive)
    recall = true_count / (true_count + false_negative)
    f1 = 2 / ((1 / precision) + (1 / recall))

    print("precision:" + str(precision))
    print("accuracy:" + str(accuracy))
    print("Recall:" + str(recall))
    print("F1-score" + str(f1))

    return (precision, accuracy, recall, f1)