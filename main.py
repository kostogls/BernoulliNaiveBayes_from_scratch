import nltk
import math

nltk.download('punkt')
from nltk import punkt


class NB():

    def __init__(self, binary=False, bernoulli=False):
        self.priors = []
        self.probs = []
        self.vocabulary = []
        self.lengths = []
        self.num_classes = 0
        self.binary = binary
        self.bernoulli = bernoulli

    def fit(self, docs):
        self.num_classes = len(docs)
        all_docs = 0
        for class_docs in docs:
            all_docs += len(class_docs)

        # number of all documents
        # print("all docs : ", all_docs)
        for class_docs in docs:
            self.priors.append(len(class_docs) / all_docs)
        # initial probability for each class
        # print("priors: ", self.priors)
        counts = []
        no_ofdocs_inclass = []
        # class docs are two lists, the first one containing 1st class' docs, and the second the 2nd's docs
        for class_docs in docs:
            # print("class docs: ", class_docs)
            no_ofdocs_inclass.append(len(class_docs))

            count = dict()
            s = 0
            for doc in class_docs:
                tokens = nltk.word_tokenize(doc)
                if self.binary or self.bernoulli == True:
                    # if in binary or bernoulli version, just remove tuples if there are in a single document
                    tokens = list(dict.fromkeys(tokens))
                s = s + len(tokens)
                for token in tokens:
                    if token in count:
                        count[token] += 1
                    else:
                        count[token] = 1
                    if not token in self.vocabulary:
                        self.vocabulary.append(token)
            self.lengths.append(s)
            counts.append(count)
            # in bernoulli version, in counts we need terms that don't exist in the class
            if self.bernoulli:
                for token in counts[0]:
                    if token not in counts[-1]:
                        counts[-1][token] = 0
                for token in counts[-1]:
                    if token not in counts[0]:
                        counts[0][token] = 0

        # Number of occurancies of tokens, for the 1st class, and for the 2nd class
        # print("length of vocabulary: ", len(self.vocabulary))
        # print("counts: ", counts)
        # print("vocabulary: ", self.vocabulary)
        # print("no: ", no_ofdocs_inclass)
        for i in (0, self.num_classes - 1):
            prob = dict()
            if self.bernoulli == False:
                for token in counts[i]:
                    prob[token] = (counts[i][token] + 1) / (self.lengths[i] + len(self.vocabulary))
                self.probs.append(prob)
            else:
                for token in counts[i]:
                    prob[token] = (counts[i][token] + 1) / (no_ofdocs_inclass[i] + 2)
                self.probs.append(prob)
        # probability of each token, based on first class, and then based on second class
        # print("likelihood: ", self.probs)

    def predict_proba(self, doc):
        tokens = nltk.word_tokenize(doc)
        scores = []
        bd = []
        sum = 0
        logs1 = []
        logs2 = []
        # in bernoulli version i create 2 lists of the log likelihood probability, one list for each class
        if self.bernoulli:
            for token in self.probs[0].values():
                logs1.append(token)
            for token in self.probs[1].values():
                logs2.append(token)
            for token in self.vocabulary:
                if token in tokens:
                    bd.append(1)
                else:
                    bd.append(0)
            # putting the probabilities in same order for the 2 classes
            for i in range(8):
                logs2 = logs2[1:] + [logs2[0]]

        for i in (0, self.num_classes - 1):

            if self.bernoulli == False:
                score = 0
                score += math.log(self.priors[i])
                for token in tokens:
                    if token in self.probs[i]:
                        score += math.log(self.probs[i].get(token))
                    else:
                        if token in self.vocabulary:
                            score += math.log(1 / (self.lengths[i] + len(self.vocabulary)))
                sum += math.exp(score)
                scores.append(math.exp(score))
            else:
                # for bernoulli the probabilites are predicted below
                score = 0
                score += math.log(self.priors[i])
                for token in range(len(self.vocabulary)):
                    if i == 0:
                        score += math.log(math.pow(logs1[token], bd[token]) * math.pow(1 - logs1[token], 1 - bd[token]))

                    elif i == 1:
                        score += math.log(math.pow(logs2[token], bd[token]) * math.pow(1 - logs2[token], 1 - bd[token]))
                sum += math.exp(score)
                scores.append(math.exp(score))

        for i in (0, self.num_classes - 1):
            scores[i] = scores[i] / sum

        return scores


def main():
    docs = []
    docs.append(['just plain boring', 'entirely predictable and lacks energy', 'no surprises and very few laugs'])
    docs.append(['very powerful', 'the most fun film of the summer'])

    test_doc = 'the film was predictable with no fun'
    mnb = NB(bernoulli=True)
    mnb.fit(docs)
    print(mnb.predict_proba(test_doc))


if __name__ == '__main__':
    main()
