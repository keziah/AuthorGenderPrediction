
#FeatureTestingClass.py

#Python file containing code to predict author gender from context-free Yelp review text.

#Authors:
    ## Keziah Plattner, SUNetID = keziah
    ## Lilith Wu, SUNetID = lilithwu
    ## Laura Hunter, SUNetID = lmhunter


import scipy.sparse as sps
import numpy as np
from scipy.sparse import *
from scipy import *
from sklearn.naive_bayes import BernoulliNB
import sqlite3
import sklearn.feature_extraction as feature_extraction
from sklearn.cross_validation import train_test_split
from sklearn import svm
from collections import Counter
from sklearn.cluster import KMeans
from sklearn import metrics
import heapq
import Twokenize
import sys

class FeatureTesting():
    #Initialize an object of the FeatureTesting class
    def __init__(self):
        self.db = sqlite3.connect('/afs/ir.stanford.edu/users/k/e/keziah/cs/cs221/project/mini_yelp_db.db')
        self.c = self.db.cursor()
        self.genderCursor = self.db.cursor()
        self.c2 = self.db.cursor()
        self.PROBABILITY_CONSTANT = 0.90
        self.COUNT_CONSTANT = 20
        self.selectQuery = "SELECT Name, ReviewText, ReviewID from Reviews r, Users u where r.UserID = u.UserID"
        self.femaleLexicon = ["she", "her", "hers", "herself", "gal", "girl", "female", "chick", "queen", "bitch", "woman", "women"]
        self.maleLexicon = ["he", "him", "his", "himself", "dude", "boy", "male", "king", "dick", "men", "man"]
        self.genderConverter = {"female":0, "male":1}
        #Features assume binary status:
        self.isBool = False
        #Normalize to numWords:
        self.normalize = False
        self.vectorizer = None
        self.extraParamsQuery = None
        self.numReviewsCap = float("inf")


###################### Initialize FTC Settings ##########################################

    ##Method: setProb
    # Sets the class instance's PROBABILITY CONSTANT to represent how strict we want
    # to be in picking names/gender? (i.e. we only want names that have a 90% chance
    # of being a certain gender.
    def setProb(self, prob):
        self.PROBABILITY_CONSTANT = prob

    ## Method: setCount
    # Just like the method above, setCount works via modifying parameters that determine when
    # a name is/is not selected to classify based on. Typicaly we only want names that have at
    # least 20 examples.
    def setCount(self, count):
        self.COUNT_CONSTANT = count

    ##Method: setFeaturesBinaryStatus
    #For cases when we desire to only look the presence/absence of a feature, enabling the self
    #variable isBool can come in very handy.
    def setFeaturesBinaryStatus(self, isBool):
        self.isBool = isBool

    ## Method: setLimitOnNumReviews
    # Primarily developed as a debugging, the limit set is helpful for when you wish to set
    # an arbitrary cutoff point on number of reviews you will analyze and score.
    def setLimitOnNumReviews(self, numReviews):
        self.numReviewsCap = numReviews
        print "Arbitrarily selecting first %d reviews" %(numReviews)

    ##Method: setClusterFile
    #Necessary in any run where you wish to take advantage of the generated text clusters.
    def setClusterFile(self, filename):
        self.clusterFilename = filename


########################## Capture SQL "SELECT" Statements ###############################

    #adding other features to the vector! Just write a select statement that finds features by reviewID
    def extraFeaturesSelectStatement(self, statement):
        self.extraParamsQuery = statement

    #needs to be in form: SELECT Name, ReviewText, ReviewID from Reviews r, Users u where r.UserID = u.UserID AND <insert modifications here>
    #useful by filtering by location or whatever
    def setSelectStatement(self, statement):
        self.selectQuery = statement

############################ Choose Classifier Type ######################################

    def setNuSVM(self, passedKernel='rbf'):
        self.classifier = svm.NuSVC(kernel=passedKernel)
        print "Using NuSVC with default settings and kernel = %s" %(passedKernel)

    def setStandardSVC(self, passedKernel='rbf'):
        self.classifier = svm.SVC(kernel=passedKernel)
        print "Using standard scikit SVM with default settings and kernal = %s" %(passedKernel)

    def setPolySVM(self, polDegree=3, C_val=.5):
        self.classifier = svm.SVC(C=C_val, kernel='poly', degree=polDegree, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)
        print "Using Polynomial SVM with degree %d" %(polDegree)

    def setLinearSVM(self, C_val=1, dual_val=True, class_weight_val=False):
        if class_weight_val:
            self.classifier = svm.LinearSVC(C=C_val, dual=dual_val, class_weight=class_weight_val)
        else:
            self.classifier = svm.LinearSVC(C=C_val, dual=dual_val)
        print "Using Linear SVM with C=%.2f and dual=%r." % (C_val, dual_val)

    def setLinearSVML1(self, C_val=1, dual_val=True, class_weight_val=False):
        if class_weight_val:
            self.classifier = svm.LinearSVC(C=C_val, dual=dual_val, penalty="l1", class_weight=class_weight_val)
        else:
            self.classifier = svm.LinearSVC(C=C_val, dual=dual_val, penalty="l1")
        print "Using LinearSVC with C=%.2f and dual=%r, penalty = L1." % (C_val, dual_val)

    def setPersonalizedClassifier(self, classifier):
        self.classifier = classifier

#################### Creating Feature vectors (bag of words and extracted features) ######################

    ##Method: bagOfWords
    #Our simplest way of vectorizing and classifying based on the words present in the model.
    #The parameters min_n and max_n refer to the desired N-gram for use in classification.
    #Precondition: self.X and self.Y must both be accessible.
    #Postcondition: You will have a feature vector ready for input to a classifier for the
    #classification step.
    def bagOfWords(self, min_n, max_n):
        print "Starting Bag of Words, no stop words, with ngram model (%d, %d)..." % (min_n, max_n)
        reviews = []
        self.y = []
        iter = 0
        for row in self.c.execute(self.selectQuery):
            iter += 1
            gender, prob, count = self.getGenderFromName(row)
            if gender:
                reviews.append(row[1])
                self.y.append(self.genderConverter[gender])
        self.vectorizer = feature_extraction.text.TfidfVectorizer(stop_words='english', smooth_idf=True, use_idf=True, sublinear_tf=True, ngram_range=(min_n, max_n), binary=self.isBool)
        self.X = self.vectorizer.fit_transform(reviews)
        self.holdOutTestSet()

    ##
    #Method: translatePunctuation
    #Translate each character in review such that all letters are removed.
    #Input: non-tokenized review
    #Output: a whitespace-heavy review which can then be split to recover
    #a review exclusively comprised of punctuation/extra-alphabetic information
    ##
    def translatePunctuation(self, to_translate, translate_to=u' '):
        all_letters = u'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        translate_table = dict((ord(char), translate_to) for char in all_letters)
        return to_translate.translate(translate_table)

    ##
    #Method: bagOfPunctuationWords
    #Used as a first pass to identify frequently used punctuation strings assigned weight during classification.
    #Primarily as a refinement tool for further feature extraction.
    #Returns: None; sets self.X to be a vector containing extracted punctuation features
    ##
    def bagOfPunctuationWords(self, min_n, max_n):
        print "Starting Bag of Punctuation Words, no stop words, with ngram model (%d, %d)..." % (min_n, max_n)
        reviews = []
        self.y = []
        for row in self.c.execute(self.selectQuery):
            gender, prob, count = self.getGenderFromName(row)
            if gender:
                punctWords = self.translatePunctuation(row[1]).split()
                punctCounter = Counter(punctWords)
                reviews.append(dict(punctCounter))
                self.y.append(self.genderConverter[gender])
        datatrans = feature_extraction.DictVectorizer()
        self.X = datatrans.fit_transform(reviews)
        self.holdOutTestSet()


    #Function: addExtraParameters
    #This method enables you to add additional features from the database to use in classification.
    #Precondition: you must have + pass a "selectQuery"
    #Returns: a dictionary mapping (feature --> feature value)
    def addExtraParameters(self, selectQuery, ReviewID, isBool):
        self.c2.execute(selectQuery, [ReviewID])
        row = self.c2.fetchone()
        dict = {}
        for idx, col in enumerate(self.c2.description):
            if isBool:
                boolVersion = 1 if row[idx] > 0 else 0
                dict[col[0]] = boolVersion
            else:
                dict[col[0]] = row[idx]
        return dict

    #Function: addClusterCountsAsFeatures
    #This method enables you to add the count of how many times words from a specific
    #cluster occur as review features.
    # Precondition: you must have already specified the name of a cluster file,
    # containing your cluster information.
    # Returns: a dictionary of mapping (cluster --> occurrences of cluster in review)
    def addClusterCountsAsFeatures(self):
        extraCursor = self.db.cursor()
        #print "Adding cluster counts from " + self.clusterFilename + " as features..."
        extraCursor.execute(self.selectQuery)
        review = extraCursor.fetchone()[1]
        tokenizedReview = Twokenize.tokenize(review)
        clusterCounts = {}
        for word in tokenizedReview:
            if self.clusterDict.has_key(word): #Check if word is in common words lexicon
                if clusterCounts.has_key(self.clusterDict[word]): #Check if we've added it yet
                    clusterCounts[self.clusterDict[word]] += 1
                else:
                    clusterCounts[self.clusterDict[word]] = 1
        return clusterCounts

    ##Method: extractedFeaturesOnly
    # Creates a model which does not use bag of words. Makes database requests
    #to pull pre-extracted features and then use them in classifications. Analagous
    #to BOW models, just without the words.
    #Precondition: Must have an extraParamsQuery already stored.
    #Postcondition: self.extraFeaturesVect is set to a vector of the extracted features.
    #Test set is held out.
    def extractedFeaturesOnly(self):
        print "Adding extracted features..."
        extraFeatures = []
        self.y = []
        self.reID = []
        print self.extraParamsQuery
        iter = 0  #"iter" tracks number added; used for debug
        for row in self.c.execute(self.selectQuery):
            gender, prob, count = self.getGenderFromName(row)
            if gender:
                if row[1] != None:
                    iter += 1
                    if iter > self.numReviewsCap:
                        break
                    self.y.append(self.genderConverter[gender])
                    self.reID.append(row[2])
                    extraDict = self.addExtraParameters(self.extraParamsQuery, row[2], self.isBool)
                    extraFeatures.append(extraDict)
        self.extraFeaturesVec = feature_extraction.DictVectorizer()
        self.X =  self.extraFeaturesVec.fit_transform(extraFeatures)
        self.holdOutTestSet()

    ##Method: bagOfWordsPlus
    #Our most frequently-used feature extractor module, bagsOfWordsPlus uses bag of words and any additional
    #  parameters you want. Min and max_n refer to the desired bigram length. This is a required parameter.
    # You can also cut based on the minimum/maximum number of words you desire to see in a review, and whether
    #or not you want to include clustors in your feature extractor's construction.
    #Precondition: You want to have all the reviews you will eventually want to look at in this set, so that
    #the dimensions of all results arrays aline.
    #Postcondition: vectors storing all extracted and BOW features
    def bagOfWordsPlus(self, min_n, max_n, minWords=0, maxWords=sys.maxint, useClusters=False):
        print "Starting Bag of Words, no stop words, with ngram model (%d, %d) and binaryFeatures=%r..." % (min_n, max_n, self.isBool)
        print "Plus extra features..."
        reviews = []
        extraFeatures = []
        #Keeps track of the actual gender of a sample
        self.y = []
        #Keeps track of the ReviewID of a sample
        self.reID = []
        print self.extraParamsQuery
        clusterFeatures = []
        if useClusters:
            self.clusterDict = self.createClusterDict(self.clusterFilename)
        iter = 0
        for row in self.c.execute(self.selectQuery):
            iter += 1
            gender, prob, count = self.getGenderFromName(row)
            if gender:
                if row[1] != None:
                    numWords = len(row[1].split())
                    if numWords > minWords and numWords < maxWords:
                        reviews.append(row[1])
                        self.y.append(self.genderConverter[gender])
                        self.reID.append(row[2])
                        extraDict = self.addExtraParameters(self.extraParamsQuery, row[2], self.isBool)
                        extraFeatures.append(extraDict)
                        if useClusters:
                            clusters = self.addClusterCountsAsFeatures()
                            clusterFeatures.append(clusters)
            if iter > self.numReviewsCap:
                 break
        self.vectorizer = feature_extraction.text.TfidfVectorizer(stop_words='english', smooth_idf=True, use_idf=True, sublinear_tf=True, ngram_range=(min_n, max_n), binary=self.isBool)
        self.extraFeaturesVec = feature_extraction.DictVectorizer()
        extra_X = self.extraFeaturesVec.fit_transform(extraFeatures)
        bow_X = self.vectorizer.fit_transform(reviews)
        if useClusters:
            cluster_X = self.extraFeaturesVec.fit_transform(clusterFeatures)
            self.X = sps.hstack([bow_X, extra_X, cluster_X])
        else:
            self.X = sps.hstack([bow_X, extra_X])
        self.holdOutTestSet()

    def createClusterDict(self, filename):
        print "Loading cluster dictionary..."
        clusterFile = open(filename, "r")
        c_dict = {}
        for line in clusterFile:
            cluster, word, occurs = line.split()
            c_dict[word] = cluster
        clusterFile.close()
        return c_dict

    ##Function: bagOfClusters
    #Classification using each cluster ID as a "word." Uses this as the bag of
    #words base upon which additional features can be added. Analogous to the
    #bagOfWordsPlus method, but with support for clustering rather than extracting
    #individual words.
    #Precondition: You must have already passed the filename of the file containing
    #the clusters you wish to classify on.
    #postcondition: a vector, with names, that you can give to a classifer.
    def bagOfClusters(self, filename, min_n=1, max_n=1):
        #Create dictionary mapping word --> cluster
        clusterDict = self.createClusterDict(filename)
        #Now go through and do bag of words technique:
        print "Starting Bag of Words on clusters, no stop words, with ngram model (%d, %d) and binaryFeatures=%r..." % (min_n, max_n, self.isBool)
        print "Plus extra features..."
        reviews = []
        extraFeatures = []
        #Keeps track of the actual gender of a sample
        self.y = []
        #Keeps track of the ReviewID of a sample
        self.reID = []
        uncommonWord = "__placeholder__"
        print self.extraParamsQuery
        iter = 0
        for row in self.c.execute(self.selectQuery):
            iter += 1
            gender, prob, count = self.getGenderFromName(row)
            if gender:
                if row[1] != None:
                    #Tokenize the review, translate it to clusters, append it to the new thing
                    twokedWords = Twokenize.tokenize(row[1])
                    reviewAsClusters = []
                    for word in twokedWords:
                        if clusterDict.has_key(word):
                            reviewAsClusters.append(clusterDict[word])
                        else:
                            reviewAsClusters.append(uncommonWord)
                    reviews.append(" ".join(reviewAsClusters))
                    self.y.append(self.genderConverter[gender])
                    self.reID.append(row[2])
                    extraDict = self.addExtraParameters(self.extraParamsQuery, row[2], self.isBool)
                    extraFeatures.append(extraDict)
            if iter > self.numReviewsCap:
                 break
        #Now vectorize the reviews on the clusters
        self.vectorizer = feature_extraction.text.TfidfVectorizer(smooth_idf=True, use_idf=True, sublinear_tf=True, ngram_range=(min_n, max_n), binary=self.isBool)
        self.extraFeaturesVec = feature_extraction.DictVectorizer()
        extra_X = self.extraFeaturesVec.fit_transform(extraFeatures)
        bow_X = self.vectorizer.fit_transform(reviews)
        self.X = sps.hstack([bow_X, extra_X])
        self.holdOutTestSet()


    ##Method: bagOfWordsSkipPlus
    # This method, borrowing heavily from the bagOfWordsPlus method, uses bag of words + extracted
    #features as parameters. It also captures skip-grams, which is where word pairs to a distance of
    # k skipped words are recorded.
    def bagOfWordsSkipPlus(self, k):
        print "Starting Bag of Words, no stop words, with unigram model and binaryFeatures=%r..." % (self.isBool)
        print "Plus %d-skip-2-grams" % k
        print "Plus extra features..."
        reviews = []
        extraFeatures = []
        self.y = []
        self.reID = []
        skipgrams = []
        print self.extraParamsQuery
        iter = 0
        for row in self.c.execute(self.selectQuery):
            iter += 1
            gender, prob, count = self.getGenderFromName(row)
            if gender:
                reviews.append(row[1])
                self.y.append(self.genderConverter[gender])
                extraDict = self.addExtraParameters(self.extraParamsQuery, row[2], self.isBool)
                extraFeatures.append(extraDict)
                sentences = row[1].split()
                words = Twokenize.tokenize(row[1])
                if len(words) == 0:
                    pass
                skipG = {}
                for i in xrange(len(words)):
                    for j in xrange(i+1, min(len(words), i + k + 2)):
                        gram = (words[i], words[j])
                        skipG[gram] = skipG.get(gram, 0) + 1
                skipgrams.append(skipG)
                self.reID.append(row[2])
        self.vectorizer = feature_extraction.text.TfidfVectorizer(stop_words='english', smooth_idf=True, use_idf=True, sublinear_tf=True, ngram_range=(1, 1), binary=self.isBool)
        self.extraFeaturesVec = feature_extraction.DictVectorizer()
        self.skipgramsVec = feature_extraction.DictVectorizer()
        extra_X = self.extraFeaturesVec.fit_transform(extraFeatures)
        bow_X = self.vectorizer.fit_transform(reviews)
        skip_X = self.skipgramsVec.fit_transform(skipgrams)
        self.X = sps.hstack([bow_X, extra_X, skip_X])
        self.allY = self.y
        self.allX = self.X
        self.holdOutTestSet()

    ##Method: twokenizeBagOfWordsPlusWordCutoff
    #This method allows you to pass words through the tokenizer from the Twitter API and
    #handle punctuation/conjoined words smoothly. Very similar to the bagOfWords/bagOfWordsPlus
    #methods. Like bagOfWordsPlus, allows you to restrict by min/max words.
    def twokenizBagOfWordsPlusWordCutoff(self, vec="count", min_n=1, max_n=1, minWords=0, maxWords=sys.maxint):
        print "Starting Bag of Words, twokenized, no stop words, with ngram model (%d, %d) and binaryFeatures=%r..." % (min_n, max_n, self.isBool)
        print "Restricting by %d < numWords < %d" % (minWords, maxWords)
        print "Plus extra features..."
        reviews = []
        twoReviews = [] #"Twokenized" reviews
        extraFeatures = []
        self.y = []
        self.reID = []
        print self.extraParamsQuery
        iter = 0
        for row in self.c.execute(self.selectQuery): #Iterate through reviews
            iter += 1
            gender, prob, count = self.getGenderFromName(row)
            if gender:
                if row[1] != None:
                    numWords = len(row[1].split())
                    if numWords > minWords and numWords < maxWords: #Check that word length is within bounds
                        twokedWords = Twokenize.tokenize(row[1])
                        twoReviews.append(" ".join(twokedWords))
                        reviews.append(row[1])
                        self.y.append(self.genderConverter[gender])
                        self.reID.append(row[2])
                        extraDict = self.addExtraParameters(self.extraParamsQuery, row[2], self.isBool)
                        extraFeatures.append(extraDict)
        if vec == "count": #Select from count or TGDIF vectorizor
            self.vectorizer = feature_extraction.text.CountVectorizer(tokenizer=Twokenize.tokenize, ngram_range=(min_n, max_n))
            bow_X = self.vectorizer.fit_transform(reviews)
        else:
            self.vectorizer = feature_extraction.text.TfidfVectorizer(stop_words='english', smooth_idf=True, use_idf=True, sublinear_tf=True, ngram_range=(min_n, max_n), binary=self.isBool)
            bow_X = self.vectorizer.fit_transform(twoReviews)
        self.extraFeaturesVec = feature_extraction.DictVectorizer()
        extra_X = self.extraFeaturesVec.fit_transform(extraFeatures)
        self.X = sps.hstack([bow_X, extra_X])
        self.holdOutTestSet() #Hold out train/dev - test split.

    ##Method: tokenizeToFileForClustering
    # Not used for a complete run of the classifier; tokenize all the input, save it
    #to a file, which can then be fed into a clustering program such as Brown-Cluster.
    #Only clustering based on words occurring in the review; not on added features
    def tokenizeToFileForClustering(self):
        outfile = open("outputClusteringAll.txt", "w")
        numAdded = 0
        for row in self.c.execute(self.selectQuery):
            gender, prob, count = self.getGenderFromName(row)
            if gender:
                if row[1] != None:
                    twokedWords = Twokenize.tokenize(row[1])
                    outfile.write(" ".join([word.encode('utf-8') for word in twokedWords]))
                    outfile.write(" ")
                    numAdded += 1
                    if numAdded > self.numReviewsCap:
                        break
        outfile.close()

############################## Feature Vector Helpers #######################################

    ##Method: getGenderFromName
    #This method queries the database to get the first name, its probability, and its count from
    #the "genders" table in the database.
    #Precondition: This only works when Name is the first element in the row result
    #Returns: the name, unless name does not exist, in which case it returns a null tuple.
    def getGenderFromName(self, row):
        try:
            name = row[0].lower()
            genderQuery = "SELECT Gender, Probability, Count FROM Genders where Name=(?) and probability > (?) and count >(?) and gender <> (?)"
            name = str(name.split()[0])
            self.genderCursor.execute(genderQuery, (name, self.PROBABILITY_CONSTANT, self.COUNT_CONSTANT, "None"))
        except UnicodeEncodeError:
            return (0, 0, None)

        result = self.genderCursor.fetchone()
        return result if result is not None else (0, 0, None)

    ##Metho: normalizeToNumWords
    # Normalize to the number of words present in the review, assuming
    #that numWords has been passed in as one of the extra features.
    #Precondition: self.X is populated with all of the data,
    #which has not yet been split. Names can be found by self.vectorizer.get_feature_names()
    #and self.extraFeaturesVec.get_feature_names(), which are names in order of self.X.
    #Postcondition: self.X will now contain the normalized numbers
    def normalizeToNumWords(self, normalizeExtraFeatures, normalizeBOW):
        self.normalize = True
        print "starting normalization..."
        #Named list of all features
        names = self.makeNamesList()
        numWordsIndex = names.index("numWords")
        for rowI in range(self.X._shape[0]):
            numWords = self.X[(rowI, numWordsIndex)]
            if numWords > 0:
                self.X.data[self.X.indptr[rowI] : self.X.indptr[rowI + 1]] /= numWords
        print "normalized self.X"

############################## Classification Step #######################################

    ##
    #Method: classify
    #This method will run the classifier set (self.classifier) on self.X. First it will
    #split the data into train and dev. Then it will fit the classifier on train, and finally
    #evaluate the dev set. Statistics on these classifications will be printed.
    #Precondition: self.X, self.y, and self.reID must be populated with features, gender designation,
    #and reviewID, respectively.
    #Postcondition: At the end of this run classifier is fit to self.X.x_train, and you have statistics
    #on how well it was able to predict classification for self.X.x_dev. Note that the test set is
    #not touched by these steps.
    ##
    def classify(self, test_size_val=0.4):
        print "Starting classifier..."
        x_train, x_dev, y_train, y_dev, re_train, re_dev = train_test_split(self.X, self.y, self.reID, test_size=test_size_val, random_state=42)

        self.classifier.fit(x_train, y_train)
        training_predicted = self.classifier.predict(x_train)
        predicted_y = self.classifier.predict(x_dev)

        print "NumSamples=%d, numFeatures=%d" % self.X.shape
        print "Gender probability: %.2f %%, count: %d" % (self.PROBABILITY_CONSTANT, self.COUNT_CONSTANT)
        print "Training accuracy is %.3f %%" % (self.accuracy(y_train, training_predicted))
        print "Dev accuracy is %.3f %%" % (self.accuracy(y_dev, predicted_y))

    ##
    #Method: classifyKmeans
    #Initialize and run a Kmeans classifier on the train and dev data. First it will
    #split the data into train and dev. Then it will fit the classifier on train, and finally
    #evaluate the dev set. Statistics on these classifications will be printed.
    #Note that Kmeans here is meant to be used instead of self.classify, rather than in
    #addition to it.
    ##
    def classifyKmeans(self, numClusters=2, test_size_val=.4):
        x_train, x_dev, y_train, y_dev, re_train, re_dev = train_test_split(self.X, self.y, self.reID, test_size=test_size_val, random_state=42)
        km = KMeans(n_clusters=numClusters, init='random', max_iter=100, n_init=1, verbose=1) #For male/female classification 2 clusters
        print "Clustering sparse data with %s" % km
        km.fit(x_dev)

        print "Training stats:"
        print "Homogeneity: %0.3f" % metrics.homogeneity_score(y_train, km.labels_)
        print "Completeness: %0.3f" % metrics.completeness_score(y_train, km.labels_)
        print "V-measure: %0.3f" % metrics.v_measure_score(y_train, km.labels_)
        print "Adjusted Rand-Index: %.3f" % \
            metrics.adjusted_rand_score(y_train, km.labels_)
        print "km train score: "
        print km.score(x_train)

        print "Dev stats:"
        km.predict(self.x_test)
        print "Homogeneity: %0.3f" % metrics.homogeneity_score(y_dev, km.labels_)
        print "Completeness: %0.3f" % metrics.completeness_score(y_dev, km.labels_)
        print "V-measure: %0.3f" % metrics.v_measure_score(y_dev, km.labels_)
        print "Adjusted Rand-Index: %.3f" % \
            metrics.adjusted_rand_score(y_dev, km.labels_)
        print "km dev score: "
        print km.score(x_dev)

########################### Classifier Analytics ####################################

    ##Method: accuracy
    #This method will compare the classification of the classifier's
    #prediction on a given input, to the actual labeling of that input.
    #Precondition: must have already computed some predicted_y on the input, and have
    #access to the real y values.
    #Postcondition: outputs the improperly classified elements over total elements.
    def accuracy(self, y, predicted_y):
        error = 0.0
        elems = len(y)
        for i in range(len(y)):
            if y[i] != predicted_y[i]:
                error += 1
        return (elems - error)/elems * 100

    ##Method: topNCoefficients
    # Select the number of coefficients to print, alongside the feature. Female features
    #will have a negative weight, while male ones have positive. A good way to isolate features
    #which have "bubbled up" during feature selection process.
    #Precondition: Must have feature names captured in self.vectorizer and self.extraFeaturesVec
    #Output: a printout of the (ordered) top N coefficient weights
    ##
    def topNCoefficients(self, numToPrint):
        weights = self.classifier.coef_.tolist()
        numTop = numToPrint
        names = self.makeNamesList()
        if len(names) < numTop:
            numTop = len(names)
        tops = heapq.nlargest(numTop, enumerate(weights[0]), key=lambda x: abs(x[1])) #Use absolute magnitude of weight as key
        for item in tops:
            print "Item " + names[item[0]].encode('utf-8') + " has weight " + str(item[1])

    ##Method: makeNamesList
    #Called by topN coefficients.akes a list of all named features in order, from self.vectorizer.get_feature_names
    #and self.extraFeaturesVec.get_feature_names(). Returns a list of the ordered feature names
    #Currently not extended to work with names not included in the BOW vectorizer (self.vectorizer) or the extracted
    #features vector.
    def makeNamesList(self):
        names = []
        if self.vectorizer != None:
            names.extend(self.vectorizer.get_feature_names())
        if self.extraParamsQuery != None:
            names.extend(self.extraFeaturesVec.get_feature_names())
        return names

    ## Method: printCoefficients
    # Print all the coefficients in a list, ordered by index. This is just if you want an ordered list
    #of ALL the features serving as support for the svm.
    def printCoefficients(self):
        weights = self.classifier.coef_.tolist()
        names = []
        if self.vectorizer != None:
            names.extend(self.vectorizer.get_feature_names())
        if self.extraParamsQuery != None:
            names.extend(self.extraFeaturesVec.get_feature_names())
        for i in range(len(weights[0])):
            print "len is " + str(len(weights[0]))
            print "weight is " + str(weights[0][i]) + " for item " + names[i].encode('utf-8')


    ##Method: misclassifiedReviews
    #This method lets you save the misclassified reviews after running the given
    #classifier. This is helpful because it lets you see the weaknesses in the current
    #model.
    #Precondition: Already have run classifier on the dev set
    #Postcondition: The misclassified reviews
    def misclassifiedReviews(self):
        print "Saving misclassified reviews..."

        getReviewTextQuery = "SELECT u.Name, ReviewText, b.Name, b.BusinessID from Reviews r, Users u, Businesses b where r.UserID = u.UserID and r.BusinessID=b.BusinessID and r.ReviewID=(?)"
        getCategoriesQuery = "Select Category from Categories c where BusinessID=(?)"

        predicted = self.classifier.predict(self.allX)
        print "Overall accuracy is %.3f %%" % (self.accuracy(self.allY, predicted))

        gender = {0:"woman", 1:"man"}

        businesses = {}
        categories = {}
        countBusinesses = {}
        countCategories = {}
        countGenderBusiness = {}
        misclassifiedGenderBusiness = {}
        countGenderCategory = {}
        misclassifiedGenderCategory = {}

        emptyDict = {0: 0, 1:0}

        for i in range(len(predicted)):
            try:
                self.genderCursor.execute(getReviewTextQuery, [self.reID[i]])
                result = self.genderCursor.fetchone()
                if result is not None:
                    countBusinesses[result[2]] = countBusinesses.get(result[2], 0) + 1
                    if result[2] not in countGenderBusiness:
                        countGenderBusiness[result[2]] = {0: 0, 1:0}
                    countGenderBusiness[result[2]][self.allY[i]] += 1
                    if predicted[i] != self.allY[i]:
                        print "For business: %s, classified %s as a %s" % (result[2], result[0], gender[predicted[i]])
                        print result[1]
                        businesses[result[2]] = businesses.get(result[2], 0) + 1
                        if result[2] not in misclassifiedGenderBusiness:
                            misclassifiedGenderBusiness[result[2]] = {0: 0, 1:0}
                        misclassifiedGenderBusiness[result[2]][predicted[i]] += 1
                        print "Categories:"
                    for row in self.c2.execute(getCategoriesQuery, [result[3]]):
                        countCategories[row[0]] = countCategories.get(row[0], 0) + 1
                        if row[0] not in countGenderCategory:
                            countGenderCategory[row[0]] = {0: 0, 1:0}
                        countGenderCategory[row[0]][self.allY[i]] += 1
                        if predicted[i] != self.allY[i]:
                            print row[0]
                            categories[row[0]] = categories.get(row[0], 0) + 1
                            if row[0] not in misclassifiedGenderCategory:
                                misclassifiedGenderCategory[row[0]] = {0: 0, 1:0}
                            misclassifiedGenderCategory[row[0]][predicted[i]] += 1
            except ValueError:
                print ValueError
                print self.reID[i]

        print "Percentage of misclassfied businesses..."
        for w in sorted(businesses, key=lambda x: countBusinesses[x], reverse=True):
            try:
                print "%s: %.3f, out of %d reviews"% (w, float(businesses[w])/countBusinesses[w], countBusinesses[w])
                print "Gender breakdown: %d men, %d women, misclassifying %d women as men and %d men as women.\n" % (countGenderBusiness[w][1], countGenderBusiness[w][0], misclassifiedGenderBusiness[w][1], misclassifiedGenderBusiness[w][0])
            except UnicodeEncodeError:
                print UnicodeEncodeError
        print "Percentage of misclassfied categories..."
        for w in sorted(categories, key=lambda x: countCategories[x], reverse=True):
            try:
                print "%s: %.3f, out of %d reviews"% (w, float(categories[w])/countCategories[w], countCategories[w])
                print "Gender breakdown: %d men, %d women, misclassifying %d women as men and %d men as women.\n" % (countGenderCategory[w][1], countGenderCategory[w][0], misclassifiedGenderCategory[w][1], misclassifiedGenderCategory[w][0])
            except UnicodeEncodeError:
                print UnicodeEncodeError

    ##Method: accuracyForCategories
    #This method allows you to subset accuracy into looking at misclassified
    #men and women. It is helpful because it allows you to identify specific gender biases.
    #After running, it will print out miscategorized men and women percentages.
    def accuracyForCategories(self, y, predicted_y):
        numMen = 0
        numWomen = 0
        misclassifiedMen = 0.0
        misclassifiedWomen = 0.00
        for i in range(len(y)):
            numMen += (1 if y[i] == 1 else 0)
            numWomen += (1 if y[i] == 0 else 0)
            if y[i] != predicted_y[i]:
                misclassifiedMen += (1 if y[i] == 1 else 0)
                misclassifiedWomen += (1 if y[i] == 0 else 0)
        print "misclassified Men=%.2f" % (misclassifiedMen/numMen)
        print "misclassified Women=%.2f" % (misclassifiedWomen/numWomen)


########################### Handling test data set ####################################

    ##
    #Method: holdoutTestSet
    #This method we use to hold out a test set from our database, to avoid accidentally training
    #on our test data.
    #Returns: None; sets self.X, self.y, .x_test, and .y_test to subset the data
    ##
    def holdOutTestSet(self):
        print "Holding out test set..."
        X, x_test, y, y_test, reID, reID_test = train_test_split(self.X, self.y, self.reID, test_size=.25, random_state=42)
        self.X = X
        self.y = y
        self.x_test = x_test
        self.y_test = y_test
        self.reID = reID
        self.reID_test = reID_test


    ##Method: classifyTestSEt
    #Only for use at the very end of experimentation (as all feature optimization, etc should be
    #done only on the train and dev sets. Will output the test accuracy of self.classifier.
    #Precondition: should have a pre-trained and optimized classifier, and an untouched test set
    #Postcondition: the output weights of your classifier; knowing how correct you are!
    def classifyTestSet(self):
        test_predicted = self.classifier.predict(self.x_test)
        print "NumSamples=%d, numFeatures=%d" % self.x_test.shape
        print "Gender probability: %.2f %%, count: %d" % (self.PROBABILITY_CONSTANT, self.COUNT_CONSTANT)
        print "Test accuracy is %.3f %%" % (self.accuracy(self.y_test, test_predicted))

    ##Method: close
    #Removes the cursors and closes the DB
    #
    def close(self):
        self.c2.close()
        self.c.close()
        self.genderCursor.close()
        self.db.close()