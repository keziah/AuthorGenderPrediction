#Authors:
    ## Keziah Plattner, SUNetID = keziah
    ## Lilith Wu, SUNetID = lilithwu
    ## Laura Hunter, SUNetID = lmhunter

from FeatureTestingClass import FeatureTesting
import sys

extractedFeaturesQuery = "SELECT numChars, numWords, hasCaps, numNonInitCaps, percentCaps, percentAllCaps," \
                   " percentYou, percentI, youVsI, numFemaleWords, numMaleWords, avgWordLength, avgSentenceLength," \
                   " numNewLines, punctuationCount, smileyFace, frownyFace, smirkFace, " \
                   "cryingFace, singleExclamationMarks, twoExclamationMarks, threeToSixExclamationMarks, " \
                   "greaterSixConsecutiveExclamationMarks, regularEllipsis, fourToSevenConsecutivePeriods, " \
                   "greaterSevenConsecutivePeriods, singleQuestionMark, doubleQuestionMark, threeToSixQuestionMarks, " \
                   "greaterSixConsecutiveQuestionMarks, interrobangTwo, interrobangThree, interrobangFourToSix, " \
                   "interrobangGreaterThanSix, numUniqueWords, numPnouns, numNounSpec, numNetSpeak, numEmoTerms, " \
                   "numBackchannel, numArabicNums, numPosEmo, numNegEmo, endsWithAble, endsWithAl, endsWithFul, " \
                   "endsWithIble, endsWithIc, endsWithIve, endsWithLess, endsWithLy, endsWithOus, numThreeVowelRepeats " \
                   "FROM Features, WFeatures, WordFeatures WHERE Features.ReviewID=(?) AND WordFeatures.ReviewID=Features.ReviewID AND WFeatures.ReviewID=Features.ReviewID"

numWordsOnlyQuery = "SELECT numWords FROM Features WHERE ReviewID=(?)"

selectQuery = "SELECT Name, ReviewText, ReviewID from Reviews r, Users u where r.UserID = u.UserID"


#Runs a unigram bag of words model without any added features.
def basicUnigram():
    ft = FeatureTesting()
    ft.setLinearSVM(.5, False)
    ft.setProb(0.9)
    ft.setCount(20)
    ft.setSelectStatement(selectQuery)
    ft.extraFeaturesSelectStatement(numWordsOnlyQuery)
    ft.setFeaturesBinaryStatus(False)
    ft.bagOfWordsPlus(1, 1, useClusters=False)
    ft.classify()
    ft.close()

#Runs our basic unigram bag of words model with added features:
def unigramPlusFeatures():
    ft = FeatureTesting()
    ft.setLinearSVM(.5, False)
    ft.setProb(0.9)
    ft.setCount(20)
    ft.setSelectStatement(selectQuery)
    ft.extraFeaturesSelectStatement(extractedFeaturesQuery)
    ft.setFeaturesBinaryStatus(False)
    ft.bagOfWordsPlus(1, 1, useClusters=False)
    ft.classify()
    ft.close()

#Runs a bigram bag of words model with added features:
def bigramPlusFeatures():
    ft = FeatureTesting()
    ft.setLinearSVM(.5, False)
    ft.setProb(0.9)
    ft.setCount(20)
    ft.setSelectStatement(selectQuery)
    ft.extraFeaturesSelectStatement(extractedFeaturesQuery)
    ft.setFeaturesBinaryStatus(False)
    ft.bagOfWordsPlus(2, 2, useClusters=False)
    ft.classify()
    ft.close()

    #Runs our basic unigram bag of words model with added features, restricting
    #length to lie between 100 and 500 words :
def unigramPlusFeaturesRestrictLength():
    ft = FeatureTesting()
    ft.setLinearSVM(.5, False)
    ft.setProb(0.9)
    ft.setCount(20)
    ft.setSelectStatement(selectQuery)
    ft.extraFeaturesSelectStatement(extractedFeaturesQuery)
    ft.setFeaturesBinaryStatus(False)
    ft.bagOfWordsPlus(1, 1, minWords=100, maxWords=500)
    ft.classify()
    ft.close()

whichModel = sys.argv[1]
if whichModel == "bu":
    print "Starting basic unigram model..."
    basicUnigram()
elif whichModel == "upf":
    print "Starting unigram plus features model..."
    unigramPlusFeatures()
elif whichModel == "bpf":
    print "Starting bigram plus features model..."
    bigramPlusFeatures()
elif whichModel == "rl":
    print "Starting restricted review length model..."
    unigramPlusFeaturesRestrictLength()
else:
    print "Invalid model choice; exiting."