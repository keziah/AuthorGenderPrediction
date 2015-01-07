#Authors:
    ## Keziah Plattner, SUNetID = keziah
    ## Lilith Wu, SUNetID = lilithwu
    ## Laura Hunter, SUNetID = lmhunter

Welcome to our mini Yelp database! Attached is code that will
classify a (very small) Yelp database, comprised of only 2000 reviews.
The full Yelp academic database we used can be found at https://www.yelp.com/academic_dataset.

To setup:
    This short script should not require much setup; all that needs to be done is to
    modify the database path at the top of the FeatureTestingClass.py file (under init).
    Make that point to the location where you have the mini_yelp_db.db (under 
    AuthorGenderPrediction->dataFolder.zip here).

To run:
At the command line, type "python runFeatureTesting.py <model>", where model is either:
-"bu", a basic unigram model
-"upf", a unigram plus some extracted features model
-"bpf", a bigram plus extracted features model
-"rl", a model that screens for word length

All of these are very basic tests and have low accuracy on such a small training
set (dev accuracy is 61.283 %), but we hope they give you an idea of how our code was implemented! We have left
more components of our actual development and test code in the FutureTestingClass.py
file, if you would like to see more!

About FeatureTestingClass.py:
This file is quite long and divided into a number of subsections.
-"Initialize FTC Settings" will set up the settings to be used for that instantiation
of the class, such as setting features to be binary or scalar, incorporate a cutoff review
number, etc.
-"Create feature vectors" is where a model to vectorize and store the features is
run
-"Feature vector helpers" is a subsection of feature vectors, containing common code
to help establish vectors.
-"Classification step" is where the classifier is run
-"Accuracy" is where we put code relating to performance metrics and error analysis
-"Handling the test set" is where we took care of segregating our test set, and later
analyzing it at the end of the project.

Other files:
In our subfolder AuthorGenderPrediction>Results, we also included some additional data files, showing the results of one of our word
clustering runs (400clusters.txt) and some highly-weighted male and female words. The file MisclassifiedReviewsBusinessesCategories contains the text of our misclassified reviews, and percent of misclassification by Business and Category. This is useful if you want to see the kind of reviews we made mistakes on, and our accuracy rate by Business/Category.
We include these to illustrate some of the other things we did, since the clustering
process took a long time and the output is the most interesting thing!
