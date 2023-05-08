Download Link: https://assignmentchef.com/product/solved-si-630-homework-1-classification
<br>
Homework 1 will introduce you to building your first NLP classifiers from scratch. Rather than use many of the wonderful, highly-customizable machine learning libraries out there (e.g., sklearn), you will build bare-bones (and much slower) implementations of two classic algorithms: Naive Bayes and Logistic Regression. Implementing these will provide you with deep understanding of how these algorithms work and how the libraries are actually calculating things. Further, you’ll be implementing two knob to tune in the data: smoothing for Naive Bayes and the learning rate for Logistic Regression. In implementing these, you will get to adjust them and not only understand their impact but understand <em>why </em>they’re impacting the result.

The programming assignment will be moderately challenging, mostly stemming from the conceptual level. The code for each algorithm is be 20-30 lines, with additional lines for simple I/O and boilerplate code. The big challenge will be from trying to ground the mathematical concepts we talk about in class in actual code, e.g., “what does it mean to calculate the likelihood???”. Overcoming this hurdle will take a bit of head scratching and possibly some debugging but it will be worth it, as you will be well on your way to understanding how newer algorithms work and even how to design your own!

There are equations in the homework. You should not be afraid of then—even if you’ve not taken a math class since high school. Equations are there to provide precise notation and wherever possible, we’ve tried to explain them more. There are also many descriptions in the lecture materials and textbooks. Getting used to equations will help you express yourself in the future. If you’re not sure what something means, please ask; there are six office hours between now and when this assignment is due, plus an active Canvas discussion board. You’re not alone in wondering and someone will likely benefit from your courage in asking.

Given that these are fundamental algorithms for NLP and many other fields, you are likely to run into implementations of them online. While you’re allowed to look at them (sometimes they can be informative!), all work you submit should be your own (see Section 8).

Finally, remember this is a no-busywork class. If you think some part of this assignment is unnecessary or to much effort, let us know. We are happy to provide more detailed explanations for <em>why </em>each part of this assignment is useful. Tasks that take inordinate amounts of time could even be a bug in the homework and will be fixed.

<h1>2           Classification Task</h1>

The School of Information’s mission is literally “We create and share knowledge so that people will use information – with technology – to build a better world.” One major impediment to people’s use of technology today is uncivil behavior online. Hate speech, bullying, toxic language, and other forms of harassment all affect people’s ability to access information and engage in social activities. In this homework, you’ll be tackling this problem head-on by building a cyber troll detector! Help mitigate people attacking one another by flagging aggressive tweets for administrators. Your task will be to build two classifiers that label text as either an aggressive tweet or a regular tweet. The fate of free discourse rests on your shoulders.

Given the toxic nature of people’s tweets, please be aware that you may see messages that are deeply offensive and an affront to all that is right and good. Neither the instructor nor the community at SI agrees with or supports these viewpoints. Their inclusion in the homework is entirely for the purposes of development new NLP technologies that help administrators and moderators identify and remove these messages from otherwise civil discourse. As such, it would be difficult to develop an cyber troll classifier without actually seeing what they look like.

Notation Notes : For the whole assignment description we’ll refer to classifier features as <em>x</em><sub>1</sub><em>,…,x<sub>n </sub></em>∈ <em>X </em>where <em>x<sub>i </sub></em>is a single feature and <em>X </em>is the set of all features. When start out each feature will correspond to the presence of a word; however, you are free in later parts of the homework to experiment with different kinds of features like bigrams that denote to consecutive words, e.g., “not good” is a single feature. We refer the to the class labels as <em>y</em><sub>1</sub><em>,…,y<sub>n </sub></em>∈ <em>Y </em>where <em>y<sub>i </sub></em>is a single class and <em>Y </em>is the set of all classes. In our case, we have a <em>binary </em>classification tasks so there’s really only <em>y</em><sub>1 </sub>and <em>y</em><sub>2</sub>. When you see a phrase like <em>P</em>(<em>X </em>= <em>x<sub>i</sub></em>|<em>Y </em>= <em>y<sub>j</sub></em>) you can read this as “the probability that we observe the feature (<em>X</em>) <em>x<sub>i </sub></em>is true, given that we have seen the class (<em>Y </em>) is <em>y<sub>j</sub></em>”.

We’ll also use the notation <em>exp</em>(<em>x<sub>i</sub></em>) to refer to <em>e<sup>x</sup></em><em><sup>i </sup></em>at times. This notation lets us avoid superscript when the font might become too small or when it makes equations harder to follow.

Implementation Note : Unless otherwise specified, your implementations should <em>not </em>use any existing off-the-shelf machine learning libraries or methods. You’ll be using plain old Python (or R, if brave) along with numeric libraries like numpy to accomplish everything.

<h1>3           Data</h1>

Homework 1 has three associated files (you can find them on our Kaggle competition associated with this homework):

<ul>

 <li>txt This file contains the tweets you will use to train your classifiers.</li>

 <li>txt This file contains the human-annotated labels of the tweets in the previous file.</li>

 <li>txt This file contains the tweets you will use as the development data.</li>

 <li>txt This file contains the human-annotated labels of the tweets in the previous file.</li>

 <li>txt This file is the tweets you will make predictions using your trained classifiers and upload results.</li>

</ul>

As a part of this assignment, we’ll be using Kaggle in the classroom to report predictions on the test data. This homework is not a challenge, per se. Instead, we’re using Kaggle so you can get a sense of how good your implementation’s predictions are relative to other students. Since we’re all using the same data and implementing the same algorithms, your scores should be relatively close to other students. If you decide to take up the optional part and do some feature engineering, you might have slightly higher scores, but no one will be penalized for this.

We’ve set up two Kaggle competitions for Logistic Regression (LR) and Naive Bayes (NB) classifiers respectively:

<ul>

 <li>LR competition link: https://www.kaggle.com/c/si630w20hw1lr</li>

 <li>LR invitation link: https://www.kaggle.com/t/5f1651bc7bab410389b14c8697b07df6</li>

 <li>NB competition link: https://www.kaggle.com/c/si630w20hw1nb</li>

 <li>NB invitation link: https://www.kaggle.com/t/2573fa891af14b53b4009f63068ead45</li>

</ul>

<h1>4           Task 1: Naive Bayes Classifier</h1>

<h2>4.1         Part 1</h2>

Implement a Naive Bayes classifier. Recall that the classifier was defined as

<em>y</em>ˆ = argmax<em>P</em>(<em>Y </em>= <em>y<sub>i</sub></em>|<em>X</em>)

<em>y<sub>i</sub></em>∈Y

where Y is the set of classes (i.e., personal insult or not, in our case). This equation can seem pretty opaque at first, but remember that <em>P</em>(<em>Y </em>= <em>y<sub>i</sub></em>|<em>X</em>) is really what’s the probability of the class <em>y<sub>i </sub></em>given the data we see in an instance. As such, we can use Bayes Rule to learn this from the training data.

In Task 1, you’ll implement each step of the Naive Bayes classifier in separate functions. As a part of this, you’ll also write code to read in and tokenize the text data. Here, tokenization refers to separating words in a string. In the second half, you’ll revisit tokenization to ask if you can do a better job at deciding what are words. Task 1 will provide much of the boiler plate code you’ll need for Task 2.

<ul>

 <li>Write a function called tokenize that takes in a string and tokenizes it by whitespace, returning a list of tokens. You should use this function as you read in the training data so that each whitespace separated word will be considered a different feature (i.e., a different <em>x<sub>i</sub></em>).</li>

 <li>Write a function called train to compute <em>P</em>(<em>X </em>= <em>x<sub>i</sub></em>), <em>P</em>(<em>Y </em>= <em>y<sub>j</sub></em>), and <em>P</em>(<em>X </em>= <em>x<sub>i</sub></em>|<em>Y </em>= <em>y<sub>i</sub></em>) from the training data. The function should also include an argument called smoothingalpha that by default is 0 but can take on any non-negative value to do additive smoothing (see Slide 143 from Lecture 1), which you might also see as Laplacian smoothing.</li>

 <li>Write a function called classify that takes in a tokenized document (i.e., a list of words) and computes the Naive Bayes classification, returning the class with the highest posterior probability. Note that not you might not have all the new document’s words in the training data. Be sure to take this into account in your implementation depending on whether you used smoothing or not!</li>

 <li>Train the classifier on the training data the run the classifier on the development data with no smoothing and report your performance in your submission in terms of F1.<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a></li>

 <li>What happens as you change the value of smoothingalpha? Include a plot of your classifier’s performance on the development data where (i) your model’s performance is on the y-axis and (ii) the choice in smoothingalpha is on the x-axis. Note that most people use <em>α </em>= 1; does this value give good performance for you?</li>

 <li>Submit your best model’s predictions on the test data to KaggleInClass competition for Naive Bayes. Note that this is a separate competition from the Logistic Regression one so that you can compare your scores.</li>

</ul>

<h2>4.2         Part 2</h2>

Notice that we are probably doing a bad job of tokenizing due to punctuation. For example, “good” and “good,” are treated as different features because the latter has a comma at the end and “Good” and “good” are also different features because of capitalization. Do we want these to be different? Furthermore, do we want to include every token as a feature? (Hint: could a regular expression help you filter possible features?) Note that you have to implement the tokenization yourself; no importing NLTK and wrapping their functions (though you might take some inspiration from them).

<ul>

 <li>Write a better tokenizing function called bettertokenize that fixes these issues. In your report, describe which kinds of errors you fixed, what kind of features you included, and why you made the choices you did.</li>

 <li>Recompute your performance on the development data using the bettertokenize method. Describe in your report what impact this had on the performance.</li>

</ul>

<h2>4.3         Part 3 (optional)</h2>

Parts 1 and 2 only used <em>unigrams</em>, which are single words. However, longer sequences of words, known as <em>n</em>-grams, can be very informative as features. For example “not offensive” and “very offensive” are very informative features whose unigrams might not be informative for classification on their own. However, the downside to using longer <em>n</em>-grams is that we now have many more features. For example, if we have <em>n </em>words in our training data, we could have <em>n</em><sup>2 </sup>bigrams in the worst case; just 1000 words could quickly turn into 1,000,000 features, which will slow things down quite a bit. As a result, many people threshold bigrams based on frequency or other metrics to reduce the number of features.

In Part 3, you’ll experiment with adding bigrams (two words) and trigrams (three words) and measuring their impact on performance. Part 3 is entirely optional and included for people who want to go a bit further into the feature engineering side of things.

<ul>

 <li>Count how many unique, unigram, bigrams, and trigrams there are in the training data and report each number.</li>

 <li>Are the bigram and trigram counts you observe close to the worst case in terms of how many we could observe? If not, why do you think this is the case? (Hint: are all words equally common? You might also check out Zipf’s Law).</li>

 <li>What percent of the unique bigrams and trigrams in the development data were also seen in the training data?</li>

 <li>Choose a minimum frequency threshold and try updating your solution to use these as features. We recommend creating a new method that wraps your tokenize method and returns a list of features.</li>

</ul>

<h1>5           Task 2: Logistic Regression</h1>

In the second task, you’ll implement logistic regression, which you might recall is

Note that when you implemented Naive Bayes, it didn’t care how many classes were present. In contrast, Logistic Regression is restricted to two classes, which we represent as <em>binary </em>so that <em>y </em>is either 0 or 1. Conveniently, your training data is already set up for this (though you’ll need to use int() to convert the string).

Your implementation will be one of the simplest possible formulations of logistic regression where you use gradient descent to iteratively find better parameters <em>β</em>. Smarter solutions like those in sklearn will use numerical solvers, which are much (much) faster. The purpose of this problem is to give you a sense of how to compute a gradient.

For Task 2, you can (and should) re-use all the boilerplate and tokenization code from the Naive Bayes classifier. It is assumed that your solution will already include something to read in the training data and construct a matrix where rows are instances and columns denote features (e.g., if <em>X</em>[0<em>,</em>7] = 3, it means that the first instance (index 0) saw the word whose feature index is 7 occur 3 times).

<ul>

 <li>Implement a function called sigmoid that implements the sigmoid function, <em>S</em></li>

</ul>

. Your function should be <em>vectorized </em>so that it computes the sigmoid of a whole vector of numbers at once. Conveniently, numpy will often do this for you, so that if you multiply a number to a numpy array, it will multiply each item in the array (the same applies to functions used on an array (hint)). If you’re not sure, please ask us! You’ll need to use the sigmoid function to make predictions later.

<ul>

 <li>Implement a function called loglikelihood that calculates the log likelihood of the training data given our parameters <em>β</em>. Note that we could calculate the likelihoood but since we’ll be using log-arithmetic version (hence the <em>log</em>-likelihood) to avoid numeric underflow and because it’s faster to work with. Note that we could calculate the log likelihood <em>ll </em>over the whole training data as</li>

</ul>

<em>ll </em>= <sup>X </sup><em>y<sub>i</sub>B<sup>T </sup>x<sub>i </sub></em>− <em>log</em>(1 + <em>exp</em>(<em>B<sup>T </sup>x<sub>i</sub></em>))

<em>i</em>=1<em>,…,n</em>

where <em>β </em>is the vector of all of our coefficients. However, you’ll be implementing <em>stochastic gradient descent</em>, where you update the weights after computing the loss for a <em>single </em>randomly-selected (stochasticly!) item from the training set.

<ul>

 <li>Given some choice of <em>β </em>to make predictions, we want to use the difference in our prediction <em><sup>Y</sup></em><sup>ˆ </sup>from the ground truth <em>Y </em>to update <em>β</em>. The gradient of the log likelihood tells us which direction (positive or negative) to make the update and how large the update should be. Write a function computegradient to compute the gradient. Note that we can compute the whole gradient using</li>

</ul>

∇<em>ll </em>= <em>X<sup>T </sup></em>(<em>Y </em>− <em>Y</em><sup>ˆ</sup>)

Note that <em>Y </em>is a binary vector with our ground truth (i.e., the training data labels) and <em><sup>Y</sup></em><sup>ˆ </sup>is the binary vector with the predictions. To get a sense of why this works, think about what gradient will equal if our prediction for item <em>i</em>, <em>Y</em><sup>ˆ</sup><em><sub>i </sub></em>is the same as the ground truth <em>Y<sub>i</sub></em>; if we use this gradient to update our weight for <em>β<sub>i</sub></em>, what effect will it have?

<ul>

 <li>Putting it all together, write a function logisticregression that takes in a

  <ul>

   <li>a matrix <em>X </em>where each row has the features for that instance</li>

   <li>a vector <em>Y </em>containing the class of the row</li>

   <li>learningrate which is a parameter to control how much you change the <em>β </em>values each step</li>

   <li>numstep how many steps to update <em>β </em>before stopping</li>

  </ul></li>

</ul>

Your function should iteratively update the weight vector <em>β </em>at each step by making predictions, <em><sup>Y</sup></em><sup>ˆ</sup>, for each row of <em>X </em>and then using those predictions to calculate the gradient. You should also include an <em>intercept </em>coefficient.<a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a>

Note that you can make your life easier by using matrix operations. For example, to compute <em><sup>Y</sup></em><sup>ˆ</sup>, multiply the whole feature matrix <em>X </em>by the <em>β </em>vector If you’re not sure how to do this, don’t worry! Please come talk to us during office hours!

<ul>

 <li>Write a function predict that given some new vector (i.e., something like a row from <em>X</em>), predict the class.</li>

 <li>Train your model on the training data learningrate=5e-5 (i.e., a very small number) and numsteps = 1000. Make a plot of the log-likelihood every step. Did the model converge at some point (i.e., does the log likelihood remain stable)?</li>

 <li>Change the learningrate to a much larger and much smaller value and repeat the training procedure for each. Plot all three curves together and describe what you observe. You’re welcome (encouraged, even!) to try additional learning rates. If your model is very slow or if it converges quickly, you can also reduce the number of steps for this question.</li>

 <li>After training on the training data, use your logistic regression classifier to make predictions on the validation dataset and report your performance using the F1.</li>

 <li>Submit your best model’s predictions on the test data to the KaggleInClass competition for Logistic Regression. Note that this is a separate competition so that you can compare your scores with the Naive Bayes model.</li>

</ul>

<h1>6           Hints</h1>

A few hints to help get you started:

<ul>

 <li>The set, defaultdict and Counter classes in python will likely come in handy— especially for Naive Bayes.</li>

 <li>Be sure you’re not implementing full gradient descent where you compute the gradient with respect to all the items. Stochastic gradient descent uses <em>one </em>instance at a time.</li>

 <li>If you’re running into issues of memory and crashing, try ensuring you’re multiplying <em>sparse </em> The dense term-document matrix is likely too big to fit into memory.</li>

</ul>

<a href="#_ftnref1" name="_ftn1">[1]</a> You can use sklearn’s implementation of F1 (sklearn.metrics.f1score) to make your life easier: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

<a href="#_ftnref2" name="_ftn2">[2]</a> The easiest way to do this is to add an extra feature (i.e., column) with value 1 to <em>X</em>; the functions np.ones and np.stack with axis=1 will help.