# In progress

# Spotify_Predict_Hit_Song
This is the final assignment using DataBricks, from the Data Science & Advanced Analytics course at the Big Data & Analytics Masters @ EAE class of 2021. 

Professor:
- Marta Tolós Rigueiro
- Pere Miquel Brull Borràs
- Alberto Villa Manrique

Team:
- Romain Baleynaud (GitHub)
- Henrique Avila
- Joseph Higaki (GitHub)
- Raquel Ganuza
- Ziyad Ashukri

## Business Understanding
The music industry generates $21.5 Billion per year. Other than the revenue it generates, music touches
each and everyone of us on a personal level. More importantly, the music industry is a crowded industry,
for example according to Spotify reporting, there are 60,000 songs uploaded on their platform each day.
Combining other streaming services, such as Apple music, Napster and Dezzer the number of uploaded
songs each day reaches hundreds of thousands.
With the massive number of songs uploaded each day, depending on which chart we are looking at, only
a super small fraction of those songs are considered to be hit songs. The questions that present
themselves in this case are, what constitutes a hit song? What makes a song a hit song? With that being
said, our project objective will be to build a Machine Learning model that can predict if a song is a hit
or not?
It is indisputable that music is part of our lives on a personal level. People relate moments, places and
even people to songs. But the music industry is a business and as such seeks to make a profit.
In the last decades many platforms have emerged where you can play music and one of the most
prominent in the music industry is Spotify. Spotify pays based on the number of plays. First the rights
holders (which can be record labels, publishers or composers) are paid, then the distributors
(representatives) and finally the artist.
So, the question is: how can you increase the revenue per playback on Spotify?

## Architecture
The outline of how we obtain the data ended up looking like this:
![arch](https://user-images.githubusercontent.com/85830810/125612220-286515ad-43fe-4bf0-9d6d-fd8876856765.png)

## Data understanding
From the three sources of information from which we have extracted our data, we have obtained a total
of 4286 songs and 31 variables, which we are going to name and describe in the following table:
![under1](https://user-images.githubusercontent.com/85830810/125612124-a7e194a8-2a2b-4430-8c1a-60e1715866d0.png)
![under2](https://user-images.githubusercontent.com/85830810/125612127-a46a5068-acf9-42cd-91ee-9c294473335c.png)

Artists data:
![under3](https://user-images.githubusercontent.com/85830810/125612128-bb39793c-2a48-4fb6-867a-b0660237dd69.png)
![under4](https://user-images.githubusercontent.com/85830810/125612129-195cb40f-90d8-4e68-bed4-72171921d3cb.png)

Once the data have been obtained and the variables have been described, the problems we may face
due to the quality of the data may be the following:
– heteroscedasticity
– missing values
– outliers
– duplicates
We will deal with these problems in the data cleansing part.

## Models creations
First, to describe the sample that participated in our study, we will carry out a univariate analysis using
descriptive statistics. We will focus on two of them: measures of central tendency (mean, median,
mode...) and measures of dispersion (standard deviation). For qualitative or categorical variables, we will
use a frequency table and for quantitative or continuous variables we will perform descriptive analysis.

Secondly, in order to discover the possible relationship between two variables, we will perform a
bivariate analysis. Within this, we will distinguish the types of variables that we wish to relate in order to
subsequently apply the appropriate statistics. In the correlation table, we will use Spearman's ranks because Spearman's correlation evaluates the monotonic relationship between two continuous or ordinal variables. That the relationship is monotonic means that the variables tend to change at the
same time, but not necessarily at a constant rate.

We will also test the Spearman coefficients to determine whether or not there is a relationship between
the variables and the total number of Streams, our objective being to reject the H0: the variable X does
not affect the number of Total Streams. We will use Dataiku to show the results.
![corralation](https://user-images.githubusercontent.com/85830810/125611710-932856a6-66e3-4410-9761-e4cdcd367798.png)

From the Spearman’s correlation table we can conclude that the variables Wks, T10, Pk, PKStreams and
Total have a high level of correlation with each other. We can also observe that the energy variable has a
correlation of 0.727 with the loudness variable. This means that there exists a high monotone correlation
between those two variables.

### Algorithms choice
Because we don’t need to do text classification with our input data; it is better to use the
KNeighborsClassifier (KNN) model instead of the Naive Bayes model . KNN made sense too with the
layout of our data because it has outliers for each parameter we want to use. Also because they are
grouped in different clusters, which is perfect for the KNN model. Finally, it brought very encouraging
evaluation results during our first test, compared to other models, deciding us to keep it in our program.

We also decided to use a Decision Tree model because it is popular and applicable to a classification
problem. The first advantage of this algorithm is that easy to understand. In addition, it is easy to
interpret the results and be presented to non-technical people. Finally, the last advantage is that the
Decision Tree is a simple algorithm that is not costly in terms of computation time. The main drawback is
the risk of over-learning or overfitting. This is when the algorithm learns training data so accurately that
it fails to generalize a satisfactory result to new data. In a Decision Tree, it also becomes easy to
understand what are the audio characteristics that we must pay special attention to for creating hit
songs.

Finally, the last model we want to use is the Random Forest, it is a method where we combine our
prediction results together and use them for classification voting. This model combines multiple random
decision trees outputs and generates an outcome based on the majority voted results for our problem
(Hit or Non Hit). Thus this model is a direct improvement of the Decision Tree which corrects the lack of
robustness and the overfitting risk of the latter.

### Configurations tuning
In our situation, it is very difficult to know what a Top Song is. Even if we have the musical characteristics
and the total score of the most streamed and popular music over a long period of time; it is not possible
to classify this target with certainty. In addition, a classification of Top songs that is too demanding, risk
of rendering our models unusable for the next datasets.
Once our models have been developed, the strategy employed will therefore be to test them on
different scenarios. These are generated by gradually increasing the requirement of our classification
over the total of our total dataset. For example, we will use to build our models a dataset where the
target only includes the 100 most popular songs. Then the 500 first, the 750, 1000, 1500 and finally the
first 2000.

### Hyperparameters
![mod](https://user-images.githubusercontent.com/85830810/125612518-8b224c54-fa71-4bc9-a14d-fb0927ffec67.png)

### Outputs
In order to obtain a vision of the performance of our classification models, we decided to use a
confusion matrix. In order to illustrate the performance of the classifier based on the values True
Positive, True Negative, False Positive, False Negative. It is then possible to generate the Precision, Recall,
and F1-Score metrics.
In our study, the positive class corresponds to a song classification being a hit.

Metric results through the different Top outputs configuration :
![outputs](https://user-images.githubusercontent.com/85830810/125611426-1b5dda1b-2f8f-48c1-9ba8-eba8c468e1b4.png)

Decision Tree representation:
![dtout](https://user-images.githubusercontent.com/85830810/125611559-fce9c003-a230-42f4-830e-5f83b179dc8b.png)

### Conclusion
Following our strategy of using different Top configurations to see the ideal balance between the
restriction of the Top hits and the performance of our models. We can see that today we are only
starting to have a reliable model from the TOP 2500, which represents 73% of our training set. This result
is not yet satisfactory but we can see that we are on the right track and that with a larger amount of data
and new parameters we will undoubtedly be able to make our model more precise on the Top Songs
restriction.

## Model evolution
Our dataset labels are based on song popularity, which at this moment we are capturing through weekly
stream count charts from Spotify.
In our first review and iteration of the models, we will not automate the model evolution because we
believe it is important to do a manual review of the datasets we’re feeding for re-train.
At this first stage, the manual model review will be done between 5 to 15 weeks. This frequency has
been determined because:
● 5 weeks is the Median from the weeks variable from our initial dataset. This means that more
than 50% of our initial sample will be 5 weeks or less on a top chart.
● 15 weeks is the Average from the weeks variable from our initial dataset.
We acknowledge that the weeks variable does not mean consecutive weeks, which could’ve been a
better source of information to determine model re-evaluation frequency, but it is the best-informed
value we are willing to take to determine re-evaluation frequency.
![image](https://user-images.githubusercontent.com/85830810/126796084-76d971b9-000f-4b16-953f-9400389ec6c1.png)

### Scrap Charts
In our MVP (Minimum Viable Product) at the DSAA course deliverable, we have parsed KWORB charts,
using excel and a text editor. We will use python web-scrapping libraries to ease this task. That way we can upload the output directly
to our databricks instance for later analysis.
### Train Models
Model Training will be done in python using Apache Spark in Databricks.
In this first stage, Databricks notebooks will not be compiled into Jobs for automation. Manual
intervention from the Data Scientist will be needed to act on any manual data cleaning if needed.
### Trained Models
Once the models are fit and tested with multiple hyperparameter combinations. The Data scientist will
determine which ones are eligible for production deployment. Fit models will be serialized using Pickle,
and objects will be uploaded into AWS S3.
Get Metadata for Datasets
Python console programs that decorate songs and artists’ datasets are running in the local machine of
the Data Scientist.
As this is not recommended, we would need to industrialize the Python console programs, so that they
can run in the cloud.
As these programs are not performance-intensive, we could deploy them on top of:
1. AWS Lambda, so that we’re only charged on-demand.
2. AWS Elastic Container Services (ECS). Furthermore, since these console programs have
stop/resume processing time, we could add AWS Spot Instances to the ECS cluster to minimize
costs.

## Summary
- [DSAA_PredictingHitSong_Presentation.pptx](https://github.com/RomainBal/Spotify_Predict_Hit_Song/files/6815496/DSAA_PredictingHitSong_Presentation.pptx)
- [EAE_DSAA_PredictingHitSong.pdf](https://github.com/RomainBal/Spotify_Predict_Hit_Song/files/6815499/EAE_DSAA_PredictingHitSong.pdf)
