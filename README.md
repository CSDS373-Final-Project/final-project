# final-project
## Project Proposal

### Discussion of the Problem and Short Literature Review

In this project, we aim to develop and evaluate a machine learning model that accurately predicts how much a user will enjoy a song based on music they enjoyed in the past. We will frame this as a classification problem; the label the model is assigning is a rating from 1 (strongly disliked) to 5 (greatly enjoyed) to reflect the user’s likely response. 
The challenge of accurately recommending music has been explored extensively by major streaming platforms such as Spotify and Apple Music, which rely on machine learning models to guide listeners through their catalogs containing tens of millions of tracks. Without such models, users would face an overwhelming amount of choice, which makes it difficult to discover new music aligned with their tastes. Effective recommendation models not only personalize the listening experience and adapt as preferences evolve, but also create opportunities for emerging and independent artists to reach wider audiences.

Beyond entertainment, music recommendation models also have applications in therapeutic contexts. Recent research highlights the potential of applying music recommendation models to enhance music therapy sessions by tailoring song choices to reflect one’s emotional state, cultural background, music preferences, and therapy goals. One research article focuses on such clinical applications, particularly exploring music therapy for dementia patients. In this article, researchers developed models to classify songs by both genre and emotion. The researchers’ model had an audio detection element in which the model analyzed several smaller segments of the tracks. The researchers looked at two proposed solutions; however, the artificial convulational neural network (ACNN) achieved the greatest accuracy (approximately 83%) for predicting genre and emotion. Additionally, the study pointed out that while the model successfully recommends appropriate music, the oversight of a human remains essential for filtering out songs that may cause negative emotions in patients. 

Another approach researchers have considered emphasizes emotional awareness, but through behavioral indicators.  Unlike the first research article, which analyzes the audio of the track, this model predicts a user’s emotional state from their keyboard and mouse interactions, including details such as stroke frequency and press duration. These signals are then mapped to the user's emotions and listening/download history. Recommendations are then generated from these mappings. The evaluation showed that user-device interactions are indeed influenced by emotion. They also found recommendations were more accurate when users were in positive moods.	

A final article discusses limitations of “greedy” recommenders, which tend to recommend popular, familiar tracks. Using reinforcement learning, researchers created a model that used a multi-armed bandit approach. This approach has the advantage of being able to better balance novelty with personal preferences. By combining a linear function of user preferences with another novelty function that decreases with repeated exposure to the track, the model was able to improve recommendation while still maintaining novelty. This could inform how we build our model by trying to add a measure of novelty as the model recommends more and more tracks. 

	
### Description of the Data

#### Training Data

We will construct our training datasets by each selecting 100 songs that we already enjoy. This will be taken from our Apple Music libraries, which consist of 400 songs for Maxann and 1600 songs for Maddy. Each song will be rated on a five-point scale (1 = strongly disliked to 5 = greatly enjoyed), which serves as the label for our classification model. At minimum, each dataset will include songs rated 3 or higher, ensuring that the training data reflects music we find at least moderately enjoyable. As a stretch goal, we plan to create additional training datasets with greater variance (including songs with lower ratings or more diverse attributes) to test how the model performs when patterns are less obvious.

#### Attributes

We plan to include up to 16 attributes for each song in our datasets. However, we may apply feature selection later to refine the model and keep only the most influential attributes, therefore improving efficiency.  Most of our information is provided by tunebat (https://tunebat.com/). A detailed description of the attributes we intend to record is provided below.
	
#### Attribute List
name : The name of the song; we will use this to identify the song when we get to making the recommendation- not to predict the rating

artist : Performer of the song;  we will list the artist when making the recommendation and will start by including it as a potential predictor.

genre : We will describe the genre of a song by using Chosic. When searching a song, there is a “Genres on Spotify” tab. We will use the first genre listed in our data.

release_year: The year the song was released; we are deciding whether or not we want to generalize to a decade. For example, a song released in 2016 would have the release year 2010s.

explicit: This attribute records whether or not the song has explicit lyrics. We will record this as a binary variable (0=None and 1=Explicit, which is based on whether or not the song has the “E” tag on Apple Music).

key: The key the track is in; this data will be taken from tunebat.

bpm: The bpm (beats per minute) of the song; this data will also be taken from tunebat. 

length: The duration of the song; this data is taken from tunebat.

popularity: This is based on “the number and recency of track plays (out of 100)” - taken from tunebat.

energy: This is how “intense the track is based on general entropy, onset rate, timbre, perceived loudness, and range (out of 100)” - taken from tunebat

danceability: This is how “appropriate the song is for dancing based on overall regularity, beat strength, rhythm stability, and tempo (out of 100)” - taken from tunebat

happiness: This is how cheerful the track is (out of 100) - taken from tunebat

acousticness: This is the likelihood that the track is acoustic (out of 100)- taken from tunebat

speechiness: This is “how present spoken words are in the track (out of 100)” - taken from tunebat

loudness: This is “the average decibel amplitude across the track ranging from -60dB to 0dB”- taken from tunebat.

lyrics: This attribute describes whether or not lyrics are present in the song. We will record this as a binary variable (0= no lyrics, 1=lyrics present).

language: If present, the language of the song. If it is set to None, that means the song has no lyrics. 

#### Testing Data

Our testing dataset will include at least 100 songs and be drawn from the “featured reviews” section of RateYourMusic, providing a collection of songs outside of our personal Apple Music libraries. This will allow us to evaluate how effectively the model predicts ratings for music that is unfamiliar to us. We are still considering whether to assign our own ratings to these songs prior to running the model or to do so after predictions are generated, in order to minimize unnecessary preprocessing. Regardless of the approach, we will ensure that the testing dataset is structured consistently with the training dataset, with the same set of attributes recorded for each song.


### Proposed Solution
We will first construct both training and testing datasets as outlined in the Description of Data section. There will be at minimum a training and testing dataset for Maxann and another training and testing dataset for Maddy. If we choose to assign ratings to the songs prior to running our model, this will allow us to evaluate and refine the model in several ways. Our predictive model will be built using the RandomForestClassifier from Scikit-Learn’s ensemble module. As an initial structure, we will generate a forest of 100 decision trees, each constrained to a max depth of 15. We will use the Gini index as a measure for the quality of splits in the trees.

During the evaluation phase, we plan to experiment with adjusting the number of trees and maximum depth to optimize accuracy using hyper parameter tuning. Additional refinement techniques may include feature selection to balance predictive accuracy and efficiency, as well as min-max normalization to scale the data effectively. We will also compute an overall accuracy of the model. We will consider the model successful if it achieves an accuracy over 60% and if 0.50 is not included in the 95% confidence interval for accuracy. These goals for the model ensures that the model performs better than random chance and provides some statistically discernible evidence of meaningful accuracy.


### Outline and Description of Project Components

Our project will consist of three major components: data, model, and evaluation. Below, we have created an outline of our project components. 

#### Data

maxann_train.csv: This file includes 100 songs from Maxann’s library; as mentioned before, we may create more datasets with greater variance. However, to begin, we will create datasets with a moderate amount of variance.

maddy_train.csv: This file includes 100 songs from Maddy’s library.

maxann_test.csv: This file includes 100 songs taken from the featured tab on Rate Your Music. The songs are assigned a rating label in accordance to Maxann’s preferences.

maddy_test.csv: This file includes 100 songs taken from the featured tab on Rate Your Music. The songs are assigned a rating label in accordance to Maddy’s preferences.


#### Model

To begin, we will assign a number of trees and maximum depth based on the information we learned in homework 2. The model will be a RandomForestClassifier with 100 trees restricted to a maximum depth of 15. 

#### Evaluation 

In the evaluation stage, we will consider several questions about how we are configuring the model as a means to improve accuracy. Below are some questions we aim to answer while refining the model, as well as questions regarding the performance of our final configuration of the model.

Parameter Choices: What choice for maximum depth and number of trees yields the greatest accuracy? We also intend to have a visual of how accuracy changes across different parameter choices. 

Scaling Data: Does the model perform when we use min-max normalization to scale the dataset? Is the difference statistically discernable?

Other Training/Testing Data: How does the model do in terms of predictive accuracy when we use datasets with greater variance?

Chosen Attributes: Which attributes are the most important? What about least important? How does accuracy of the model compare when we increase the number of attributes? When is the difference statistically discernable? 

Final Configuration Questions: What maximum depth and number of trees did we end up using? Did we scale the data? What was the minimum accuracy observed and for what datasets? What about the maximum? What were the confidence intervals for the accuracy of these datasets? Was the accuracy >60% and did the confidence intervals include 0.50 as a prediction for the true accuracy of the model?

#### Description of Existing Progress

Aside from working on the project proposal and considering our approach to creating a music recommendation model, our progress is fairly limited. However, we have created a repository to start working in: https://github.com/CSDS373-Final-Project/final-project .

#### Timeline

We intend to split the project into several smaller goals starting with data processing, then creating our initial model, optimizing our initial model to get the best runtime possible, next evaluating our model, refining the model as needed, evaluating our final model, and finally creating other datasets to see how the final model changes when the data is more complex or has more conflicting information. Below, is a plan for when we want to finish these smaller goals:

November 20, 2025: Deadline to create training and testing datasets
November 31, 2025: Deadline to create and optimize initial model
December 12, 2025: Deadline to perform evaluation, refinement, and analysis
December 16, 2025: Deadline to create new datasets if time
December 18, 2025: Deadline to prepare presentation and report

