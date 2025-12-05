# Final-project
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

## Project Update

### Progress So Far
Our work for the project these last couple weeks have been focused on compiling training data, troubleshooting and eventually finding a useful testing dataset, and beginning the process of transforming our testing dataset to mirror our training dataset and make it more useful for testing our model. As we started working on the project and received feedback, we have also updated our approach to working on this problem.
First, when we started compiling our training data, we started by creating a comprehensive list of songs that only included tracks that we rated a 4 or 5. Initially, we thought that we would want to train our model using good examples of what a 4 or 5 looks like; however, as we progressed, we realized that including songs with lower ratings would be good to show what “bad” songs look like. We were also able to successfully gather all the information we wished to include for each track in our training data so far. The revised list of attributes we are using include: label (rating), artists, track_name, popularity, duration (in seconds), explicit, danceability, energy, loudness, speechiness, acousticness, tempo, and genre. We retrieved this information using data from Tunebat and Chosic.  All of the training data is currently separated into two Google Sheets, one for each of our training data. Once completed, they will be added to our  GitHub repository. As we started making progress on our training datasets, we realized that manually entering song data is a time consuming, tedious task. While it works for creating smaller data sets, we want our model to be able to pull recommendations from a very large dataset. In hindsight, we realize it would have been more efficient to identify pre-existing data for our testing data first so we could model our training data after it. However, we were able to find and modify a dataset found on Kaggle. Using Google Sheets, we modified the data to only include attributes relevant to our project. This refined dataset is uploaded on Github under the name testing_set_dataset.csv. There is still some work to be done with modifying the dataset. We have started making those modifications using pandas. We have already/plan to modify the following in Python:
Using one-hot encodings to create dummy variables for the explicit category- the data currently labels it as “TRUE” or “FALSE”
Converting song duration from milliseconds to seconds to allow for greater human readability
Scaling the speechiness, danceability, acousticness, and energy attributes so they range from 0-100 (as seen on tunebat) instead of 0-1, and converting them to integers so they are consistent with our training data.
Converting loudness and tempo values into integers to also match our training data
Once these transformations are done, the script will save the results into a new CSV file, which we will use as testing for our model. 
Beyond progress that we have made on the project, we have also considered your feedback and decided to reframe our problem as a regression problem rather than a classification problem. This is definitely the more intuitive option as we don’t want to penalize the model for, say , assigning a song a 1 instead of a 5 the same way we’d penalize a model for assigning a song a 4 instead of a 5. In the case of this problem, the label the model ended up choosing matters beyond simply whether or not it predicted the correct label. Finally, we also want to consider models other than a Random Forest to see if one model works better for this problem than another as it poses a more interesting question than our initial one. 

### Challenges
One of the earliest challenges we faced was when developing our training dataset. We encountered an issue with genre identification; the genres provided by Chosic for our selected songs were often too specific to be useful when attempting to generalize and identify larger patterns across songs. For example, one song was categorized under a highly specific genre, “Texas Pop Punk”. The level of detail in certain genre listings can make it difficult to align it with songs in other datasets. To fix this, we simplified our genre labels into broader categories. In this example, we shortened “Texas Pop Punk” to “Pop Punk” so that it would better align with similar songs across datasets. This required us to intervene and use our own judgement for genre classification; however, we believe this was useful for generalization purposes. 
Another challenge with genre identification was considering global music groups. For instance, the group Katseye releases songs in both Korean and English. While their songs are composed of majority English lyrics, they promote as a K-pop group. This raised a question of whether we want to classify the group as Pop and K-pop. The decision was somewhat important because we had songs from the group on both of our spreadsheets, so we wanted them to be consistent with one another. We ultimately decided to label the group’s genre by their cultural classification, K-pop, despite their lyrics containing more English than Korean.
 Beyond genre classification, we also encountered difficulties with songs including multiple collaborators. Our initial plan was to treat the contents of the artist column as a single artist. However, we realized that some of our songs featured more than one artist. If we treated combinations of artists as a unique artist, the model would again struggle to generalize since collaborations are usually one-off events. To avoid this issue, we decided to include only the primary artist listed on each track. While we do lose some detail, it gives us a more generalizable dataset that doesn’t include rare, non-generalizable combinations of artists. 
A final challenge that we are currently facing is the question of what should be the output of our assignment. It would be unrealistic for us to manually listen to a large number of songs in the testing dataset (as it includes 114,000 songs), which made mean absolute error as an initial output for our model impractical. We have started deciding on an alternative approach; however, we are open to suggestions on what our output should be. Our plan is to generate playlists of a user-specified size by randomly selecting songs rated a 5. If we run out of songs the model predicted to be a 5, we will then draw from songs rated a 4 and so on. We can listen to these generated playlists, assign our own ratings, and calculate mean absolute error based on the difference of our ratings. One concern with this, however, is bias. Another approach to our playlist idea could be to create a mixed playlist (containing songs of all ratings). We can run our model on the other’s training set (meaning Maddy would run a model using Maxann’s training data and Maxann would run the model using Maddy’s data), and give the playlist to the other to evaluate. To further clarify our thoughts on the output. Our initial output would be a table containing song, artist, and rating. We would then give each other the output without the rating, assign ratings ourselves, and calculate mean absolute error based on what we observe. 

### What Has Gone Well

Despite the challenges listed above, we are happy with the progress we have made in preparing our datasets. Specifically, being able to identify a testing data that aligns with the goal of our model that can also be reasonably transformed to fit a structure that makes sense for our project was definitely an encouraging moment with the progress we have made so far. Additionally, accessing consistent data/measures for our training sets has been straightforward. We feel confident going into the implementation stage of our project as we have a strong idea on how we want to go about our implementation. 

### Revised Timeline
December 7, 2025: Finish training data and script that transforms testing data
December 12, 2025: Deadline to create initial model
December 16, 2025: Deadline to optimize model/get our outputs
December 17, 2025: Work on presentation and report

### Questions
The biggest question we have moving forward is regarding our output. Given that our testing set is 114,000 songs, it would be impractical to manually listen to tracks, which makes mean absolute error difficult to use. Do you have any suggestions on what we could use as the output for our model that balances practicality with a meaningful, rigorous output. 
