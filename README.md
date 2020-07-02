# Roman-urdu-Sentiment-Analyzer

### About the project
Understanding an overview of wider public opinion behind certain topics is essential despite languages in order to measure the rate of increase in product management. But finding an emotion of a comment based on a certain language has always been a challenge. This Web application focuses on finding emotions in a user defined Urdu comment and provides the result instantaenously.
 
### Development plan
 I used Support Vector Classifier to accomplish the classification of different emotions by training the data that was taken from https://archive.ics.uci.edu/ml/datasets/Roman+Urdu+Data+Set. Then I implemented Flask which is a microweb framework that intergates web app with python code. This goes along with stylesheets, html code and js files that were downloaded from https://materializecss.com/getting-started.html. There is also an Ipython notebook that speaks more about an in-depth construction and visualizaion of the model.  
 
### Challenges
1. During the traning and testing of data, Most of the classification algorithms showed around an average of 60% performance score. Even with hyperparameter optimization, there was only a slight increase in a few algorithms by a solid 7%. After many trial and errors and further comparisons with other models, I came with the decision to go with the highest performer by 67.2% (SVC).

2. It was time-taking to create a web application using Flask as I was new to the concepts. But I learnt a lot during the process 

### Special Thanks:
  I want to express my sincere gratitude to my mentor Lakshmana Nutulpati (https://www.linkedin.com/in/lakshmana-nutulapati-9b947664/) for his guidance and advice throughout the developement.
  
### Links
1. For Web templates with CSS,JS files go to: https://materializecss.com/getting-started.html
2. For the Data, I have already added the CSV file. But for more information on the data, go to: https://archive.ics.uci.edu/ml/datasets/Roman+Urdu+Data+Set for more information.

 

 
 
