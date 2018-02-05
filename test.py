"""
Testing stuff goes here
"""

import FeatureExtractor as fe
import SentimentAnalysisModel as model
import numpy as np



seq_length = 200

featureExtractor = fe.FeatureExtractor(seq_length, "./Dataset/reviews.txt", "./Dataset/labels.txt")
sentimentModel = model.SentimentAnalysisModel(featureExtractor, seq_length)

#sentimentModel.Train()
#sentimentModel.Test()


sentimentModel.load_model(None)
print ("Test:")
print ("-------------")
texts = ["this movie is excellent",
         "the film is terrible",
         "i thought the film needed to be better",
         "the film was very slow and boring",
         "this is an excellent movie, I highly recommend it",
         "i think the actors did a phenomenal job here and they get all the credit.",
         "the film was all action and no substance avoid this movie",
         "I recommend this film",
         "I do not recommend this film",
         "I do not recommend this film at all",
         "I do not recommend this film right now",
         "this is a good movie",
         "this is a bad movie",
         "this movie is good",
         "this movie is not good"

]

for text_item in texts:
    score = sentimentModel.Evaluate(text_item)    
    prediction = "NEGATIVE"
    if score > 0.5:
        prediction = "POSITIVE"
            
    print("{} -- {:.2f} -- {}".format(text_item, score, prediction))


print("-----------")
print("Done")
