"""
Testing stuff goes here
"""

import FeatureExtractor as fe
import SentimentAnalysisModel as model
import numpy as np



seq_length = 200

featureExtractor = fe.FeatureExtractor(seq_length, "./Dataset/reviews.txt", "./Dataset/labels.txt")
sentimentModel = model.SentimentAnalysisModel(featureExtractor, seq_length)

sentimentModel.Train()
sentimentModel.Test()

#sentimentModel.Evaluate("this review is awesome")
#sentimentModel.Evaluate("i think the movie could be better missed opportunity")

#sentimentModel.run_model("this review is awesome")
        
        
print (1)