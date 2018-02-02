"""
Testing stuff goes here
"""

import FeatureExtractor as fe
import SentimentAnalysisModel as model



a = 1
b =2

c = (a,b)

z1, z2 = c


seq_length = 200

featureExtractor = fe.FeatureExtractor(seq_length, "./Dataset/reviews.txt", "./Dataset/labels.txt")
sentimentModel = model.SentimentAnalysisModel(featureExtractor, seq_length)

sentimentModel.Evaluate("this review is awesome")
sentimentModel.Evaluate("i think the movie could be better missed opportunity")

#sentimentModel.run_model("this review is awesome")
        
        
print (1)