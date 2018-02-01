"""
Testing stuff goes here
"""

import FeatureExtractor as fe
import SentimentAnalysisModel as model



seq_length = 200

featureExtractor = fe.FeatureExtractor(seq_length, "./Dataset/reviews.txt", "./Dataset/labels.txt")
sentimentModel = model.SentimentAnalysisModel(featureExtractor, seq_length)

sentimentModel.Train2()
        
        
print (1)