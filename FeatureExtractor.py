'''
Extract files and convert them to features.
'''
from string import punctuation
from collections import Counter
import numpy as np



class FeatureExtractor:
    '''
    stuff
    '''
    def __init__(self, max_sequence_length, trainingDataPath, labelDataPath):
        self.RawLabels = np.array([])
        self.RawTrainingData = []
        self.Features = []
        self.ExpectedOutputs = np.array([])
        self.Max_Seq_length = max_sequence_length
        self.TrainingDataPath = trainingDataPath
        self.LabelDataPath = labelDataPath
        self.Label_Vocab_To_Int = []
        self.TrainingData_Vocab_To_Int = []

    def __remove_punctuation(self, text):
        '''
        Removes all punctuations from text
        '''
        all_text = [c for c in text if c not in punctuation]
        all_text = "".join(all_text)

        return all_text

    def __get_unique_words(self, text):
        '''
        Returns a list of unique words in text
        '''
        words = text.split()
        counts = Counter(words)
        vocab = sorted(counts, key=counts.get, reverse=True)

        return vocab

    def __encode_words(self, text_array, word_to_int, all_text_has_one_word):
        '''
        Takes a list of text and converts all words to integers
        '''
        result = []
        for text in text_array:
            encoded_text = [word_to_int[word] for word in text.split()]
            if all_text_has_one_word:
                encoded_text = encoded_text[0]

            result.append(encoded_text)

        if all_text_has_one_word:
            result = np.array(result)

        return result

    def __add_left_zero_padding(self, encoded_text_array):
        array_height = len(encoded_text_array)
        result = np.zeros((array_height, self.Max_Seq_length), dtype=int)

        for i, encoded_text in enumerate(encoded_text_array):
            truncated = np.array(encoded_text)[:self.Max_Seq_length]
            result[i, -len(truncated):] = truncated

        return result


    def __load_files(self, trainingDataPath, labelDataPath):
        '''
        Load text files
        '''

        trainingDataFile = open(trainingDataPath, 'r')
        trainingData = trainingDataFile.read()
        trainingData = self.__remove_punctuation(trainingData)
        trainingData = trainingData.split("\n")

        labelFile = open(labelDataPath, 'r')
        labels = labelFile.read()
        labels = labels.split("\n")

        #Removing lines that are invalid with 0 review length
        non_zero_idx = [index for index, data in enumerate(trainingData) if len(data.strip()) != 0]
        trainingData = [trainingData[index] for index in non_zero_idx]
        labels = [labels[index] for index in non_zero_idx]

        self.RawLabels = labels
        self.RawTrainingData = trainingData

    def ExtractFeatures(self):
        '''
        stuff
        '''
        self.__load_files(self.TrainingDataPath, self.LabelDataPath)

        all_text = " ".join(self.RawTrainingData)
        words = self.__get_unique_words(all_text)
        vocab_to_int = {word: index for index, word in enumerate(words, 1)}
        encodedTrainingData = self.__encode_words(self.RawTrainingData, vocab_to_int, False)
        encodedTrainingData = self.__add_left_zero_padding(encodedTrainingData)
        self.Features = encodedTrainingData
        self.TrainingData_Vocab_To_Int = vocab_to_int

        vocab_to_int = {"positive":1, "negative":0}
        encodedLabels = self.__encode_words(self.RawLabels, vocab_to_int, True)
        self.ExpectedOutputs = encodedLabels
        self.Label_Vocab_To_Int = vocab_to_int
