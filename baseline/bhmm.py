from codecs import open
from random import randint, uniform
from collections import defaultdict
import numpy as np

from utility import change_count
from utility import get_value


class BHMM(object):
    """ Bayesian Hidden Markov Model with Gibbs sampling. """

    def __init__(self, args):
        # Input file
        self.fin = args.input
        # Output file
        self.fout = args.output
        # Number of possible labels
        self.labels = args.labels
        # Number of sampling iterations
        self.iterations = args.iterations
        # self.transition hyperparameter
        self.alpha = args.alpha
        # self.emission hyperparameter
        self.beta = args.beta
        # Lists of observations
        self.data = []
        # Uniform distribution of observations
        self.frequencies = defaultdict(float)
        # Delimits single observation
        self.delimiter = " "
        # Emission matrix: C(previous label, label)
        self.emission = defaultdict(int)
        # Transition Matrix: C(label, emission)
        self.transition = defaultdict(int)
        # Lists labels sequences
        self.sequences = []
        # Base probability
        self.base_probability = 1.0 / self.labels
        self.stemlist=list()
        self.stemdict=defaultdict(int)


    def __read_data(self):
        """ Creates a uniform distribution. """
        print ("Reading corpus")

        with open(self.fin, encoding="utf8") as f:
            for line in f:
                # Sequence of observations
                unit = ["START"]
                unit.append("START")
                self.stemlist.append("START")
                for item in line.split(self.delimiter):
                    item = item.strip()
                    unit.append(item)
                    self.frequencies[item] += 1.0
                    self.stemlist.append(item)
                unit.append("END")
                unit.append("END")
                self.stemlist.append("END")
                self.data.append(unit)
                

    def __create_frequencies(self):
        """ Calculate relative frequency. """
        print ("Creating frequencies")
        # Total number of observations
        total = sum(self.frequencies.values())

        for key in self.frequencies.keys():
            self.frequencies[key] /= total

    def __create_matrixes(self):
        """ Creates transition and emission matrix. """
        print ("Creating matrixes")

        for unit in self.data:
            # Ordered list of hidden labels framed by -1
            sequence = [-1,-1 ]

            for observation in unit[2:-2]:
                # Assign random label to observation
                label = randint(0, self.labels - 1)
                # Add C(label|previous label)
                change_count(self.transition, label, sequence[-2], sequence[-1], 1)
#                print(label, sequence[-2], sequence[-1])
                # Add C(emission|label)
                change_count(self.emission, observation, label, 1)
                sequence.append(label)
                # Last transition add C(-1|previous label)
            change_count(self.transition, "-1", sequence[-2],sequence[-1], 1)
#            print( "-1", sequence[-2],sequence[-1])
            sequence.append(-1)
            change_count(self.transition, "-1", sequence[-2],sequence[-1], 1)
#            print( "-1", sequence[-2],sequence[-1])
            sequence.append(-1)

#            print(len(sequence),len(unit))
            # Add sequence of observations list of sequences
            self.sequences.append(sequence)
        for stemm in self.stemlist:
            change_count(self.stemdict, stemm, 1)
#            print(unit)
#            print(sequence)


    def __initialize_model(self):
        """ Initializes the HMM """
        print ("Initializing model")
        self.__read_data()
        print ("Corpus read")
        self.__create_frequencies()
        print ("Frequencies created")

        self.__create_matrixes()
        print ("Matrixes created")

    def __compute_probabilityEmission(self, matrix, items, hyper):
        """ Calculating posterior.
        
        Arguments:
        matrix -- transition or emission
        items -- (hypothesis, evidence)
        base -- base probability
        hyper -- hyperparameter
        """
        x = get_value(matrix, items[0], items[1])
        y = get_value(matrix, items[1])

        return (x +pow((1/29),len(items[0]))*hyper) / (y + hyper)
    
    
    def __compute_probability1(self, matrix, items, base, hyper):
        """ Calculating posterior.
        
        Arguments:
        matrix -- transition or emission
        items -- (hypothesis, evidence)
        base -- base probability
        hyper -- hyperparameter
        """
        x = get_value(matrix, items[0], items[1])
        y = get_value(matrix, items[1])

        return (x +hyper) / (y + base *  hyper)
    
    
    
    def __compute_probability(self, matrix, items, base, hyper):
        """ Calculating posterior.
        
        Arguments:
        matrix -- transition or emission
        items -- (hypothesis, evidence)
        base -- base probability
        hyper -- hyperparameter
        """
        x = get_value(matrix, items[0], items[1], items[2])
        y = get_value(matrix, items[1], items[2])

        return (x +hyper) / (y + base *  hyper)
        
    def __compute_probabilityCheck(self, matrix, items, base, hyper,check3,check2):
        """ Calculating posterior.
        
        Arguments:
        matrix -- transition or emission
        items -- (hypothesis, evidence)
        base -- base probability
        hyper -- hyperparameter
        """
        x = get_value(matrix, items[0], items[1], items[2])
        y = get_value(matrix, items[1], items[2])
        
        return (x + check3+ hyper) / (y +check2+ base *hyper)
    
    def __compute_probabilityCheck1(self, matrix, items, base, hyper,blanketNew):
        """ Calculating posterior.
        
        Arguments:
        matrix -- transition or emission
        items -- (hypothesis, evidence)
        base -- base probability
        hyper -- hyperparameter
        """
        x = get_value(matrix, items[0], items[1], items[2])
        y = get_value(matrix, items[1], items[2])
        
        return (x + self.__check41(blanketNew)+ hyper) / (y +self.__check31(blanketNew)+ base *hyper)
    
    def __compute_probabilityCheck2(self, matrix, items, base, hyper,blanketNew):
        """ Calculating posterior.
        
        Arguments:
        matrix -- transition or emission
        items -- (hypothesis, evidence)
        base -- base probability
        hyper -- hyperparameter
        """
        x = get_value(matrix, items[0], items[1], items[2])
        y = get_value(matrix, items[1], items[2])
        
        return (x + self.__check21(blanketNew)+self.__check42(blanketNew)+ hyper) / (y +self.__check22(blanketNew)+self.__check32(blanketNew)+ base *hyper)
    def __check31(self,blanketNew):
       label,previous_previous_label,previous_label, following_label,following_following_label=blanketNew
       if label==previous_label and label==previous_previous_label:
           return 1
       else:
            return 0
    def __check32(self,blanketNew):
       label,previous_previous_label,previous_label, following_label,following_following_label=blanketNew
       if label==previous_label and label==following_label:
           return 1
       else:
            return 0
        
    def __check41(self,blanketNew):
       label,previous_previous_label,previous_label, following_label,following_following_label=blanketNew
       if label==previous_label and label==following_label and label==previous_previous_label :
           return 1
       else:
            return 0
    
    def __check42(self,blanketNew):
       label,previous_previous_label,previous_label, following_label,following_following_label=blanketNew
       if label==previous_label and label==following_label and label==following_following_label :
           return 1
       else:
            return 0
            
    def __check21(self,blanketNew):
       label,previous_previous_label,previous_label, following_label,following_following_label=blanketNew
       if label==previous_previous_label and label==following_following_label and previous_label==following_label :
           return 1
       else:
            return 0
    def __check22(self,blanketNew):
       label,previous_previous_label,previous_label, following_label,following_following_label=blanketNew
       if label==previous_previous_label and previous_label==following_label :
           return 1
       else:
            return 0

    def __sample_label(self, probabilities):
        """ Sample label.
        
        Arguments:
        probabilities -- probabilities of all labels
        """
#        z = sum(probabilities)
#        remaining = uniform(0, z)
#
#        for i in range(len(probabilities)):
#            probability= probabilities[i]
#            remaining -= probability
#            if remaining <= 0:
#                return  i

        probabilities=np.cumsum(probabilities)
        probabilities=probabilities/probabilities[-1]
        randomNumber=np.random.rand()
        for i in range(len(probabilities)):
            probability= probabilities[i]
            if randomNumber<probability:
                return i



    def __compute_label_probabilities(self, blanket):
        """ Computes the probability of each label.
        
        Arguments:
        blanket -- Markov blanket
        """
        _, previous_previous_label,previous_label, following_label,following_following_label, current_observation = blanket
        # Probabilities of each possible label
        probabilities = []

        for label in range(self.labels):
            blanketNew=label,previous_previous_label,previous_label, following_label,following_following_label
            # Chain rule
            probability = (self.__compute_probability(self.transition,
                                                      (label,previous_previous_label, previous_label),
                                                      self.labels,
                                                      self.alpha) *
                           self.__compute_probabilityCheck1(self.transition,
                                                      (following_label,previous_label, label),
                                                      self.labels,
                                                      self.alpha,blanketNew) *
                           self.__compute_probabilityCheck2(self.transition,
                                                      (following_following_label, label,following_label),
                                                      self.labels,
                                                      self.alpha,blanketNew) *
                           self.__compute_probability1(self.emission,
                                                       (current_observation, label),
                                                      get_value(self.stemdict, current_observation),
                                                      self.beta))
                           
            probabilities.append(probability)
        return probabilities

    def __write_labeled_data(self):
        """ Writes labeled data to output file. """
        print ("Writing data")

        with open(self.fout, "w", encoding="utf8") as f:
            for i in range(len(self.data)):
                labeled_unit = []

                for j in range(len(self.sequences[i])):
                    labeled_unit.append("%s/%s" % (self.data[i][j],
                                                   self.sequences[i][j]))
                f.write("%s\n" % " ".join(labeled_unit[2:-2]))

   

    def __change_sample(self, blanket, i):
        """ Adds (i = 1) or removes (i = -1) a sample.
        
        Arguments:
        blanket -- affected labels
        i -- add or remove
        """
        current_label,previous_previous_label, previous_label, following_label, following_following_label,current_observation = blanket

        change_count(self.transition,current_label,previous_previous_label,   previous_label,i)
        change_count(self.transition, following_label,  previous_label,current_label, i)
        change_count(self.transition, following_following_label,current_label,following_label,  i)
        change_count(self.emission, current_observation, current_label, i)
        change_count(self.stemdict, current_observation, i)

    def run(self):
        """ Gibbs sampling. """
        self.__initialize_model()
        print ("Model initialized\nStarting iterations\n")

        for _ in range(self.iterations):

            for i, sequence in enumerate(self.sequences):
                for j in range(2, len(self.sequences[i]) - 2):
                    #print(len(sequence),len(self.data[i]))
                    # Markov blanket affected by changing label
                    blanket = [sequence[j],
                               sequence[j - 2],
                               sequence[j - 1],
                               sequence[j + 1],
                               sequence[j + 2],
                               self.data[i][j]]
                    # Remove sample
                    self.__change_sample(blanket, -1)
                    # Probabilities of each label
                    probabilities = self.__compute_label_probabilities(blanket)
                    # Sample current label
                    sequence[j] = self.__sample_label(probabilities)
                    # Update blanket
                    blanket[0] = sequence[j]
                    # Add sample
                    self.__change_sample(blanket, 1)
            print ("Iteration %s" % (_ + 1))
        print ("\nIterations finished")
        self.__write_labeled_data()
        print ("Data written")