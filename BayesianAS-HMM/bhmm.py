from codecs import open
from random import randint, uniform, random
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
        # self.emission hyperparameter
        self.gama = args.gama
        # self.emission hyperparameter
        self.delta = args.delta
        # Lists of observations
        self.data = []
        # Lists of stems
        self.stems = []
        # Lists of stems
        self.affixs = []
        # Uniform distribution of observations
        self.frequencies = defaultdict(float)
        # Uniform distribution of stems
        self.stemFrequencies = defaultdict(float)
        # Uniform distribution of affix
        self.affixFrequencies = defaultdict(float)
        # Delimits single observation
        self.delimiter = " "
        # Emission matrix: C(previous label, label)
        self.emission = defaultdict(int)
        # Transition Matrix: C(label, emission)
        self.transition = defaultdict(int)
        # Emission matrix: C(label,stem)
        self.stemEmission = defaultdict(int)
        # Emission matrix: C(label,affix)
        self.affixEmission = defaultdict(int)
        self.affixTransition = defaultdict(int)
        # Lists labels sequences
        self.sequences = []
        # Base probability
        self.base_probability = 1.0 / self.labels
        self.wordcount=0
        self.stemlist=list()


    def __read_data(self):
        """ Creates a uniform distribution. """
        print ("Reading corpus")

        with open(self.fin, encoding="utf8") as f:
            for line in f:
                # Sequence of observations
                unit = ["START"]
                unit.append("START")
                unitStem = ["START"]
                unitStem.append("START")
                unitAffix = ["START"]
                unitAffix.append("START")
                self.stemlist.append("START")
                self.wordcount+=1
                for item in line.split(self.delimiter):
                    self.wordcount+=1
                    item = item.strip().lower()
                    unit.append(item)
                    self.frequencies[item] += 1.0
                    if len(item)>2:
                       x=np.random.randint(3,len(item)+1)
                       if item[x::]=="":
                            affix="#"
                            stem=item
                       else:
                            affix=item[x::]
                            stem=item[:x]
                       unitStem.append(stem)
                       self.stemlist.append(stem)
                       unitAffix.append(affix)
                       self.stemFrequencies[stem]+=1.0
                       self.affixFrequencies[affix]+=1.0
                    else:
                       unitStem.append(item)
                       self.stemlist.append(item)
                       unitAffix.append("#")
                       self.stemFrequencies[item]+=1.0
                       self.affixFrequencies["#"]+=1.0
                unit.append("END")
                unit.append("END")
                unitStem.append("END")
                unitStem.append("END")
                unitAffix.append("END")
                unitAffix.append("END")
                self.stemlist.append("END")
                self.wordcount+=1
                self.stems.append(unitStem)
                self.affixs.append(unitAffix)
                self.data.append(unit)


    def __create_frequencies(self):
        """ Calculate relative frequency. """
        print ("Creating frequencies")
        # Total number of observations
        total = sum(self.frequencies.values())
        totalStem = sum(self.stemFrequencies.values())
        totalAffix = sum(self.affixFrequencies.values())
        
        for key in self.stemFrequencies.keys():
            self.stemFrequencies[key] /= totalStem
        
        for key in self.affixFrequencies.keys():
            self.affixFrequencies[key] /= totalAffix

        for key in self.frequencies.keys():
            self.frequencies[key] /= total

    def __create_matrixes(self):
        """ Creates transition and emission matrix. """
        print ("Creating matrixes")

        for i in range(len(self.data)):
            unit=self.data[i]
            unitStem=self.stems[i]
            unitAffix=self.affixs[i]
            # Ordered list of hidden labels framed by -1
            sequence = [-1,-1]
            
            for aa in range(2,len(unitStem)-2):
                item=unit[aa]
                stem=unitStem[aa]
                affix=unitAffix[aa]
                # Assign random label to observation
                label = randint(0, self.labels - 1)
                # Add C(label|previous label)
                change_count(self.transition, label, sequence[-2], sequence[-1], 1)
                change_count(self.affixTransition, affix, unitAffix[aa-2],unitAffix[aa-1], 1)
#                print(label, sequence[-2], sequence[-1])
                # Add C(emission|label)
                change_count(self.emission, item, label, 1)
                change_count(self.stemEmission, stem, label, 1)
                change_count(self.affixEmission, affix, label, 1)
                sequence.append(label)
                # Last transition add C(-1|previous label)
            change_count(self.transition, "-1", sequence[-2],sequence[-1], 1)
            change_count(self.affixTransition, affix, unitAffix[-4],unitAffix[-3], 1)
            change_count(self.affixTransition, affix, unitAffix[-2],unitAffix[-1], 1)
#            print( "-1", sequence[-2],sequence[-1])
            sequence.append(-1)
            change_count(self.transition, "-1", sequence[-2],sequence[-1], 1)
#            print( "-1", sequence[-2],sequence[-1])
            sequence.append(-1)
            # Add sequence of observations list of sequences
            self.sequences.append(sequence)

    def __initialize_model(self):
        """ Initializes the HMM """
        print ("Initializing model")
        self.__read_data()
        print ("Corpus read")
        self.__create_frequencies()
        print ("Frequencies created")
        self.__create_matrixes()
        print ("Matrixes created")

		
    def __compute_probabilityUnigram(self, matrix, item, hyper):
        """ Calculating posterior.
        
        Arguments:
        matrix -- transition or emission
        items -- (hypothesis, evidence)
        base -- base probability
        hyper -- hyperparameter
        """
        y = self.wordcount
        x = get_value(matrix, item)
        z=len(set(self.stemlist))
#        print(x,y,base)
        return (x +hyper) / (y + z*hyper)
    
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

    
    def __sample_label(self, probabilities,tagProb,stemProb,affixProb):
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
#                return  tagProb[i],stemProb[i]
        probabilities=np.cumsum(probabilities)
        probabilities=probabilities/probabilities[-1]
        randomNumber=np.random.rand()
        for i in range(len(probabilities)):
            probability= probabilities[i]
            if randomNumber<probability:
                return  tagProb[i],stemProb[i],affixProb[i]
         

    def __compute_label_probabilities(self, blanket):
        """ Computes the probability of each label.
        
        Arguments:
        blanket -- Markov blanket
        """
        
        current_label,previous_previous_label, previous_label, following_label, following_following_label,current_observation,current_stem,current_affix,previous_previous_affix, previous_affix, following_affix, following_following_affix = blanket
        # Probabilities of each possible label
        probabilities = []
        tagProb=[]
        stemProb=[]
        affixProb=[]
                
        affix="#"
        stem=current_observation
        if len(current_observation)>2:
            
            for label in range(self.labels):
                blanketNewLabel=label,previous_previous_label,previous_label, following_label,following_following_label
                blanketNewAffix=current_affix,previous_previous_affix,previous_affix, following_affix,following_following_affix

                for i in range(3,len(current_observation)+1):                    
                    if current_observation[i::]!="":
                        affix=current_observation[i::]
                        stem=current_observation[:i]
                        
                        # Chain rule
                    probability = (self.__compute_probability(self.transition,
                                                      (label,previous_previous_label, previous_label),
                                                      self.labels,
                                                      self.alpha) *
                           self.__compute_probabilityCheck1(self.transition,
                                                      (following_label,previous_label, label),
                                                      self.labels,
                                                      self.alpha,blanketNewLabel) *
                           self.__compute_probabilityCheck2(self.transition,
                                                      (following_following_label, label,following_label),
                                                      self.labels,
                                                      self.alpha,blanketNewLabel) *
                            self.__compute_probability(self.affixTransition,
                                                      (affix,previous_previous_affix, previous_affix),
                                                      self.labels,
                                                      self.alpha) *
                           self.__compute_probabilityCheck1(self.affixTransition,
                                                      (following_affix,previous_affix, affix),
                                                      self.labels,
                                                      self.alpha,blanketNewAffix) *
                           self.__compute_probabilityCheck2(self.affixTransition,
                                                      (following_following_affix, affix,following_affix),
                                                      self.labels,
                                                      self.alpha,blanketNewAffix) *
                           self.__compute_probability1(self.affixEmission,
                                                      (affix, label),
                                                      self.affixFrequencies[affix],
                                                      self.delta)*
                           self.__compute_probabilityUnigram(self.stemEmission,
                                                      stem,
                                                      self.beta))
                    probabilities.append(probability)
                    tagProb.append(label)
                    stemProb.append(stem)
                    affixProb.append(affix)
#            return probabilities,tagProb,stemProb
        else:

            for label in range(self.labels):
                blanketNewLabel=label,previous_previous_label,previous_label, following_label,following_following_label
                blanketNewAffix=current_affix,previous_previous_affix,previous_affix, following_affix,following_following_affix
                        
                # Chain rule
                probability = (self.__compute_probability(self.transition,
                                                      (label,previous_previous_label, previous_label),
                                                      self.labels,
                                                      self.alpha) *
                           self.__compute_probabilityCheck1(self.transition,
                                                      (following_label,previous_label, label),
                                                      self.labels,
                                                      self.alpha,blanketNewLabel) *
                           self.__compute_probabilityCheck2(self.transition,
                                                      (following_following_label, label,following_label),
                                                      self.labels,
                                                      self.alpha,blanketNewLabel) *
                            self.__compute_probability(self.affixTransition,
                                                      (affix,previous_previous_affix, previous_affix),
                                                      self.labels,
                                                      self.alpha) *
                           self.__compute_probabilityCheck1(self.affixTransition,
                                                      (following_affix,previous_affix, affix),
                                                      self.labels,
                                                      self.alpha,blanketNewAffix) *
                           self.__compute_probabilityCheck2(self.affixTransition,
                                                      (following_following_affix, affix,following_affix),
                                                      self.labels,
                                                      self.alpha,blanketNewAffix) *
                            self.__compute_probability1(self.affixEmission,
                                                      (affix, label),
                                                      self.affixFrequencies[affix],
                                                      self.delta)*
                           self.__compute_probabilityUnigram(self.stemEmission,
                                                      stem,
                                                      self.beta))
                probabilities.append(probability)
                tagProb.append(label)
                stemProb.append(stem)
                affixProb.append(affix)
#        print(probabilities,tagProb,stemProb,affixProb)
        return probabilities,tagProb,stemProb,affixProb

    def __write_labeled_data(self):
        """ Writes labeled data to output file. """
        print ("Writing data")

        with open(self.fout, "w", encoding="utf8") as f:
            for i in range(len(self.data)):
                labeled_unit = []

                for j in range(len(self.sequences[i])):
                    labeled_unit.append("%s&%s/%s" % (self.data[i][j],self.stems[i][j],
                                                   self.sequences[i][j]))
                f.write("%s\n" % " ".join(labeled_unit[2:-2]))


    def __change_sample(self, blanket, i):
        """ Adds (i = 1) or removes (i = -1) a sample.
        
        Arguments:
        blanket -- affected labels
        i -- add or remove
        """
        current_label,previous_previous_label, previous_label, following_label, following_following_label,current_observation,current_stem,current_affix,previous_previous_affix, previous_affix, following_affix, following_following_affix = blanket
        
        change_count(self.transition,current_label,previous_previous_label,   previous_label,i)
        change_count(self.transition, following_label,  previous_label,current_label, i)
        change_count(self.transition, following_following_label,current_label,following_label,  i)
        
        change_count(self.affixTransition,current_affix,previous_previous_affix,   previous_affix,i)
        change_count(self.affixTransition, following_affix,  previous_affix,current_affix, i)
        change_count(self.affixTransition, following_following_affix,current_affix,following_affix,  i)
        
        change_count(self.emission, current_observation, current_label, i)
        change_count(self.emission, current_affix, current_label, i)
        change_count(self.stemEmission, current_stem, current_label, i)

    def run(self):
        """ Gibbs sampling. """
        self.__initialize_model()
        print ("Model initialized\nStarting iterations\n")

        for _ in range(self.iterations):

            for k in range(len(self.sequences)):
                sequence=self.sequences[k]
                for j in range(2, len(self.sequences[k]) - 3):
                    # Markov blanket affected by changing label
                    blanket = [sequence[j],
                               sequence[j - 2],
                               sequence[j - 1],
                               sequence[j + 1],
                               sequence[j + 2],
                               self.data[k][j],
                               self.stems[k][j],
                               self.affixs[k][j],
                               self.affixs[k][j-2],
                               self.affixs[k][j-1],
                               self.affixs[k][j+1],
                               self.affixs[k][j+2]]
         
                    # Remove sample
                    self.stemlist.remove(self.stems[k][j])
                    self.__change_sample(blanket, -1)
                    # Probabilities of each label
                    probabilities,tagProbb,stemProbb,affixProbb = self.__compute_label_probabilities(blanket)
                    # Sample current label
                    sequence[j],self.stems[k][j],self.affixs[k][j] = self.__sample_label(probabilities,tagProbb,stemProbb,affixProbb)
                    # Update blanket
                    blanket[0] = sequence[j]
                    blanket[6] = self.stems[k][j]
                    blanket[7] = self.affixs[k][j]
                    self.stemlist.append(self.stems[k][j])
                    # Add sample
                    self.__change_sample(blanket, 1)
            print ("Iteration %s" % (_ + 1))
        print ("\nIterations finished")
        self.__write_labeled_data()
        print ("Data written")