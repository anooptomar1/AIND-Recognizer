import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        Reference: 
        http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
        http://www.stanfordphd.com/BIC.html
        http://www.modelselection.org/bic/
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
        BIC = −2 log L + p log N
        where L is the likelihood of the fitted model, p is the number of parameters,
        and N is the number of data points. The term −2 log L decreases with
        increasing model complexity (more parameters), whereas the penalties 2p or
        p log N increase with increasing complexity.

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import math 

        # TODO implement model selection based on BIC scores

        models_and_scores = []     

        for n_components in range(self.min_n_components, self.max_n_components):
            
            model = self.base_model(n_components)

            if model is not None:
                try:
                    logL = model.score(self.X, self.lengths)
                    # BIC = −2 log L + p log N

                    # p is the number of free parameters;
                    # p = num components * num components  + 2 * num components * num feature -1            
                    p = n_components * n_components + 2 * n_components * model.n_features - 1
                    # N is the number of data points
                    N = float(len(self.X))
                    BIC = (-2 * logL) + (p * math.log(N))
                    models_and_scores.append((model, BIC))
                except:
                    pass                    
                
            else:
                if self.verbose:
                    print('Failed to create a model with {} number of components using the word {}'.format(
                        n_components, self.this_word))                
        
        # http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
        # Model selection: The lower the AIC/BIC value the better the model 
        # (only compare AIC with AIC and BIC with BIC values!).
        models_and_scores.sort(key = lambda item : item[1], reverse=False)

        if self.verbose:
            print('Best model {} with word {} average logL {}'.format(
                models_and_scores[0][0].n_components, self.this_word, models_and_scores[0][1]))

        return models_and_scores[0][0]


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))

    "X(i) is simply the word in evaluation. log(P(X(i))) in terms of hmmlearn is simply the model's score 
    for that particular word." - @Mohan-27 (https://discussions.udacity.com/t/what-is-x-i-in-the-formula-to-calculate-dic/239844)

    '''

    def select(self, words_to_train):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # TODO implement model selection based on DIC scores

        models_and_scores = []   

        #Y = [seq_len[0] for seq_len in self.hwords[word] for word in words_to_train if word is not self.this_word]
        #Y_lengths = [seq_len[1] for seq_len in self.hwords[word] for word in words_to_train if word is not self.this_word]        
        other_word_sequences = [seq_len for seq_len in self.hwords[word] for word in words_to_train if word is not self.this_word]        
        other_words_length = float(len(words_to_train)-1)

        for n_components in range(self.min_n_components, self.max_n_components):                        
            model = self.base_model(n_components)

            if model is not None:
                try:
                    logL = model.score(self.X, self.lengths)
                    others_logL = [model.score(word_Y[0], word_Y[1]) for word_Y in other_word_sequences]
                    sum_others_logL = sum(others_logL)
                    DIC = logL - sum_others_logL/other_words_length

                    models_and_scores.append((model, DIC))
                except: 
                    pass 
        
        models_and_scores.sort(key = lambda item : item[1], reverse=True)

        if self.verbose:
            print('Best model {} with word {} average logL {}'.format(
                models_and_scores[0][0].n_components, self.this_word, models_and_scores[0][1]))

        return models_and_scores[0][0]


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
    '''

    def select(self):
        # TODO implement model selection using CV
        from sklearn.model_selection import KFold
        import asl_utils
        import sys

        warnings.filterwarnings("ignore", category=DeprecationWarning)

        models_and_scores = []     

        for n_components in range(self.min_n_components, self.max_n_components):

            trained_model = None # keep a valid reference to the trained model in-case the last iteration fails
            logL_results = [] # create a list to hold the results (used to find the average)

            split_method = KFold(n_splits=2)   
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):    
                self.X, self.lengths = asl_utils.combine_sequences(cv_train_idx, self.sequences)
                test_X, test_lengths = asl_utils.combine_sequences(cv_test_idx, self.sequences)                

                model = self.base_model(n_components)
                if model is not None:
                    try:
                        logL = model.score(test_X, test_lengths)
                        logL_results.append(logL)
                        trained_model = model 
                    except:
                        pass                    

            if len(logL_results) > 0:
                avg_logL_score = sum(logL_results) / max( float(len(models_and_scores)), sys.float_info.epsilon )
                models_and_scores.append((trained_model, avg_logL_score))

                # lets append it to the model as well so we can surface it in the notepad 
                #model['avg_logL_score'] = avg_logL_score
            else:
                if self.verbose:
                    print('Failed to create a model with {} number of components using the word {}'.format(
                        n_components, self.this_word))                
        
        # want to maximise likelihood therefore sort from largest to smallest and return the first 
        # one 
        # http://blog.stata.com/2011/02/16/positive-log-likelihood-values-happen/13/
        models_and_scores.sort(key = lambda item : item[1], reverse=True)

        if self.verbose:
            print('Best model {} with word {} average logL {}'.format(
                models_and_scores[0][0].n_components, self.this_word, models_and_scores[0][1]))

        return models_and_scores[0][0]
