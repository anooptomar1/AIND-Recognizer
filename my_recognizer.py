import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer

    probabilities = []
    guesses = [] 

    #word_indices = sorted([word_idx for seq_index in test_set.sentences_index for word_idx in test_set.sentences_index[seq_index]])
    word_indices = [word_idx for seq_index in test_set.sentences_index for word_idx in test_set.sentences_index[seq_index]]    

    for word_index in word_indices:        

        best_guess_word = None 
        best_guess_logL = float("-inf")

        X, lengths = test_set.get_item_Xlengths(word_index)
        
        #seq_probabilities = [(model_word, model.score(X, lengths)) for model_word, model in models.items()]
        seq_probabilities = {}
        for model_word, model in models.items():
            try:
                logL = model.score(X, lengths)
                #seq_probabilities.append((model_word, logL))
                seq_probabilities[model_word] = logL

                if logL > best_guess_logL:
                    best_guess_logL = logL
                    best_guess_word = model_word
            except:
                #seq_probabilities.append((model_word, float("-inf"))) 
                seq_probabilities[model_word] = float("-inf")

        #seq_probabilities = sorted(seq_probabilities, key=lambda item: item[1], reverse=True)

        probabilities.append(seq_probabilities)
        guesses.append(best_guess_word)

    return probabilities, guesses
