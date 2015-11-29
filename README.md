# pos_tagger

This part-of-speech tagger takes a list of tokens and returns a list of parts of speech corresponding to the tokens.

It works by training a hidden markov model over tagged training data, calculating bigram frequencies for parts of speech, as well as frequencies of each word to each part of speech.  Then, to determine the parts of speech for a given list of tokens, it uses Viterbi to calculate the parts of speech with the highest probability for the input ("decode" functionality for a hidden markov model).

Some notes on the implementation:
* The NLTK universal tagset was chosen, and 80% of the Brown corpus was used a training data (with the rest reserved for tuning and testing).
* The model is case-insensitive, and input will be normalized to a single case no matter how it is provided.
* In the case of an out-of-vocabulary token in the input, that "slot" will be evaluated based only on the part-of-speech bigram probabilities with respect to the other tokens, and not based on any characteristics of the token itself or the relative frequencies of parts-of-speech overall.  See below for ways to improve this.

## Usage
### Prerequisites and required libraries
This package requires python3 to run, and uses nltk, and pytest. It assumes that the NLTK Brown corpus has been installed. For more information, see http://www.nltk.org/data.html.

### Running from the command line
To run for a single sentence, run `python3 postag.py <sentence>`.  To run for multiple sentences, it is faster to run in interactive mode, to avoid having to recompute the model.  To enter interactive mode, run `python3 postag.py --mode interactive <sentence>`.  You will be prompted again for input after the output is provided.

For more details, run `python postag.py --help`.

### Using as a library
To use a tagger, create a `BigramPosModel` and provide a tagged corpus, and then call `decode`. In addition, the general purpose `HiddenMarkovModel` class can be extended or instantiated for use in other models.  For more detail, see class documentation and sample code in tests and in `postag.py` and `homework.py`. 
