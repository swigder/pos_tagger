from pos_tagger.corpus import TrainingBrownCorpus
from pos_tagger.bigram_pos_model import BigramPosModel
import argparse


model = BigramPosModel(TrainingBrownCorpus())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='POS Tagger.')

    parser.add_argument('sentence', type=str, help='sentence with spaces between all tokens, including punctuation.')

    args = parser.parse_args()

    pos = model.decode(args.sentence.split(' '))
    print(' '.join(pos))
