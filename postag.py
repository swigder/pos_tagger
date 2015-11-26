from pos_tagger.corpus import TrainingBrownCorpus
from pos_tagger.bigram_pos_model import BigramPosModel
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='POS Tagger.')

    parser.add_argument('sentence', type=str, help='sentence with spaces between all tokens, including punctuation.')
    parser.add_argument('--mode', default='command', choices=['command', 'interactive'],
                        help='the file where the sum should be written')

    args = parser.parse_args()

    model = BigramPosModel(TrainingBrownCorpus())
    pos = model.decode(args.sentence.split(' '))
    print(' '.join(pos))
    if args.mode == 'interactive':
        while True:
            sentence = input(">> ")
            pos = model.decode(sentence.split(' '))
            print(' '.join(pos))
