# Create a vocabulary wrapper
import argparse
import json
import os
import pickle
from collections import Counter

import nltk

annotations = {
    'conceptual': ['train_caption.json']
}


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(data_path, data_name, jsons, threshold):
    """Build a simple vocabulary wrapper."""
    counter = Counter()
    for path in jsons[data_name]:
        full_path = os.path.join(os.path.join(os.path.join(data_path, data_name), 'img_lists'), path)
        captions = json.load(open(full_path,'r'))
        for i, caption in enumerate(captions):
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)

            if i % 1000 == 0:
                print("[%d/%d] tokenized the captions." % (i, len(captions)))

    # Discard if the occurrence of the word is less than min_word_cnt.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def main(data_path, data_name):
    vocab = build_vocab(data_path, data_name, jsons=annotations, threshold=0)
    with open('./pixlens/editing/impl/open_edit/vocab/%s_vocab.pkl' % data_name, 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
    print("Saved vocabulary file to ", './vocab/%s_vocab.pkl' % data_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data')
    parser.add_argument('--data_name', default='conceptual')
    opt = parser.parse_args()
    main(opt.data_path, opt.data_name)
