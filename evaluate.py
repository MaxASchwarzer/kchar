import codecs
import numpy as np
import re
import argparse
import pickle as pickle
from model.LSTMCNN import LSTMCNN
from util.BatchLoaderUnk import Tokens, encoding # needed by pickle.load()
from math import exp
from collections import OrderedDict

def vocab_unpack(vocab):
    return vocab['idx2word'], vocab['word2idx'][()], vocab['idx2char'], vocab['char2idx'][()]

class Vocabulary:
    def __init__(self,
                 tokens,
                 word_vocab_file,
                 char_vocab_file,
                 max_word_l=17,
                 word_vocab_size = 60000):
        self.tokens = tokens
        self.max_word_l = max_word_l
        self.prog = re.compile('\s+')

        print('loading vocabulary file...')
        with open(word_vocab_file, "rb") as f:
            word2idx_large = pickle.load(f)
            self.word2idx = {k:v for k, v in word2idx_large.items() if v <= word_vocab_size}
            self.idx2word = {v:k for k, v in self.word2idx.items()}
        with open(char_vocab_file, "rb") as f:
            self.char2idx = pickle.load(f)
            self.idx2char = {v:k for k,v in self.char2idx.items()}
        self.vocab_size = len(self.idx2word)
        print('Word vocab size: %d, Char vocab size: %d' % (len(self.idx2word), len(self.idx2char)))
        self.word_vocab_size = word_vocab_size
        self.char_vocab_size = len(self.idx2char)

    def index(self, word):
        if word[0] == self.tokens.UNK and len(word) > 1: # unk token with character info available
            word = word[2:]
            w = self.word2idx[self.tokens.UNK]
        else:
            w = self.word2idx[word] if word in self.word2idx else self.word2idx[self.tokens.UNK]

        c = np.zeros(self.max_word_l, dtype='int32')
        chars = [self.char2idx[self.tokens.START]] # start-of-word symbol
        chars += [self.char2idx[char] for char in word if char in self.char2idx]
        chars.append(self.char2idx[self.tokens.END]) # end-of-word symbol
        if len(chars) >= self.max_word_l:
            chars[self.max_word_l-1] = self.char2idx[self.tokens.END]
            c = chars[:self.max_word_l]
        else:
            c[:len(chars)] = chars

        return w, c

    def get_input(self, line):
        output_words = []
        output_chars = []

        line = line.replace('<unk>', self.tokens.UNK)  # replace unk with a single character
        line = line.replace(self.tokens.START, '')  # start-of-word token is reserved
        line = line.replace(self.tokens.END, '')  # end-of-word token is reserved
        words = self.prog.split(line)
        for rword in [_f for _f in words if _f]:
            w, c = self.index(rword)
            output_words.append(w)
            output_chars.append(c)
        if self.tokens.EOS != '':   # PTB does not have <eos> so we add a character for <eos> tokens
            w, c = self.index(self.tokens.EOS)   # other datasets don't need this
            output_words.append(w)
            output_chars.append(c)

        words = np.array(output_words[-1:] + output_words[:-1], dtype='int32')
        chars = np.array(output_chars[-1:] + output_chars[:-1], dtype='int32')[:, np.newaxis, :]
        output = np.array(output_words, dtype='int32')[:, np.newaxis, np.newaxis]
        return ({'word':words, 'chars':chars}, output)


class evaluator:
    def __init__(self, name, word_vocab_file, char_vocab_file, word_vocab_size, init):
        self.opt = pickle.load(open('{}.pkl'.format(name), "rb"))
        self.opt.batch_size = 1
        self.opt.seq_length = 1
        tokens = Tokens(
            EOS="eos",
            UNK='UNK',    # unk word token
            START='WORDSTART',  # start-of-word token
            END='eos',    # end-of-word token
            ZEROPAD='ZEROPAD' # zero-pad token
        )
        self.reader = Vocabulary(tokens, word_vocab_file, char_vocab_file,
                                 max_word_l=self.opt.max_word_l, word_vocab_size = word_vocab_size)
        self.model = LSTMCNN(self.opt)
        self.model.load_weights('{}.h5'.format(name))
        if init:
            self.state_mean = np.load(init)
        else:
            self.state_mean = None

    def logprob(self, line):
        x, y = self.reader.get_input(line)
        nwords = len(y)
        if self.state_mean is not None:
            self.model.set_states_value(self.state_mean)
        return self.model.evaluate(x, y, batch_size=1, verbose=0), nwords

def main(name, word_vocab_file, char_vocab_file, word_vocab_size, text, calc, init = None):
    ev = evaluator(name, word_vocab_file, char_vocab_file, word_vocab_size, None if calc else init)

    f = codecs.open(text, 'r', encoding)
    if calc:
        lp = 0;
        nw = 0;
        nl = 0;
        state_sum = [np.zeros_like(a) for a in ev.model.state_updates_value]
        for line in f:
            lprob, nwords = ev.logprob(line)
            lp += lprob*nwords
            nw += nwords
            for ssum, update in zip(state_sum, ev.model.state_updates_value):
                ssum += update
            nl += 1
            print("Perplexity = ", exp(lp/nw), "\t(", nl, ")", ssum[0][0]/nl)

        state_mean = [a/nl for a in state_sum]
        np.save(init, state_mean)
    else:
        lp = 0;
        nw = 0;
        nl = 0;
        f.seek(0)
        for line in f:
            lprob, nwords = ev.logprob(line)
            lp += lprob*nwords
            nw += nwords
            nl += 1
            print("Perplexity = ", exp(lp/nw), "\t(", nl, ")")

    exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--vocab_char', type=str)
    parser.add_argument('--vocab_word', type=str)
    parser.add_argument('--init', type=str)
    parser.add_argument('--text', type=str)
    parser.add_argument("--vocab_size", type = int)
    parser.add_argument('--calc', action='store_true')

    args = parser.parse_args()

    main(args.model, args.vocab_word, args.vocab_char, args.vocab_size, args.text, args.calc, init)
