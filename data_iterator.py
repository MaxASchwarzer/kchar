import numpy
import random
import copy
from subprocess import check_output

import pickle as pkl
import gzip


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


class TextIterator:
    """Simple Bitext iterator."""
    def __init__(self, source,
                 source_dict, target_dict,
                 batch_size=32,
                 maxlen=100,
                 char_mode=True,
                 max_word_len=16,
                 n_words_source=-1,
                 n_words_target=-1,
                 k = 100):
        self.source = fopen(source, 'r')
        with open(source_dict, 'rb') as f:
            self.source_dict = pkl.load(f)
        with open(target_dict, 'rb') as f:
            self.target_dict = pkl.load(f)

        self.inverse_source_dict = {v: k for k, v in self.source_dict.items()}
        self.inverse_target_dict = {v: k for k, v in self.target_dict.items()}

        self.batch_size = batch_size
        self.maxlen = maxlen
        length = check_output(["wc", "-l", str(source)])
        length = int(length.split()[0])
        self.num_batches = length//batch_size
        self.max_word_len = max_word_len
        self.char_mode = char_mode

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target

        self.source_buffer = []
        self.target_buffer = []
        self.k = batch_size * k

        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)
        self.target.seek(0)

    def __next__(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        # fill buffer, if it's empty
        #assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'

        if len(self.source_buffer) == 0:
            for k_ in range(self.k):
                ss = self.source.readline()
                if ss == "":
                    break

                self.source_buffer.append(ss.strip().split())

            # sort by target buffer
            #tlen = numpy.array([len(t) for t in self.target_buffer])
            #tidx = tlen.argsort()

            #_sbuf = [self.source_buffer[i] for i in tidx]
            #_tbuf = [self.target_buffer[i] for i in tidx]

            #self.source_buffer = _sbuf
            #self.target_buffer = _tbuf

        if len(self.source_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:
            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break

                tt = copy.copy(ss)
                tt = tt[1:]
                tt = [2] + [self.target_dict[w] if w in self.target_dict else 1 for w in tt]
                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_source else 1 for w in tt]

                if(self.char_mode):
                    sschars = []
                    for i, word in enumerate(ss):
                        characters = numpy.asarray(list(map(self.source_dict.get, word[:self.max_word_len])))
                        characters.resize(self.max_word_len)
                        sschars.append(characters)
                    ss = sschars
                else:
                    ss = [self.source_dict[w] if w in self.source_dict else 1 for w in ss]
                    if self.n_words_source > 0:
                        ss = [w if w < self.n_words_source else 1 for w in ss]

                # read from source file and map to word index
                source.append(ss)
                target.append(tt)

                if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size:
                    break

        except IOError:
            self.end_of_data = True

        if len(source) <= 0 or len(target) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration


        max_length = max(list(map(len, source)) + list(map(len, target)))

        parsed_source = [numpy.asarray(s) for s in source]
        source_array = numpy.zeros((self.batch_size, max_length, self.max_word_len), dtype = "int32")
        for i, array in enumerate(parsed_source):
            source_array[i, :array.shape[0], :] = array

        target_array = numpy.zeros((self.batch_size, max_length, 1), dtype = "int32")
        for i, array in enumerate(target):
            target_array[i, :len(array), 0] = array

        #import ipdb; ipdb.set_trace()
        return source_array, target_array
