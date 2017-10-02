import numpy
import pickle as pkl

import sys
import fileinput

from collections import OrderedDict

def main():
    for filename in sys.argv[1:]:
        print('Processing', filename)
        word_freqs = OrderedDict()
        char_freqs = OrderedDict()
        with open(filename, 'r') as f:
            for line in f:
                words_in = line.strip().split(' ')
                for w in words_in:
                    if w not in word_freqs:
                        word_freqs[w] = 0
                    word_freqs[w] += 1
                    for char in w:
                        if char not in char_freqs:
                            char_freqs[char] = 0
                        char_freqs[char] += 1
        words = list(word_freqs.keys())
        freqs = list(word_freqs.values())

        chars = list(char_freqs.keys())
        cfreqs = list(char_freqs.values())

        #Sort words
        sorted_idx = numpy.argsort(freqs)
        sorted_words = [words[ii] for ii in sorted_idx[::-1]]
        worddict = OrderedDict()
        worddict['eos'] = 0
        worddict['UNK'] = 1
        worddict['SENTENCESTART'] = 2
        worddict["ZEROPAD"] = 3
        for ii, ww in enumerate(sorted_words):
            worddict[ww] = ii+4

        #Sort chars
        sorted_idx = numpy.argsort(cfreqs)
        sorted_chars = [chars[ii] for ii in sorted_idx[::-1]]
        chardict = OrderedDict()
        chardict["ZEROPAD"] = 0
        for ii, ch in enumerate(sorted_chars):
            chardict[ch] = ii + 1


        with open('worddict.pkl', 'wb') as f:
            pkl.dump(worddict, f)
        with open('chardict.pkl', 'wb') as f:
            pkl.dump(chardict, f)

        print('Done')




if __name__ == '__main__':
    main()
