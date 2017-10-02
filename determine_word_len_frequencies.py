import numpy as np
import sys
import argparse

def determine_word_len_frequencies(files):
    frequencies = np.zeros(101, dtype="float32")
    for f in files:
        with open(f, "r") as f:
            for line in f:
                for word in line.split():
                    if len(word) > 100:
                        print("PROBLEM WORD:", word)
                        continue
                    frequencies[len(word)] += 1

    frequencies = frequencies / np.sum(frequencies)

    print("Total frequency at each length:")
    print(frequencies[:30])

    cumulative_freqs = np.cumsum(frequencies)
    print("Cumulative frequency:")
    print(cumulative_freqs[:30])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Count word length freqs")
    parser.add_argument('files', metavar='F', type=str, nargs='+',
                    help='Files to process')
    args = parser.parse_args()
    determine_word_len_frequencies(args.files)
