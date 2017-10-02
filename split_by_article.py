import numpy as np
import argparse

def choose_file(valid_fraction, test_fraction):
    choice = np.random.rand()
    if choice < test_fraction:
        return(0)
    elif choice < (valid_fraction + test_fraction):
        return(1)
    else:
        return(2)

def split_by_article(files, valid_fraction = 0.05,
 test_fraction = 0.05, lowercase = True, sentences = False):
    """
    This code is designed to split wikipedia files from Dr. Kauchak's script
    by article, to improve robustness.

    If desired, lowercasing can also be performed.

    The data should be organized in the format:
    article \t paragraph \t text

    Data is written to train.txt, valid.txt and test.txt
    """
    articles = {}
    for f in files:
        with open(f, "r") as f:
            files = [open("test.txt", "w"),
                     open("valid.txt", "w"),
                     open("train.txt", "w")]
            articles = {}
            for line in f:
                groups = line.split("\t")
                article = groups[0]
                text = groups[2]
                if lowercase:
                    text = text.lower()
                if not sentences:
                    if not article in articles:
                        articles[article] = choose_file(valid_fraction,
                                                        test_fraction)
                    target = articles[article]
                else:
                    target = choose_file(valid_fraction,
                                            test_fraction)
                if text != "":
                    files[target].write(text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Split files by article")
    parser.add_argument('files', metavar='F', type=str, nargs='+',
                    help='Files to process')
    parser.add_argument('-l', const=True, default = True,
                        dest = "lowercase", action = "store_const",
                        help = "Lowercase text (default: false)")
    parser.add_argument("-v", default = 0.05, dest = "valid", type = float,
                        help = "Validation split size (default: 0.05)")
    parser.add_argument("-t", default = 0.05, dest = "test", type = float,
                        help = "Test split size (default: 0.05)")
    parser.add_argument("-s", default = False, const = True,
                        action = "store_const",
                        help = "Ignore articles and split by sentences" +
                        " (default: false)", dest = "sentences")

    args = parser.parse_args()
    split_by_article(args.files,
                     args.valid,
                     args.test,
                     args.lowercase,
                     args.sentences)
