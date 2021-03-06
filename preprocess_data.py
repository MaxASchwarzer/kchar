from subprocess import check_output
import subprocess
import os
import argparse

def preprocess(data, batch_size):
    """
    This is a handy utility to reorder a text file into "threads", to allow
    stateful processing for language models.  This is necessary due to the
    batch-based nature of modern language models.  For example, with batch
    size of three and nine sentences, we would want the following order
    of sentences (assuming that at least immediately adjacent sentences
    are related):

    s1
    s4
    s7
    s2
    s5
    s8
    s3
    s6
    s9

    This script gives you that, given a file and a number of batches.
    The newly-reordered file is put in (data).batchsplit
    """
    length = check_output(["wc", "-l", str(data)])
    length = int(length.split()[0])

    num_lines_per_file = length // (batch_size)
    lines_to_cut = length % batch_size


    _ = check_output("head -n -{} ".format(lines_to_cut) +
                     data + " > temp.txt; mv temp.txt " +data, shell = True)
    _ = check_output(["split", str(data), "-l", str(num_lines_per_file), "tempsplitfile"])
    _ = check_output(" ".join(["paste",
                        "-d", "\'\\n\'",
                        "./tempsplitfile*", "> "+str(data)+".temp"]),
                        stderr=subprocess.STDOUT, shell=True)

    check_output("rm tempsplitfile*", shell=True)

    check_output("sed  '/^$/d' {} > {}".format(str(data)+".temp",
                                               str(data)+".batchsplit"), shell= True)
    os.remove(str(data)+".temp")

def merge_lines_to_constant_length(data, line_length):
    with open(data, "r") as f:
        word_acc = []
        outputfile = open(data+".equallines", "w")
        #import ipdb; ipdb.set_trace()
        for line in f:
            words = line.replace("\n", " eos").split(" ")
            word_acc += words
            while (len(word_acc) > line_length):
                words = word_acc[:line_length]
                outputfile.write((" ".join(words) + "\n"))
                word_acc = word_acc[line_length:]

        if len(word_acc) != 0:
            outputfile.write(" ".join(word_acc) + "\n")

def split_and_preprocess(files, line_length, batch_size):
    """
    Runs the entire set of functions, and produces f.equallines.batchsplit for every
    file f in the input.
    """
    for f in files:
        merge_lines_to_constant_length(f, line_length)
        preprocess(f+".equallines", batch_size)
        os.remove(str(f)+".equallines")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Batch-split text files")
    parser.add_argument('files', metavar='F', type=str, nargs='+',
                    help='Files to process')
    parser.add_argument("--batch_size", dest = "batch_size", default = 20,
        type = int, help = "Batch size (default = 20)")
    parser.add_argument("--line_length", default = 35, type = int,
                        help = "Line length (default: 35)")

    args = parser.parse_args()
    for f in args.files:
        merge_lines_to_constant_length(f, args.line_length)
        preprocess(f+".equallines", args.batch_size)
        os.remove(str(f)+".equallines")
