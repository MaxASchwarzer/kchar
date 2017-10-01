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

    num_lines_per_file = length // batch_size

    _ = check_output(["split", str(data), "-l", str(num_lines_per_file), "tempsplitfile"])
    _ = check_output(" ".join(["paste",
                        "-d", "\'\\n\'",
                        "./tempsplitfile*", "> "+str(data)+"batchsplit"]),
                        stderr=subprocess.STDOUT, shell=True)

    check_output("rm tempsplitfile*", shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Batch-split text files")
    parser.add_argument('files', metavar='F', type=str, nargs='+',
                    help='Files to process')
    parser.add_argument("--batch_size", dest = "batch_size", default = 32,
        type = int, help = "Batch size (default = 32)")

    args = parser.parse_args()
    for f in args.files:
        preprocess(f, args.batch_size)
