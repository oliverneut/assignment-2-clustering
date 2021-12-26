import numpy as np
from numpy.lib.function_base import diff

def main():
    labels = np.loadtxt('labels.txt', delimiter=',', usecols = 1)
    diff_lbls, counts = np.unique(labels, return_counts=True)
    print(diff_lbls)
    print(counts)
    print((counts[1] + counts[2] + counts[3]) / sum(counts) * 100 , '%')

if __name__ == '__main__':
    main()