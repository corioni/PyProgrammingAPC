
import random
import math
from math import sqrt
import sys


def main():
    '''
    Simple function to estimate the value of pi, using a certain number of sample \n
    you pass the number of samples on command line\n
    be carefull it must be an integer
    '''
    N = sys.argv[1]
    if not N.isdigit():
        print("N must be an integer")
        sys.exit(1)
    n_inside = 0

    # calculate pi and error

    for i in range(int(N)):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        l = sqrt(x**2 + y**2)
        if l < 1:
            n_inside += 1

    pi = 4 * n_inside / int(N)
    print(pi)

if __name__ == "__main__":
    main()
