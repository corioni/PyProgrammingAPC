
import random
import math
from math import sqrt
import sys



def bootstrap_pi(N, n_bootstrap):
    pi_values = []
    for _ in range(n_bootstrap):
        n_inside = 0
        for i in range(int(N)):
            x = random.uniform(0, 1)
            y = random.uniform(0, 1)
            l = sqrt(x**2 + y**2)
            if l < 1:
                n_inside += 1
        pi_values.append(4 * n_inside / int(N))
    return pi_values

def error_bootstrap(pi_values):
    pi_mean = sum(pi_values) / len(pi_values)
    pi_std = sqrt(sum((x - pi_mean) ** 2 for x in pi_values) / len(pi_values))
    return pi_mean, pi_std

def main():
    '''Estimate pi using the quarter circle method and bootstrap error estimation.'''
    pass
    
    if (len(sys.argv) != 3) or not sys.argv[1].isdigit() or not sys.argv[2].isdigit():
        print("Usage: python3 estimate_pi.py N n_bootstrap\t example: python3 estimate_pi.py 1000 20")  
        sys.exit(1)
    
    N = int(sys.argv[1])
    n_bootstrap = int(sys.argv[2])

    pi_values = bootstrap_pi(N, n_bootstrap)
    pi_mean, pi_std = error_bootstrap(pi_values)

    print(f"Estimated pi: {pi_mean:.6f} ± {pi_std:.6f}")


if __name__ == "__main__":
    main()
