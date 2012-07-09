from numpy import dot, ones, zeros
from numpy.random import beta, multinomial, uniform
from numpy.random.mtrand import dirichlet

def categorical(dist):
    return multinomial(1, dist).argmax()

def generate_sbm_data(N, K, alpha, a, b, m=None):
    """
    N is the number of nodes
    K is the number of blocks
    alpha is the concentration parameter
    a and b are the shape parameters
    m is the base measure
    """

    if m == None:
        m = ones(K) / K # uniform base measure

    Z = zeros((N, K)) # block assignments

    # sample (global) distribution over blocks

    [theta] = dirichlet(alpha * m, 1)

    # sample between- and within-block edge probabilities

    phi = beta(a, b, (K, K))

    # sample block assignments

    for n in range(1, N+1):
        Z[n-1,:] = multinomial(1, theta)

    # sample edges

    Y = (uniform(size=(N, N)) <= dot(dot(Z, phi), Z.T)).astype(int)

    return Z, Y

def generate_mmsbm_data(N, K, alpha, a, b, m=None):
    """
    N is the number of nodes
    K is the number of blocks
    alpha is the concentration parameter
    a and b are the shape parameters
    m is the base measure
    """

    if m == None:
        m = ones(K) / K # uniform base measure

    Y = zeros((N, N), dtype=int) # edges

    # sample node-specific distributions over blocks

    [theta] = dirichlet(alpha * m, (1, N))

    # sample between- and within-block edge probabilities

    phi = beta(a, b, (K, K))

    # sample block assignments and edges

    for i in range(1, N+1):
        for j in range(1, N+1):
            idx = (categorical(theta[i-1,:]), categorical(theta[j-1,:]))
            Y[i-1,j-1] = uniform() <= phi[idx]

    return theta, Y

def main():

    N=4 # number of nodes
    K=3 # number of blocks
    alpha = 1 # concentration parameter
    a = 1 # shape parameter
    b = 1 # shape parameter

    Z, Y = generate_sbm_data(N, K, alpha, a, b)
    theta, Y = generate_mmsbm_data(N, K, alpha, a, b)

if __name__ == '__main__':
    main()

