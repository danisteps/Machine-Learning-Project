import random
import math
from decimal import Decimal, localcontext
from numpy import float64

def computePrototypes(U, D, wt, m, n, q, k):
    """ Computes the prototype G_k which minimizes the clustering criterion J. (Proposition 2.1) """
    Gk = [] # Will be the prototype
    candidates = [] # Auxiliar list

    for h in range(n):
        tmp = []
        for i in range(n):
            r = float64(U[i][k]) ** float64(m)

            d = 1
            j = 0
            while (j < len(D)):
                d += float64(wt[j][k]) * float64(D[j][i][h])
                j += 1

            r = r * d
            tmp.append(r)
        J = sum(tmp)
        c = (h, J)
        candidates.append(c)

    #AJUSTAR OS SORTS
    # Sorts the candidates according to the adequacy criterion
    candidates.sort(lambda (i, J_i), (k, J_k) : -1 if (J_i < J_k) else 1 if (J_i > J_k) else 0)

    # Sets the prototype to have the elements such that the criterion is minimum
    i = 0
    while i < q:
        c = candidates[i][0]
        Gk.append(c)
        i += 1

    return Gk

def _selectRandomPrototypes(K, n, q):
    """ Creates K random prototypes with cardinality q. """
    G = [] # Vector of prototypes
    elements = range(n) # Index of the elements of E
    for k in range(K):
        Gk = random.sample(elements, q)
        G.append(Gk)
    return G

def _extendedDissimilarity(D, Gk, e):
    """ Extended dissimilarity function. """
    tmp = []
    for g in Gk:
        tmp.append(D[e][g])
    return sum(tmp)

def _updateMembershipDegree(D, G, K, wt, n, m):
    """ Updates the membership degree based on the new prototypes. """
    U = []
    exp = float64(1) / float64(m-1)
    for i in range(n):
        U_i = []
        for k in range(K):

            tmp = []
            j = 0
            while (j < len(D)):
                wtjk = float64(wt[j][k])
                exj = float64(_extendedDissimilarity(D[j], G[k], i))
                num = wtjk * exj # Converts to float
                for h in range(K):
                    wtjh = float64(wt[j][h])
                    exh = float64(_extendedDissimilarity(D[j], G[h], i))
                    r = (num / (wtjh * exh)) ** exp
                    tmp.append(r)
                j += 1

            u_i_k = sum(tmp) ** -1
            U_i.append(u_i_k)
        U.append(U_i)
    return U

def _goalFunction(D, G, U, K, wt, n, m):
    """ Computes the goal function based on the new membership degree matrix and the new prototypes. """
    J = 0
    for k in range(K):
        for i in range(n):
            u = float64(U[i][k]) ** float64(m)

            j = 0
            while (j < len(D)):
                d = float64(wt[j][k]) * float64(_extendedDissimilarity(D[j], G[k], i))
                j += 1

            J += float64(u) * float64(d)
    return J

def _setWeightVector(U, G, D, K, m, n):

    w = [[0 for i in range(K)] for j in range(len(D))]

    for k in range(K):
        for p in range(len(D)):
            prod = float64(1)
            det = float64(0)
            for h in D:
                sum = float64(0)
                for i in range(n):
                    sum += (float64(U[i][k]) ** float64(m)) * float64(_extendedDissimilarity(h, G[k], i))
                prod *= sum
            num = prod ** (float64(1)/float64(len(D)))
            for i in range(n):
                det += (float64(U[i][k]) ** float64(m)) * float64(_extendedDissimilarity(D[p], G[k], i))

            if (det == 0):
                det = float64(1)

            w[p][k] = num / det

    return w

def fuzzyClustering(E, D, K, T, m, q, epsilon):
    """ Partitioning Fuzzy K-Medoids Clustering Algorithm Based on a Multiple Dissimilarity Matrix. (Section 2.1)
        - E: Set/List of elements;
        - D: Dissimilarity matrix;
        - K: Number of clusters;
        - T: Maximum number of iterations;
        - m: parameter of fuzziness of membership of elements;
        - q: cardinality of the prototypes;
        - epsilon: threshold for the goal function.
        - w : matrix of relevance weight """

    n = len(E)
    W = [[[0 for i in range(K)] for j in range(len(D))] for k in range(T)] #initial weight
    W[0] = [[1 for i in range(K)] for j in range(len(D))]

    G = _selectRandomPrototypes(K, n, q) # Initial prototypes
    U = _updateMembershipDegree(D, G, K, W[0], n, m) # Membership degree Matrix
    J = _goalFunction(D, G, U, K, W[0], n, m) # Homogeneity / Goal function
    t = 0 # Current Iteration step

    while t < T:
        t += 1
        print "Iteration #%d" % t

        # Step 1
        print "Updating prototypes..."
        for k in range(K):
            G[k] = computePrototypes(U, D, W[t-1], m, n, q, k)
        # Step 2
        print "Updating weights..."
        if (t < T):
            W[t] = _setWeightVector(U, G, D, K, m, n)
        # Step 3
        print "Updating membership degree..."
        U = _updateMembershipDegree(D, G, K, W[t-1], n, m)
        # Step 4
        print "Computing goal function..."
        g = _goalFunction(D, G, U, K, W[t-1], n, m)
        if abs(J - g) < epsilon :
            break
        else :
            J = g

    return (U, G, J)
