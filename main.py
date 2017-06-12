import sys
import dissimilarity_matrix
from clustering import fuzzyClustering

def run(E, D, K, T, m, q, epsilon):
    """ Runs the fuzzy clustering 100 times and selects the best solution based on the heterogeneity parameter. """
    optimal = None
    for t in range(100):
        # print "Round #%d" % (t+1)
        current = fuzzyClustering(E, D, K, T, m, q, epsilon)
        if (optimal == None) or (optimal[2] > current[2]):
            optimal = current
    return optimal

def writeMatrix(M, filename):
    """ Prints the matrix M to a .txt file named `filename`. """
    f = open(filename+'.txt', 'w')
    f.write(filename+':\n')
    m = len(M)
    n = len(M[0])
    for i in range(m):
        for j in range(n):
            e = M[i][j]
            s = str(e) + ' '
            f.write(s)
        f.write('\n')
    f.close()

def computeHardPartition(U):
    """ Given the membership degree matrix U, computes the hard partitioning. """
    H = [] # Will be the hard partitioning
    m = len(U)
    n = len(U[0])
    for i in range(m):
        k = 0
        u_max = U[i][0]
        for j in range(1, n):
            if U[i][j] > u_max:
                k = j
                u_max = U[i][j]
        H.append([k])
    return H

def computeRandIndex(E, H, Y):
    """ Computes the Adjusted Rand Index. """
    m = len(E)
    a = 0
    b = 0
    c = 0
    d = 0
    for i in range(m):
        for j in range(i, m):
            sameY = Y[i] == Y[j]
            sameH = H[i] == H[j]
            if sameY and sameH:
                a += 1
            elif (not sameY) and (not sameH):
                b += 1
            elif sameY and (not sameH):
                c += 1
            else:
                d += 1
    R = (a + b) / float(a + b + c + d)
    return R

if __name__ == "__main__":
    FILENAME = 'database/segmentation.test.txt'

    # Calcula a matriz de dissimilaridades
    (E, Y, D) = dissimilarity_matrix.proccessData(FILENAME)
    K = 7
    T = 100
    m = 1.6
    q = 3
    s = 1
    epsilon = 10 ** -10
    (U, G, J) = run(E, D, K, T, m, q, epsilon)
    H = computeHardPartition(U)
    R = computeRandIndex(E, H, Y)

    f = open('rand-index.txt', 'w')
    f.write("Rand index: %f\n" % R)
    f.write("J: %f\n" % J)
    f.close()

    writeMatrix(U, 'fuzzy_partition')
    writeMatrix(G, 'medoids')
    writeMatrix(H, 'hard_partition')
