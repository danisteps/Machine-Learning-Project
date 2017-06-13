import sys
import dissimilarity_matrix
from clustering import fuzzyClustering
from sklearn.metrics.cluster import adjusted_rand_score

def cluster(E, D, K, T, m, q, epsilon):
    """ Runs the fuzzy clustering 100 times and selects the best solution based on the heterogeneity parameter. """
    optimal = None
    for t in range(100):
        print "-- Round %d" % t
        # print "Round #%d" % (t+1)
        current = fuzzyClustering(E, D, K, T, m, q, epsilon)
        if (optimal == None) or (optimal[2] > current[2]):
            print "Found new optimal solution...\nJ=%f" % optimal[2]
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
    return adjusted_rand_score(true_labels, predicted_labels)

def run():
    FILENAME = 'database/segmentation.test.txt'

    # Calcula a matriz de dissimilaridades
    print "Computing dissimilarity matrix..."
    (E, Y, D) = dissimilarity_matrix.proccessData(FILENAME)
    
    print "Initializing parameters..."
    K = 7
    T = 10
    m = 1.6
    q = 3
    s = 1
    epsilon = 10 ** -10

    print "Running clustering algorithm..."
    (U, G, J) = cluster(E, D, K, T, m, q, epsilon)

    print "Computing Hard Partition..."
    H = computeHardPartition(U)

    print "Computing Adjusted Rand Index..."
    R = computeRandIndex(H, Y)
    print "ARI: %f" % R

    f = open('rand-index.txt', 'w')
    f.write("Rand index: %f\n" % R)
    f.write("J: %f\n" % J)
    f.close()

    writeMatrix(U, 'fuzzy_partition')
    writeMatrix(G, 'medoids')
    writeMatrix(H, 'hard_partition')

if __name__ == '__main__':
    run()
