import math

def readData(filename):
    """ Returns a matrix with the data from the file. """
    f = open(filename, 'r')
    lines = f.readlines()
    # Pre-proccesses the data by removing the semicolons
    mp = lambda l: l.replace('\n', '').split(',')
    data = map(mp, lines)
    getPart = lambda l : l[0]
    part = map(getPart, data)
    toFloat = lambda x: float(x)
    drp = lambda l: map(toFloat, l[1:len(l)]) # Converts to float
    data = map(drp, data) # Removes the last element of every line (the class parameter)
    f.close()

    spt1 = lambda l: l[0:8]
    spt2 = lambda l: l[9:18]
    data1 = map(spt1, data)
    data2 = map(spt2, data)

    return (data, data1, data2, part)


def computeDissimilarityMatrix(data):
    """ Computes the Dissimilarity Matrix. The data should be pre-proccessed. """
    # Dissimmilarity function
    delta = lambda (x_ik, x_jk) : math.pow(x_ik - x_jk, 2)
    d = lambda x_i, x_j : sum(map(delta, zip(x_i, x_j)))

    # Number of examples
    n = len(data)

    matrix = [[0 for j in range(n)] for i in range(n)] # Matrix filled with 0s

    for i in range(n):
        for j in range(i+1, n): # The dissimilarity matrix is symmetrical
          matrix[i][j] = math.sqrt(d(data[i], data[j]))
          matrix[j][i] = matrix[i][j]

    return matrix


def proccessData(filename):
    (data, data1, data2, part) = readData(filename)
    D1 = computeDissimilarityMatrix(data1)
    D2 = computeDissimilarityMatrix(data2)
    # D = [[[0 for i in range(len(data))] for j in range(len(data))] for k in range(2)]
    D = [None, None]
    D[0] = D1
    D[1] = D2
    return (data, data1, data2, part, D)


if __name__ == "__main__":
   FILENAME = 'database/segmentation.test.txt'

   #Calcula a matriz de dissimilaridades
   (E, Y, D) = proccessData(FILENAME)
   print len(E)
