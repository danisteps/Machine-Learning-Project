import math

def readData(filename):
    """ Returns a matrix with the data from the file. """
    f = open(filename, 'r')
    lines = f.readlines()
    # Pre-proccesses the data by removing the semicolons
    mp = lambda l: l.split(',')
    data = map(mp, lines)
    getPart = lambda l : l[0]
    part = map(getPart, data)
    drp = lambda l: l[1:len(l)]
    data = map(drp, data) # Removes the last element of every line (the class parameter)
    f.close()
    return (data, part)

def computeDissimilarityMatrix2(filename):
    count = 0
    matrix = []

    f = open(filename, 'r')

    for line in f:
        matrix.append([j for j in line.split(',')])

    attCount = len(matrix)
    view1AttCount = 9
    view2AttCount = 10
    view1Dis = [[0 for i in range(attCount)] for j in range(view1AttCount)]
    view2Dis = [[0 for i in range(attCount)] for j in range(view2AttCount)]

    xi = 0;
    while (xi < attCount):
        xj = 0

        while (xj < view1AttCount):
            dij = 0.0
            j = xi + 1
            j2 = xj + 1
            i = 0
            #print 'coord: ' + str(xi) + ',' + str(xj)

            while (i < len(matrix)):
                dij += pow(float(matrix[i][j]) - float(matrix[i][j2]), 2)
                i += 1

            dij = math.sqrt(dij)
            view1Dis[xi][xj] = dij;
            xj += 1

        xi += 1

    print 'View 1:';
    print len(view1Dis)
    for i in view1Dis:
        print i;

    xi = 0;
    while (xi < attCount):
        xj = 0

        while (xj < view2AttCount):
            dij = 0.0
            j = xi + view1AttCount + 1
            j2 = xj + view1AttCount + 1
            i = 0
            #print 'coord: ' + str(xi) + ',' + str(xj)

            while (i < len(matrix)):
                dij += pow(float(matrix[i][j]) - float(matrix[i][j2]), 2)
                i += 1

            dij = math.sqrt(dij)
            view2Dis[xi][xj] = dij;
            xj += 1

        xi += 1

    print 'View 2:';
    for i in view2Dis:
        print i;

    return (view1Dis, view2Dis)

def computeDissimilarityMatrix(data):
    """ Computes the Dissimilarity Matrix. The data should be pre-proccessed. """
    # Dissimmilarity function
    delta = lambda (x_ik, x_jk) : 0 if (x_ik == x_jk) else 1
    d = lambda x_i, x_j : sum(map(delta, zip(x_i, x_j)))

    # Number of examples
    n = len(data)

    matrix = [[0 for j in range(n)] for i in range(n)] # Matrix filled with 0s

    for i in range(n):
        for j in range(i+1, n): # The dissimilarity matrix is symmetrical
          matrix[i][j] = d(data[i], data[j])
          matrix[j][i] = matrix[i][j]

    return matrix


def proccessData(filename):
    (data, part) = readData(filename)
    (D1) = computeDissimilarityMatrix(data)
    #D2 = computeDissimilarityMatrix(data, 9, 10)
    D = [[[0 for i in range(len(data))] for j in range(len(data))] for j in range(2)]
    D[0] = D1
    D[1] = D1
    return (data, part, D)

#if __name__ == "__main__":
 #   FILENAME = 'database/segmentation.test.txt'

    # Calcula a matriz de dissimilaridades
  #  (E, Y, D) = proccessData(FILENAME)