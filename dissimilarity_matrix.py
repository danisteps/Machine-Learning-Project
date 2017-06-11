import math

def computeDissimilarityMatrix(filename):
    count = 0
    matrix = []

    f = open(filename, 'rb+')

    for line in f:
        if (count > 4):
            matrix.append([j for j in line.split(',')])
            var = line.split(',')

        count = count + 1

    attCount = len(matrix[0])
    view1AttCount = 9
    view2AttCount = 10
    view1Dis = [[0 for i in range(view1AttCount)] for j in range(view1AttCount)]
    view2Dis = [[0 for i in range(view2AttCount)] for j in range(view2AttCount)]

    xi = 0;
    while (xi < view1AttCount):
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
    while (xi < view2AttCount):
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