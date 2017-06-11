import dissimilarity_matrix

if __name__ == "__main__":
    FILENAME = 'database/segmentation.test.txt'

    #Calcula a matriz de dissimilariidades
    (V1, V2) = dissimilarity_matrix.computeDissimilarityMatrix(FILENAME)