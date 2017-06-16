from math import sqrt

def calculate_ranks(l):
    n = len(l)

    sumranks = 0
    dupcount = 0

    newarray = [0]*n

    for i in xrange(n):
        sumranks += i
        dupcount += 1

        if i == n - 1 or l[i][0] != l[i + 1][0]:
            averank = sumranks / float(dupcount) + 1

            for j in xrange(i - dupcount + 1, i + 1):
                newarray[j] = averank

            sumranks = 0
            dupcount = 0

    return newarray


def friedman_test(results, viewId):
    N = 300 #N de conjuto de dados
    k = 3 #N de algoritmos

    ranks = {}
    for j in results:
        ranks[j] = []

    for i in range(N):
        scores = []

        for j in results:
            scores.append((results[j][i], j))

        scores.sort()

        aux = calculate_ranks(scores) #Lista dos ranks

        for j in range(k):
            ranks[scores[j][1]].append(aux[j])

    testFile = open("tests-log-view-%d.txt" % viewId, "w")
    testFile.write("Ranks \n\n")
    for i in ranks:
        testFile.write(i + "\n")
        r = 1
        for j in ranks[i]:
            testFile.write("- Round %s : %s\n" % (r, j))
            r += 1
        testFile.write("\n")

    R = {} #Rank medio
    for r in ranks:
        R[r] = sum(ranks[r]) / N

    #Print Medias
    testFile.write("Medias\n\n")
    for i in R:
        testFile.write(i + ": " + str(R[i]) + "\n")

    testFile.write("\n")

    first = (12 * N) / float(k * (k + 1)) #Primeira parte da funcao
    second = 0 #Segunda parte da funcao

    #Somatorio Rj ^ 2
    for r in results:
        second += R[r]**2

    second -= ((k * (k + 1)**2) / 4)

    Xf = first * second

    Ff = ((N - 1) * Xf) / ((N * (k - 1)) - Xf)

    valorCritico = 1.85

    testFile.write("Friedman Test \n\n")
    testFile.write("Xf = " + str(Xf) + "\n")
    testFile.write("Ff = " + str(Ff) + "\n")
    testFile.write("Graus de liberdade = (%i, %i)\nValor Critico = 1.88\n\n" % ((k-1), ((k - 1) * (N - 1))))

    if Ff > valorCritico:
        testFile.write("Ff(%s) > Valor Critico(%s)\n"  % (str(Ff), str(valorCritico)))
        testFile.write("Hipotese H0 rejeitada\n\n")
        testFile.close()

        nemenyi_test(R, viewId)
    else:
        testFile.close()


def nemenyi_test(R, viewId):
    N = 300 #N de conjuto de dados
    k = 3 #N de algoritmos

    #Pos teste
    qa = 3.164 # k = 11,  alpha = 0.05

    CD = qa * sqrt((k * (k + 1)) / float((6 * N)))

    testFile = open("tests-log-view-%d.txt" % viewId, "a")

    testFile.write("Nemenyi Test (Pos Teste)\n\n")

    testFile.write("CD = " + str(CD) + "\n\n")

    #Comparacao entre os algoritmos = Comparacao da diferenca dos ranks e a diferenca critica
    count = 0
    for i in R:
        for j in R:
            if (j != i):
                if R[i] - R[j] > CD:
                    count += 1
                    testFile.write("%s e significativamente pior que %s.  (Rank %s = %s e Rank %s = %s)\n" %(i, j, i, str(R[i]), j, str(R[j])) )
        if count == 0:
            testFile.write("\n")

    testFile.close()


def compare(results, viewId):
    friedman_test(results, viewId)
