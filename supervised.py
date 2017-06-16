from tests import compare
from operator import mul
from dissimilarity_matrix import proccessData
import math
import random

def estimateError(testY, predictedY):
    """ Estimates the error on the classification. """
    num_errors = 0
    n = len(testY)
    for i in range(n):
        if testY[i] != predictedY[i]:
            num_errors += 1
    e_rate = num_errors / float(n)
    tmp = e_rate*(1 - e_rate)
    se = math.sqrt(tmp / float(n))
    return (e_rate, se)


def confidenceInterval(testY, predictedY):
    """ Computes a confidence interval for the error using alpha=0.05. """
    (e_rate, se) = estimateError(testY, predictedY)
    tmp  = 1.96*se
    interval = [e_rate - tmp, e_rate + tmp]
    return (e_rate, se, interval)


def getClasses(Y):
    classes = []
    for y in Y:
        if not y in classes:
            classes.append(y)

    classesIds = {}
    idsToClasses = {}
    for y in range(len(classes)):
        classesIds[y] = classes[y]
        idsToClasses[classes[y]] = y

    proccessedY = map(lambda y: idsToClasses[y], Y)

    return (len(classes), classesIds, proccessedY)


def getKFold(data, k, foldSize):
    start = k * foldSize
    end = start + foldSize
    rest = data[:start] + data[end+1:]
    return (rest, data[start:end])


def prod(lst):
    """ Returns the product of the elements on the list. """
    return reduce(mul, lst, 1)


def estimateP(Y):
    """ Estimates P(w_j). """
    N = len(Y)
    count = {}
    for i in range(N):
        w = Y[i] # classe
        if w in count:
            count[w] += 1
        else:
            count[w] = 1

    P = [0 for i in range(len(count))]
    for w in count:
        P[w] = count[w] / float(N)

    return P


def conditional(x, w, d):
    """ Computes the conditional probability P(x | w_j). """
    TwoPi = 2*math.pi
    u = []
    sig = []
    for j in range(d):
        u_ij = 0
        for k in range(d):
            u_ij += x[k]
        u_ij = u_ij / float(d)
        u.append(u_ij)

    for j in range(d):
        l_ij = 0
        for k in range(d):
            diff = (x[k] - u[j]) ** 2.0
            l_ij += diff
        l_ij = l_ij / float(d)
        sig.append(l_ij)

    expSum = 0
    for j in range(d):
        diff = (x[j] - u[j]) ** 2.0
        diff = diff / sig[j] if sig[j] != 0 else 0
        expSum += diff
    expSum = -0.5 * expSum

    tmp = []
    tmp.append(TwoPi ** (-d * 0.5))
    tmp.append(prod(sig) ** (-0.5) if prod(sig) != 0 else 1)
    tmp.append(math.exp(expSum))

    return prod(tmp)


def bayes(x, w, d, c, P):
    """ Uses Maximum Likelihood and Bayes Theorem to estimate P(w_j | x). """
    tmp = []
    for k in range(c):
        res = conditional(x, k, d) * P[k]
        tmp.append(res)
    num = conditional(x, w, d) * P[w] * 1.0
    denum = sum(tmp)
    bt = num / denum
    return bt


def bayesian(trainX, trainY, testX, testY, W, d):
    P = estimateP(trainY)

    predictedY = []
    for x in testX:
        currentClass = None
        maxProb = None
        for w in range(W):
            bt = bayes(x, w, d, W, P)
            if maxProb == None or bt > maxProb:
                maxProb = bt
                currentClass = w
        predictedY.append(w)

    (e_rate, se, interval) = confidenceInterval(testY, predictedY)

    return (e_rate, se, interval, predictedY)


def _classifyKNN(trainX, trainY, testX, testY, k, W):
    P = []

    delta = lambda (x1, x2): abs(x1 - x2) ** 2
    dist = lambda x1, x2: math.sqrt(sum(map(delta, zip(x1, x2)))) # Euclidean distance

    m = len(trainX)
    n = len(testX)
    ids = range(m)

    ND = [[0 for j in range(m)] for i in range(n)] # Matrix filled with 0s

    for i in range(n):
        for j in range(m):
            ND[i][j] = dist(testX[i], trainX[j])

    for i in range(n):
        P.append([])

        N = zip(ids, ND[i]) # Neighbors
        N.sort(lambda (l, a), (m, b) : -1 if (a < b) else 1 if (a > b) else 0) # Sorts by distance
        N = N[:k] # K Nearest Neighbors

        freq = [0 for o in range(W)]
        for nb in N:
            n = nb[0]
            freq[trainY[n]] += 1

        for j in range(W):
            p = freq[j] / float(k)
            P[i].append(p)

    return P


def kNN(trainX, trainY, testX, testY, W):
    trainSize = len(trainX)
    validationSize = int(0.3 * trainSize)

    validTrainX = trainX[:trainSize-validationSize]
    validTrainY = trainY[:trainSize-validationSize]

    validX = trainX[trainSize-validationSize:]
    validY = trainY[trainSize-validationSize:]

    neighbors = None
    currentError = None
    for k in range(1, 20, 2):
        predictedY = _classifyKNN(validTrainX, validTrainY, validX, validY, k, W)
        (_, se) = estimateError(validY, predictedY)

        if (currentError == None) or (currentError > se):
            currentError = se
            neighbors = k
        else:
            break

    predictedY = _classifyKNN(trainX, trainY, testX, testY, neighbors, W)
    (e_rate, se, interval) = confidenceInterval(testY, predictedY)

    return (e_rate, se, interval, predictedY, neighbors)


def majorRule(classifications, Y, W):
    """ For each data point, a classification ex: for x[k] classifications[k] = [0, 1, 1, 0] """
    predictedY = []
    N = len(Y)

    for k in range(N):
        currentClass = 0
        argmax = -1
        for i in range(len(classifications)):
            for c in range(W):
                count = classifications[i].count(c)
                if count > argmax:
                    argmax = count
                    currentClass = c
        predictedY.append(c)

    return confidenceInterval(Y, predictedY)


def writeResults(f, e_rate, se, interval):
    f.write("(a) Error rate: %f\n" % e_rate)
    f.write("(b) SE: %f\n" % se)
    f.write("(c) Confidence Interval: [%f, %f]\n\n" % (interval[0], interval[1]))


def _runView(X, Y, viewId):
    classificadores = ["Bayesian", "KNN", "MajorRule"]
    errorResults = {}

    for c in classificadores:
        errorResults[c] = []

    resultsBay = open('part2-results-bayesian-view-%d.txt' % viewId, 'a')
    resultsKn = open('part2-results-knn-view-%d.txt' % viewId, 'a')
    resultsMaj = open('part2-results-major-view-%d.txt' % viewId, 'a')

    d = len(X[0])
    K = 10
    N = len(X)
    foldSize = N / K

    (W, classes, Y) = getClasses(Y)

    print "Running for View #%d\nData length:%d\nFold size:%d" % (viewId, N, foldSize)

    for j in range(30):
        data = zip(X, Y)
        random.shuffle(data)

        for k in range(K):
            print "Round %d" % (j*10+k+1)

            resultsBay.write("Round %d\n\n" % (j*10+k+1))
            resultsKn.write("Round %d\n\n" % (j*10+k+1))
            resultsMaj.write("Round %d\n\n" % (j*10+k+1))

            train, test = getKFold(data, k, foldSize)

            trainX, trainY = [list(t) for t in zip(*train)]
            testX, testY = [list(t) for t in zip(*test)]

            ## Bayesian Classifier
            print "Running Bayesian Classifier..."
            (e_rate_bay, se_bay, interval_bay, predictedBay) = bayesian(trainX, trainY, testX, testY, W, d)
            resultsBay.write("- Bayesian\n")
            writeResults(resultsBay, e_rate_bay, se_bay, interval_bay)
            errorResults["Bayesian"].append(e_rate_bay)

            ## k-NN Classifier
            print "Running k-NN Classifier..."
            (e_rate_kn, se_kn, interval_kn, predictedKn, neighbors) = kNN(trainX, trainY, testX, testY, W)
            resultsKn.write("- KNN (n = %d)\n" % neighbors)
            writeResults(resultsKn, e_rate_kn, se_kn, interval_kn)
            errorResults["KNN"].append(e_rate_kn)

            ## Major Rule Classifier
            print "Running Major Rule Classifier..."
            (e_rate_maj, se_maj, interval_maj) = majorRule([predictedKn, predictedBay], testY, W)
            resultsMaj.write("- Maj")
            writeResults(resultsMaj, e_rate_maj, se_maj, interval_maj)
            errorResults["MajorRule"].append(e_rate_maj)

    resultsBay.close()
    resultsKn.close()
    resultsMaj.close()

    compare(errorResults, viewId)


def run():
    FILENAME = 'database/segmentation.test.txt'

    print "Reading data..."
    (_, view1, view2, Y, _) = proccessData(FILENAME)

    _runView(view1, Y, 1)
    _runView(view2, Y, 2)

if __name__ == '__main__':
    run()
