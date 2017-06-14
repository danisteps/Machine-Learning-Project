from operator import mul
from dissimilarity_matrix import proccessData
import math

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


def confidenceInterval(se):
    """ Computes a confidence interval for the error using alpha=0.05. """
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
        diff = diff / sig[j]
        expSum += diff
    expSum = -0.5 * expSum

    tmp = []
    tmp.append(TwoPi ** (-d * 0.5))
    tmp.append(prod(sig) ** (-0.5))
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

    return estimateError(testY, predictedY)


def kNN(trainX, trainY, testX, testY):
    pass


def majorRule(classifications, classes):
    """ For each data point, a classification ex: for x[k] classifications[k] = [0, 1, 1, 0] """
    C = []
    currentClass = 0
    argmax = -1
    for i in range(len(classifications)):
        for c in classes:
            count = classifications.count(c)
            if count > argmax:
                argmax = count
                currentClass = c
            C.append(c)
    return C


if __name__ == '__main__':
    FILENAME = 'database/segmentation.test.txt'

    (_, view1, view2, Y, _) = proccessData(FILENAME)
    (W, classes, Y) = getClasses(Y)

    d = len(view1[0])
    K = 10
    N = len(view1)
    foldSize = N / K

    print "%d %d" % (N, foldSize)

    for k in range(K):
        data = zip(view1, Y)
        train, test = getKFold(data, k, foldSize)

        trainX, trainY = [list(t) for t in zip(*train)]
        testX, testY = [list(t) for t in zip(*test)]

        (e_rate, se) = bayesian(trainX, trainY, testX, testY, W, d)
        print "%f %f" % (e_rate, se)
