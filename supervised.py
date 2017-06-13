import math

### NEEDS TESTING!!!!!!

def prod(lst):
    """ Returns the product of the elements on the list. """
    return reduce(mul, lst, 1)

def estimateP(data):
     """ Estimates P(w_j). """
    N = len(data)
    P = []
    count = {}
    for i in range(N):
        w = data[i][1] # classe
        if w in count:
            count[w] += 1
        else:
            count[w] = 1

    for w in count:
        P.append(w, count[w] / float(N))

    return P

def conditional(x, w, d):
    """ Computes the conditional probability P(x | w_j). """
    2pi = 2*math.pi
    u = []
    sig = []
    n = len(x[0])
    for j in range(d):
        u_ij = 0
        for k in range(n):
            u_ij += x[j]
        u_ij = u_ij / float(n)
        u.append(u_ij)

    for j in range(d):
        l_ij = 0
        for k in range(n):
            diff = (x[j] - u[j]) ** 2.0
            l_ij += diff
        l_ij = l_ij /float(n)
        sig.append(l_ij)

    expSum = 0
    for j in range(d):
        diff = (x[j] - u[j]) ** 2.0
        diff = diff / sig[j]
        expSum += diff
    expSum = -0.5 * expSum

    tmp = []
    tmp.append(2pi ** (-d * 0.5))
    tmp.append(prod(sig) ** (-0.5))
    tmp.append(math.exp(expSum))

    return prod(tmp)

def bayes(x, w, d, c, P):
    """ Uses Maximum Likelihood and Bayes Theorem to estimate P(w_j | x). """
    tmp = []
    for k in range(c):
        res = conditional(x, k, d)
        tmp.append(res)
    num = conditional(x, w, d) * P[w] * 1.0
    denum = prod(tmp)
    bt = num / denum
    return bt

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
