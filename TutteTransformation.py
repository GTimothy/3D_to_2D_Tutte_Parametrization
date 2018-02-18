
import numpy
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gc


def loadVW(object_file):
    with open(object_file) as file:
        line = file.readline()
        print(line)
        while (line.startswith("#")):
            line = file.readline()
        i = 0
        while (line.startswith("v")):
            # split = line[2:].split(" ")
            # V.append({"x": split[0], "y": split[1], "z": split[2]})
            # x.append(float(split[0]))
            # y.append(float(split[1]))
            # z.append(float(split[2]))
            line = file.readline()
            i += 1
        print(i)
        W = numpy.zeros((i, i))
        print(W)
        while (line.startswith("f")):
            split = line[2:].split(" ")
            W[int(split[0]) - 1][int(split[1]) - 1] += 1
            W[int(split[0]) - 1][int(split[2]) - 1] += 1
            W[int(split[0]) - 1][int(split[0]) - 1] += 1
            W[int(split[1]) - 1][int(split[0]) - 1] += 1
            W[int(split[1]) - 1][int(split[2]) - 1] += 1
            W[int(split[1]) - 1][int(split[1]) - 1] += 1
            W[int(split[2]) - 1][int(split[0]) - 1] += 1
            W[int(split[2]) - 1][int(split[1]) - 1] += 1
            W[int(split[2]) - 1][int(split[2]) - 1] += 1

            line = file.readline()
        for j in range(0, i):
            W[j][j] += 1

    return W  # V, W, x, y, z


def getContourPoints(crsW):
    """

    :param crsW:
    :type crsW : csr_matrix
    :return:
    """
    val = crsW.data
    # J = crsW.indices
    rowPTR = crsW.indptr

    s = set()
    # s_bool = [False] * len(len(rowPTR))

    for i in range(0, len(rowPTR) - 1):
        for jj in range(rowPTR[i], rowPTR[i + 1]):
            # j = J[jj]
            v = val[jj]
            if v == 1:
                s.add(i)
                # s_bool[i]=True
                break
    return s#,s_bool


def getContourPoints_old(W):
    for i in range(0, len(W)):
        for j in range(0, len(W)):
            if W[i][j] == 1:
                list.append(i)
    return list


def iterationGaussSeidel(csrA, X, B, contour_points):
    val = csrA.data
    J = csrA.indices
    rowPTR = csrA.indptr

    for i in range(0, len(rowPTR) - 1):
        if i in contour_points:
        # if contour_points[i]:
            continue
        aii = 0.0
        s = 0
        for jj in range(rowPTR[i], rowPTR[i + 1]):
            j = J[jj]
            a = val[jj]
            if i == j:
                aii = a
            else:
                s += a * X[j]
        X[i] = (B[i] - s) / aii


def iterationGaussSeidel2(csrA, X, other_points):
    """

    :param csrA:
    :type csrA: csr_matrix
    :param X:
    # :type X: numpy.array
    :param other_points:
    :return:
    """
    val = csrA.data
    J = csrA.indices
    rowPTR = csrA.indptr

    for i in other_points:
        aii = 0.0
        s = 0
        for jj in xrange(rowPTR[i], rowPTR[i + 1]):
            j = J[jj]
            a = val[jj]
            if i == j:
                aii = a
            else:
                s += X[j]
        X[i] = s / aii


def initialXY(W, contourPointSet):
    # randTetas = numpy.random.uniform(low=0, high=2 * numpy.pi, size=(len(W),))
    # randCoef = numpy.random.uniform(low=0, high=1, size=(len(W),))
    # X = numpy.multiply(randCoef, numpy.cos(randTetas))
    # Y = numpy.multiply(randCoef, numpy.sin(randTetas))
    X = numpy.zeros(len(W))
    Y = numpy.zeros(len(W))

    # ordering contour naively
    c = contourPointSet.pop()
    contour_points_ordered = [c]
    while True:
        if len(contourPointSet) == 0:
            break
        else:
            for c2 in contourPointSet:
                if W[c][c2] == 1:
                    c = c2
                    contour_points_ordered.append(c)
                    contourPointSet.remove(c2)
                    break

    # plotting border points on a unit circle
    nbBord = len(contour_points_ordered)
    for i in range(0, nbBord):
        teta = 2 * numpy.pi * i / nbBord
        X[contour_points_ordered[i]] = numpy.cos(teta)
        Y[contour_points_ordered[i]] = numpy.sin(teta)

    return X, Y, contour_points_ordered


def buildAndShowNetworkxGraph(crsW, X, Y):
    import networkx as nx

    G = nx.Graph()

    val = crsW.data
    J = crsW.indices
    rowPTR = crsW.indptr
    for i in xrange(0, len(X)):
        G.add_node(i, pos=(X[i], Y[i]))
        # print(str(i)+"/"+str(len(X)))
        for jj in range(rowPTR[i], rowPTR[i + 1]):
            j = J[jj]
            if j > i:
                v = val[jj]
                if v > 0:
                    G.add_edge(i, j)
    print("Drawing...")
    nx.draw(G, pos=nx.get_node_attributes(G,'pos'),node_size=4)
    print("...done drawing")
    print("Showing...")
    plt.show()
    print("...done showing")


if __name__ == '__main__':
    print("main")
    # V, W, x, y, z = loadVW()
    object_file_path = "icosa1.obj"
    # object_file_path = "mask.obj"
    W = loadVW(object_file_path)
    # print(min(timeit.Timer("loadVW()", setup="from __main__ import loadVW").repeat(repeat=3, number=1)))
    # print(measure("loadVW()", "from __main__ import loadVW"))

    print("VW creation DONE")

    # W = [[3.,1.,2.,0.,1.],
    #      [1.,3.,2.,1.,0.],
    #      [2.,2.,4.,2.,2.],
    #      [0.,1.,2.,2.,1.],
    #      [1.,0.,2.,1.,4.]]
    # W = [[3., 1., 0., 1., 0., 2.],
    #      [1., 4., 1., 0., 2., 2.],
    #      [0., 1., 4., 1., 2., 0.],
    #      [1., 0., 1., 4., 2., 2.],
    #      [0., 2., 2., 2., 4., 2.],
    #      [2., 2., 0., 2., 2., 4.]]
    # W = [[3., 1., 0., 1., 0., 2., 0.],
    #      [1., 4., 1., 0., 2., 2., 2.],
    #      [0., 1., 4., 1., 2., 0., 0.],
    #      [1., 0., 1., 4., 2., 2., 2.],
    #      [0., 2., 2., 2., 4., 0., 2.],
    #      [2., 2., 0., 2., 0., 4., 2.],
    #      [0., 2., 0., 2., 2., 2., 4.]]
    crsW = sp.csr_matrix(W)

    print("W Sparsing DONE")
    contour_points_set = getContourPoints(crsW)
    # contour_points_set, contour_bool_list = getContourPoints(crsW)
    print(contour_points_set)


    # print(contour_points)
    # contour_points_list = list(contour_points)
    # contour_points_array = numpy.array(contour_points_list)
    # print(list(contour_points)[0])
    # print("Contour finding DONE")
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # ax.scatter(numpy.asarray(x), numpy.asarray(y), numpy.asarray(z))
    # #
    # ax.scatter(numpy.asarray(x)[contour_points_array], numpy.asarray(y)[contour_points_array],
    #            numpy.asarray(z)[contour_points_array], color='red')


    X, Y, contour_pts_ordered = initialXY(W, contour_points_set)
    contour_points = set(contour_pts_ordered)
    other_points = set(range(0,len(W)))-contour_points
    print(other_points)
    print("CONTOUR DONE")

    print ()

    # clear W from memory
    del W
    print("Collecting...")
    gc.collect()
    print("\t...done")

    #
    plt.plot(X, Y, 'r.')
    plt.show()
    #
    plt.figure(2)
    iterationGaussSeidel2(crsW, X, other_points)
    iterationGaussSeidel2(crsW, Y, other_points)
    plt.plot(X, Y, 'r.')
    plt.show()

    plt.figure(3)
    iterationGaussSeidel2(crsW, X, other_points)
    iterationGaussSeidel2(crsW, Y, other_points)
    plt.plot(X, Y, 'r.')
    plt.show()

    plt.figure(4)
    iterationGaussSeidel2(crsW, X, other_points)
    iterationGaussSeidel2(crsW, Y, other_points)
    plt.plot(X, Y, 'r.')
    plt.show()

    plt.figure(5)
    for i in xrange(100):
        print(str(i)+"/"+str(100))
        iterationGaussSeidel2(crsW, X, other_points)
        iterationGaussSeidel2(crsW, Y, other_points)

    try:
        buildAndShowNetworkxGraph(crsW, X, Y)
    except ImportError:
        print("could not import networkx library, is the networkx python package installed?")



