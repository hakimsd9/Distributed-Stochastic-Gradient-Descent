__author__ = 'Hakim'
"""
Implementation of the Distributed Stochastic Gradient Descent algorithm using Spark
https://people.mpi-inf.mpg.de/~chteflio/publications/gemulla11dsgd-short.pdf
"""

from pyspark import SparkContext
import numpy as np
import functools
import SGDMF
import csv
import math

sc = SparkContext("local", "SGDMF Job")

def dsgd(F,num_machines,num_iters,beta,l,train,w_out,h_out):
    """
    Apply the Distributed Stochastic Gradient Descent algorithm over the iterations
    :param F: Number of factors
    :param numMachines: Number of workers
    :param numIters: Number of iterations
    :param beta: Parameter beta
    :param l: Parameter lambda
    :param train: csv file containing the training matrix V
    :param w_out: csv file containing the learned matrix W
    :param h_out: csv file containing the learned matrix H
    :return:
    """
    tau0 = 100
    sigma_m = 0  # sum the number of all previous updates

    # Read the input file and create V, Ni_ and N_j
    (V,n_users,n_movies) = create_v(train)

    W = {}
    H = {}

    # Randomly initialize W and H
    for user in V:
        W[user] = np.random.rand(F)

    for movie in n_movies:
        H[movie] = np.random.rand(F)

    # Make V and W RDDs
    dist_V = sc.parallelize(V.items(),num_machines).persist()
    dist_W = sc.parallelize(W.items(),num_machines).persist()

#    # Partition V
#    distV = distV.partitionBy(numMachines).persist()

    # l, n_users, n_movies, tau0, beta
    broadcast_l = sc.broadcast(l)
    broadcast_n_users = sc.broadcast(n_users)
    broadcast_n_movies = sc.broadcast(n_movies)
    broadcast_tau0 = sc.broadcast(tau0)
    broadcast_beta = sc.broadcast(beta)

    # Iterate over the strata
    for iteration_number in range(num_iters):
        # Filter matrix V to define a strata
        filtered_V = dist_V.mapPartitionsWithIndex(functools.partial(filter_V,num_machines,iteration_number),True)
        filtered_V.count(),filtered_V.getNumPartitions()

        # Join the rows of W corresponding to the stratum
        vw = filtered_V.join(dist_W)
        vw.count(),vw.getNumPartitions()
        # broadcast the current value of sigmaM
        broadcast_sigma = sc.broadcast(sigma_m)
        # Apply the SGD algorithm on each block of the strata in parallel
        update = vw.mapPartitions(functools.partial(SGDMF.sgd,H,broadcast_sigma,broadcast_l,broadcast_n_users,broadcast_n_movies,broadcast_tau0,broadcast_beta), True)
        update.getNumPartitions()
        # Update W, H and n
        ws = update.filter(lambda (x,y) : y[0] == 'w').mapValues(lambda val: val[1])
        hs = update.filter(lambda (x,y) : y[0] == 'h')
        new_n = update.filter(lambda (x,y) : y[0] == 'n').map(lambda val: val[0])

        new_w = dist_W.subtractByKey(ws).union(ws)
        dist_W.unpersist()
        dist_W = new_w
        dist_W = dist_W.partitionBy(num_machines).persist()

        h_col = hs.collect()

        for (m,tpl) in h_col:
            H[m] = tpl[1]

        # Update sigmaM
        sigma_m += new_n.sum()

    # Sort the movies and user ids in order to write the learned matrices W and H
    sorted_movies = sorted(H.keys())
    sorted_W_by_user = sorted(dist_W.collect(), key = lambda tup: tup[0])

    # Write W to a csv file
    with open(w_out,'w') as dst_W:
        wrt = csv.writer(dst_W,delimiter = ',')
        for userW in sorted_W_by_user:
            to_write = []
            for wf in userW[1]:
                to_write.append(wf)
            wrt.writerow(to_write)

    # convert H to a numpy matrix
    # transpose it and write it to csv
    e = []
    for k in sorted_movies:
        e.append(H[k].tolist())
    mat = np.asmatrix(e)
    mat = np.transpose(mat)

    np.savetxt(h_out,mat,delimiter=',')


def compute_loss(v,w,h):
    """
    Compute the loss function given learned matrices W and H
    Used to plot the loss function over iterations
    :param V: Ratings matrix
    :param W:
    :param H:
    :return:
    """
    loss = 0
    for (u,w_u) in w:
        for tpl in v[u]:
            loss += math.pow(tpl[1] - np.dot(w_u,h[tpl[0]]),2)
    return loss


def filter_V(num_machines,iteration_number,split_index,iterator):
    """
    Filter user and movie to keep only the relevant blocks in the stratum
    :param num_machines: number of workers
    :param iteration_number:number of iterations
    :param split_index: current worker
    :param iterator: iterator over the rows of V
    :return: Filtered matrix V that contains only the values for a strata
    """
    filtered_v = []
    for (uid, m_lst) in iterator:
        filtered_lst = []
        for lst in m_lst:
            movie = lst[0]
            if (movie + iteration_number) % num_machines == split_index:
                filtered_lst.append(lst)
        if len(filtered_lst) > 0:
            filtered_v.append((uid,filtered_lst))
    return filtered_v


def create_v(ratings_file):
    """
    Create the V matrix - dictionary V = {user_id:[(movie_id:rating)]}
    Count the number of movies each user has rated
    Count the number of users who have rated each movie
    :param ratingsFile
    :return: ratings file as a dictionary V
    """
    v = {}
    n_users = {} # Number of movies each user has rated {user_id : number_of_movies_rated}
    n_movies = {}    # Number of users who have rated each movie {movie_id : number_of_raters}
    tf = sc.textFile(ratings_file).collect()
    for line in tf:
        [uid, mid, r] = line.split(',')
        uid = int(uid)
        mid = int(mid)
        r = int(r)
        if uid in v:
            v[uid].append((mid,r))
            n_users[uid] += 1
            if mid in n_movies:
                n_movies[mid] += 1
            else:
                n_movies[mid] = 1
        else:
            v[uid] = [(mid,r)]
            n_users[uid] = 1
            if mid in n_movies:
                n_movies[mid] += 1
            else:
                n_movies[mid] = 1
    return (v,n_users,n_movies)


if __name__ == '__main__':
    from sys import argv
    # argv[1] F
    # argv[2] numMachines
    # argv[3] numIters
    # argv[4] beta
    # argv[5] lambda
    # argv[6] autolab_train.csv
    # argv[7] w.csv
    # argv[8] h.csv
    dsgd(int(argv[1]), int(argv[2]), int(argv[3]), float(argv[4]), float(argv[5]), argv[6], argv[7], argv[8])
