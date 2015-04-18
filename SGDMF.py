__author__ = 'hakim'

"""
Stochastic gradient descent for one block of a matrix
"""

import numpy as np
import math

def sgd(H,sigma_m,l,Ni_,N_j,tau0,beta,iterator):
    """
    Update matrices W and H
    :param H: F x Movies - {movie_id : [column in H]}
    :param sigmaM: sum(m_ib) Total number of SGD updates made across all strata in all previous iterations
    :param l: Regularization parameter
    :param Ni_: Number of movies user i has rated
    :param N_j: Number of users who have rated movie j
    :param tau0: tau0 to set epsilon
    :param beta: beta to set epsilon
    :param: iterator: [(uid,([movie,rating],w_uid))]
    :return: Updated matrices W and H
    """
    n = sigma_m.value   # Keep track of the number of updates
    Wnew = {}
    Hnew = {}
    n_dash = 0

    # go over each user_id in V
    # and each movie this user has rated
    # and perform gradient update on this point
    for (uid,tpl) in iterator:
        # tpl : ([(movieid,rating)],w_uid)
        movie_ratings = tpl[0]
        Wuid = tpl[1]
        for movie_rating in movie_ratings:
            # Compute the epsilon_n used in the gradient update
            eps_n = epsilon(tau0.value, n, beta.value)
            # increment the number of updates by 1
            n = n + 1
            n_dash = n_dash + 1
            mid = movie_rating[0]
            rating = movie_rating[1]
            # Perform update on W_uid and H_mid
            if mid in Hnew:
                H[mid] = Hnew[mid]

            if uid in Wnew:
                # If uid has already been updated in Wnew, use the updated value
                Wnew[uid] = Wnew[uid] - eps_n * gradientWi(rating, Wnew[uid], H[mid], Ni_.value[uid], l.value)
                Hnew[mid] = H[mid] - eps_n * gradientHj(rating, Wnew[uid], H[mid], N_j.value[mid], l.value)
            else:
                # else create a new value
                Wnew[uid] = Wuid - eps_n * gradientWi(rating, Wuid, H[mid], Ni_.value[uid], l.value)
                Hnew[mid] = H[mid] - eps_n * gradientHj(rating, Wuid, H[mid], N_j.value[mid], l.value)

    for uid in Wnew:
        yield (uid, ('w',Wnew[uid]))

    for movie_id in Hnew:
        yield (movie_id, ('h',Hnew[movie_id]))

    yield (n_dash, ('n',n_dash))


def loss(v, Wi_, H_j):
    """
    Compute the loss function
    :param v: Vij entry on which the gradient is being computed
    :param Wi_: Row i of W - numpy.array([row i in W])
    :param H_j: Column j of H - numpy.array([column j in H])
    :return: NZSL
    """
    return math.pow(v - np.dot(Wi_,H_j),2)

def gradientWi(v,Wi_,H_j,Ni_,l):
    """
    Compute the gradient of the NZSL loss function with respect to W
    :param v: Vij entry on which the gradient is being computed
    :param Wi_: Row i of W - numpy.array([row i in W])
    :param H_j: Column j of H - numpy.array([column j in H])
    :param Ni_: Number of movies user i has rated
    :param l: regularization parameter
    :return: gWi - Gradient of the loss function with respect to Wi
    """
    WH = np.dot(Wi_,H_j)
    gWi = -2 * (v-WH) * H_j + 2 * (l/Ni_) * Wi_
    return gWi

def gradientHj(v,Wi_,H_j,N_j,l):
    """
    Compute the gradient of the NZSL loss function with respect to W
    :param v: Vij entry on which the gradient is being computed
    :param Wi_: Row i of Wi
    :param H_j: Column j of H
    :param N_j: Number of users who have rated movie j
    :param l: regularization parameter
    :return: gWi - Gradient of the loss function with respect to Wi
    """
    WH = np.dot(Wi_,H_j)
    gWi = -2 * (v-WH) * Wi_ + 2 * (l/N_j) * H_j
    return gWi

def epsilon(tau0,n,beta):
    """
    Compute epsilon_n used in the gradient update
    :param tau0:
    :param n:
    :param beta:
    :return:
    """
    return 1 / math.pow(tau0 + n,beta)