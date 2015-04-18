Spark implementation of the Distributed Stochastic Gradient Descent Algorithm (https://people.mpi-inf.mpg.de/~chteflio/publications/gemulla11dsgd-short.pdf)

Usage - spark-submit dsgd_mf.py number_of_factors number_of_workers number_of_iterations beta lambda autolab_train.csv w.csv h.csv 


dsgd_mf.py:
	create_V: Read the data file (<user_id>,<movie_id>,<rating>)
		  Create a dictionary V {user_id:[(movie_id,rating)]} 
		  Compute: - Ni_ - Number of movies user i has rated
			   - N_j - Number of users who have rated movie j
	filter_V: Filter the movies that will be processed in worker j so as to create a valid sequence of strata (all the blocks should be interchangeable)

	compute_loss: Helper function to compute the loss resulting from a factorization W,H - Used to plot the loss as a function of the number of iterations

SGDMF.py: Apply Stochastic gradient descent to a given strata
	sgd: Compute gradient update on all the points
	loss: Helper function to compute the loss
	gradientWi: Compute the gradient with respect to Wi
	gradientHj: Compute the gradient with respect to Hj
	epsilon: Compute the value epsilon_n used in the gradient update (decays with the number of iterations)
