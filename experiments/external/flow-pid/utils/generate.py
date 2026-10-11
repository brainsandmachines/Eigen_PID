#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy
from scipy.stats import wishart, ortho_group


def generate_randomBC(dT,dX,dY):
	# Generate random B and C matrices
	hx = np.random.randn(dX,dT)
	hy = np.random.randn(dY,dT)
	return hx, hy


def rotn_mat_2d(theta):
	return np.array([[np.cos(theta), -np.sin(theta)],
					 [np.sin(theta), np.cos(theta)]])


def sample_mult_poisson(n, lamda_m, w_x, w_y, lamda_x, lamda_y):
	"""
	XXX: not yet fully general - X and Y have no shared randomness except M.
	"""

	d_M = lamda_m.size
	d_X = lamda_x.size
	d_Y = lamda_y.size

	rng = np.random.default_rng()
	m = rng.poisson(lam=lamda_m, size=(n, 2)).T

	x = np.zeros((d_X, n))
	for j in range(d_X):
		x[j, :] = (sum(rng.binomial(m[i], w_x[j, i]) for i in range(d_M))
				   + rng.poisson(lam=lamda_x[j], size=n))

	y = np.zeros((d_Y, n))
	for j in range(d_Y):
		y[j, :] = (sum(rng.binomial(m[i], w_y[j, i]) for i in range(d_M))
				   + rng.poisson(lam=lamda_y[j], size=n))

	return m, x, y


def random_rotation_mxy(cov, dm, dx, dy, seed=None):
	"""
	Perform random rotations on the M, X and Y components of a covariance matrix.
	Returns a copy.
	"""
	cov = cov.copy()

	#rng = np.random.default_rng(seed)

	#R = ortho_group.rvs(dm, seed=rng)
	#cov[:dm, :] = R @ cov[:dm, :]
	#cov[:, :dm] = cov[:, :dm] @ R.T
	#R = ortho_group.rvs(dx, seed=rng)
	#cov[dm:dm+dx, :] = R @ cov[dm:dm+dx, :]
	#cov[:, dm:dm+dx] = cov[:, dm:dm+dx] @ R.T
	#R = ortho_group.rvs(dy, seed=rng)
	#cov[dm+dx:, :] = R @ cov[dm+dx:, :]
	#cov[:, dm+dx:] = cov[:, dm+dx:] @ R.T

	R = ortho_group.rvs(dm)
	cov[:dm, :] = R @ cov[:dm, :]
	cov[:, :dm] = cov[:, :dm] @ R.T
	R = ortho_group.rvs(dx)
	cov[dm:dm+dx, :] = R @ cov[dm:dm+dx, :]
	cov[:, dm:dm+dx] = cov[:, dm:dm+dx] @ R.T
	R = ortho_group.rvs(dy)
	cov[dm+dx:, :] = R @ cov[dm+dx:, :]
	cov[:, dm+dx:] = cov[:, dm+dx:] @ R.T

	return cov


def move_cov_block_to_end(cov, i, j):
	"""
	Moves the variables indexed by i:j to the end of the covariance matrix.
	Assumes the covariance matrix is symmetric.
	Negative indices will not work.
	"""
	if i >= j:
		raise ValueError('Index i must be less than index j')
	cov_new = np.delete(cov, np.s_[i:j], axis=0)
	cov_new = np.delete(cov_new, np.s_[i:j], axis=1)
	diag_block = cov[i:j, i:j]
	nondiag_block = np.delete(cov[i:j, :], np.s_[i:j], axis=1)
	return np.block([[cov_new, nondiag_block.T], [nondiag_block, diag_block]])


def swap_x_and_y(cov, dm, dx, dy):
	return move_cov_block_to_end(cov, dm, dm+dx), dm, dy, dx


def merge_covs(cov1, cov2, dm1, dx1, dy1, dm2=None, dx2=None, dy2=None,
			   random_rotn=False, seed=None):
	"""
	Merges two covariance matrices of given M, X and Y dimensions.
	Performs a random rotation on M, X and Y after merging, if random_rotn is True.
	"""
	if dm2 is None:
		dm2 = dm1
	if dx2 is None:
		dx2 = dx1
	if dy2 is None:
		dy2 = dy1

	dm = dm1 + dm2
	dx = dx1 + dx2
	dy = dy1 + dy2

	zero_block = np.zeros((cov1.shape[0], cov2.shape[0]))
	cov = np.block([[cov1, zero_block], [zero_block.T, cov2]])

	# Move required blocks to end: order of ops is important!
	cov = move_cov_block_to_end(cov, dm1, dm1 + dx1 + dy1)
	# Diag now reads: M1, M2, X2, Y2, X1, Y1
	cov = move_cov_block_to_end(cov, dm, dm + dx2)
	# Diag now reads: M1, M2, Y2, X1, Y1, X2
	cov = move_cov_block_to_end(cov, dm + dy2 + dx1, dm + dy2 + dx1 + dy1)
	# Diag now reads: M1, M2, Y2, X1, X2, Y1
	cov = move_cov_block_to_end(cov, dm, dm + dy2)
	# Diag now reads: M1, M2, X1, X2, Y1, Y2

	if random_rotn:
		cov = random_rotation_mxy(cov, dm, dx, dy, seed=seed)

	return cov, dm, dx, dy


def generate_cov_from_config(**kwargs):
	"""
	Generate a covariance matrix from a given "configuration".
	The configuration can be one of two sets of keyword arguments. The first
	set generates a new covariance matrix corresponding to the parameters:
		d: tuple of (int, int, int)
			Represents (dm, dx, dy). Only (2, 2, 2) currently supported.
		gain: tuple of (float, float)
			Represents (gain_x, gain_y), i.e., gain for X and Y respectively
		theta: float
			Counterclockwise rotation to be applied. Makes most sense to be in
			the interval [0, np.pi].
	The second set combines the covariance matrices from two previously
	generated configurations.
		pid_table: pd.DataFrame
			Table containing columns dm, dx, dy, gain_x, gain_y, and theta,
			corresponding to the first configuration
		id1: int
			Row index (loc) of the first configuration to choose
		id2: int
			Row index (loc) of the second configuration to choose
		random_rotn: False (use as default) or None or int seed
			Whether to use random rotations while combining covariance matrices
			(if not False), and what random seed to use (any int). Set as None
			to use random rotations without setting a seed.
	"""

	if all(item in kwargs for item in ['d', 'gain', 'theta']):
		if kwargs['d'] != (2, 2, 2):
			raise ValueError('Only d=(2, 2, 2) is supported')
		dm, dx, dy = kwargs['d']
		gain_x, gain_y = kwargs['gain']
		theta = kwargs['theta']

		sigm = np.eye(dm)
		hx = np.array([[gain_x, 0], [0, 1]]) @ rotn_mat_2d(-theta)
		hy = np.array([[1, 0], [0, gain_y]])
		sigx_m = np.eye(dx)
		sigy_m = np.eye(dy)
		sigw = np.zeros((dx, dy))

		# Covariance matrix construction for both unique or redundant
		cov = np.block([[sigm, sigm @ hx.T, sigm @ hy.T],
						[hx @ sigm, hx @ sigm @ hx.T + sigx_m, hx @ sigm @ hy.T + sigw],
						[hy @ sigm, hy @ sigm @ hx.T + sigw.T, hy @ sigm @ hy.T + sigy_m]])

		return cov, dm, dx, dy

	elif all(item in kwargs for item in ['pid_table', 'id1', 'id2', 'random_rotn']):
		pid_table = kwargs['pid_table']
		id1 = kwargs['id1']
		id2 = kwargs['id2']
		random_rotn = kwargs['random_rotn']

		if random_rotn is False:
			seed = None  # Doesn't matter what seed is: merge_covs won't use it
		else:
			seed = random_rotn
			random_rotn = True

		covs = []
		ds = []
		config1_params = ['dm', 'dx', 'dy', 'gain_x', 'gain_y', 'theta']
		config2_params = ['id1', 'id2', 'random_rotn']

		for index in [id1, id2]:
			if pid_table.loc[index, config1_params].notna().all():
				dm, dx, dy = pid_table.loc[index, ['dm', 'dx', 'dy']]
				gain_x, gain_y = pid_table.loc[index, ['gain_x', 'gain_y']]
				cov, *d = generate_cov_from_config(d=(dm, dx, dy),
												   gain=(gain_x, gain_y),
												   theta=pid_table['theta'][index])
			elif pid_table.loc[index, config2_params].notna().all():
				sub_id1, sub_id2, sub_rotn = pid_table.loc[index, config2_params]
				cov, *d = generate_cov_from_config(pid_table=pid_table,
												   id1=sub_id1, id2=sub_id2,
												   random_rotn=sub_rotn)
			else:
				raise ValueError('Index %d in pid_table does not have a valid '
								 'configuration' % index)
			covs.append(cov)
			ds.append(d)

		# TODO: Test the way random_rotn and seed are being used here
		return merge_covs(covs[0], covs[1], *ds[0], *ds[1], random_rotn, seed)

	else:
		raise ValueError('Unrecognized keyword combination %s' % list(kwargs.keys()))


def generate_system(system='random',dm=5,dx=5,dy=5,dof=0,V=None):
	"""
	Generate appropriate gain and covariance matrices

	Param
	-----------
	system : str
		Specify which kind of system to generate. Either 'random' for
		a random covariance matrix or one of the other special
		examples

	dm : int
		dimensionality of M vector

	dx : int
		dimensionality of X vector

	dy : int
		dimensionality of Y vector

	dof : int
		number of degrees of freedom for generating covariance matrix
		from Wishart distribution. if n < dm+dx+dy, it will
		default to n = dm+dx+dy
		***only applies when system=='random'

	V : np.ndarray [dm+dx+dy,dm+dx+dy]
		Wishart scale matrix. defaults to identity.
		***only applies when system=='random'

	Returns
	-------
	hx : np.ndarray [dx,dm]
		Channel gain matrix for M->X (i.e. X|M has mean hx.dot(M))

	hy : np.ndarray [dy,dm]
		Channel gain matrix for M->Y

	hxy : np.ndarray [dx+dy,m]
		Channel gain matrix for M->(X,Y)

	sigx : np.ndarray [dx,dx]
		Channel covariance matrix for M->X (i.e. X|M has cov sigx)

	sigy : np.ndarray [dy,dy]
		Channel covariance matrix for M->Y

	sigxy : np.ndarray [dx+dy,dx+dy]
		Channel covariance matrix for M->(X,Y)

	covxy : np.ndarray [dx+dy,dx+dy]
		Covariance matrix for (X,Y)

	sigm : np.ndarray [dm,dm]
		Covariance matrix for M

	"""
	systems = ['random',
			   'both fully unique',
			   'only one unique',
			   'one is very unique',
			   'some of each',
			   'univariate',
			   'overdetermined']

	assert system in systems, f'system must be on of {systems}'

	# generate random system with Wishart distributed covariance matrix
	if system == 'random':

		# generate the full covariance matrices and submatrices
		mean = np.zeros(dm+dx+dy)
		if V is None:
			V = np.eye(dm+dx+dy)
		if dof < dm+dx+dy:
			dof = dm+dx+dy
		G = np.random.multivariate_normal(mean,V,dof)
		cov = np.dot(G,G.T)
		covm = cov[:dm,:dm]
		covx = cov[dm:dm+dx,dm:dm+dx]
		covy = cov[dm+dx:dm+dx+dy,dm+dx:dm+dx+dy]
		covxy = cov[dm:dm+dx+dy,dm:dm+dx+dy]
		covx_m = cov[:dm,dm:dm+dx].T
		covy_m = cov[:dm,dm+dx:dm+dx+dy].T
		covxy_m = cov[:dm,dm:dm+dx+dy].T
		# channel gain matrices (using multivariate conditional mean)
		hx = covx_m.dot(np.linalg.inv(covm))
		hy = covy_m.dot(np.linalg.inv(covm))
		hxy = covxy_m.dot(np.linalg.inv(covm))
		# channel covariance matrices (using mvar conditional covariance)
		sigm = covm
		sigx = covx - covx_m.dot(np.linalg.inv(covm)).dot(covx_m.T)
		sigy = covy - covy_m.dot(np.linalg.inv(covm)).dot(covy_m.T)
		sigxy = covxy - covxy_m.dot(np.linalg.inv(covm)).dot(covxy_m.T)

	# would expect both to be equally deficient
	elif system == 'both fully unique':
		dm = 2
		dx = 1
		dy = 1

		# X is M1 plus noise
		hx = np.array([1,0]).reshape(1,2)
		# Y is M2 plus noise
		hy = np.array([0,1]).reshape(1,2)

		sigm = np.eye(dm)+0.3*np.eye(dm).T
		sigx = 0.0001*np.eye(dx)
		sigy = 0.0001*np.eye(dy)

	# would expect one to be deficient but not the other
	elif system == 'only one unique':
		dm = 2
		dx = 2
		dy = 2

		# make both X and Y exact copies of M
		hx = np.eye(2)
		hy = np.eye(2)

		sigm = np.eye(dm)
		# but X is noisier
		sigx = 4*np.eye(dx)
		sigy = 0.01*np.eye(dy)

	# would X is fully redundant
	elif system == 'one is very unique':
		dm = 10
		dx = 1
		dy = dm

		# X only gets dx of the dm dimensions
		hx = np.hstack((np.eye(dx),np.zeros((dx,dm-dx))))
		# Y gets a copy of everything
		hy = np.eye(dy)

		sigm = np.eye(dm)
		sigx = 1*np.eye(dx)
		sigy = 1*np.eye(dy)

	# expect some uniqueness and some redundancy
	elif system == 'some of each':
		dm = 3
		dx = 2
		dy = 2

		# make both X and Y get some of dim 1 and one of the other 2 dimensions
		hx = np.array([[1,0,0],[0,0,1]])
		hy = np.array([[np.random.uniform(-1,1),0,0],[0,1,0]])

		sigm = np.eye(dm)
		sigx = np.eye(dx)
		sigy = np.eye(dy)

	# the system considered by Barrett
	elif system == 'univariate':
		dm = 1
		dx = 1
		dy = 1

		hx = np.random.uniform(-1,1,size=(dx,dm))
		hy = np.random.uniform(-1,1,size=(dx,dm))

		sigm = np.eye(dm)
		sigx = np.eye(dx) - hx.dot(sigm).dot(hx.T)
		sigy = np.eye(dy) - hy.dot(sigm).dot(hy.T)

	# for the non-random systems, get necessary matrices for MI calculation
	if system != 'random':
		hxy = np.vstack((hx,hy))
		sigxy = np.bmat([[sigx,np.zeros((dx,dy))],[np.zeros((dy,dx)),sigy]])
		covx = sigx + hx.dot(sigm).dot(hx.T)
		covy = sigy + hy.dot(sigm).dot(hy.T)
		covxy = np.bmat([[covx,np.zeros((dx,dy))],[np.zeros((dy,dx)),covy]])

	# whiten
	hx = scipy.linalg.sqrtm(np.linalg.inv(sigx)).dot(hx)
	sigx = np.eye(dx)
	hy = scipy.linalg.sqrtm(np.linalg.inv(sigy)).dot(hy)
	sigy = np.eye(dy)

	return hx, hy, hxy, sigx, sigy, sigxy, covxy, sigm