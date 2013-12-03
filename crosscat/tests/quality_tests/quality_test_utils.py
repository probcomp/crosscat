import numpy

def bincount(X, bins=None):
	""" Counts the elements in X according to bins.
		Inputs:
			- X: A 1-D list or numpt array or integers.
			- bins: (optional): a list of elements. If bins is None, bins will
				be range range(min(X), max(X)+1). If bins is provided, bins
				must contain at least each element in X
		Outputs: 
			- counts: a list of the number of element in each bin
		Ex:
			>>> import quality_test_utils as qtu
			>>> X = [0, 1, 2, 3]
			>>> qtu.bincount(X)
			[1, 1, 1, 1]
			>>> X = [1, 2, 2, 4, 6]
			>>> qtu.bincount(X)
			[1, 2, 0, 1, 0, 1]
			>>> bins = range(7)
			>>> qtu.bincount(X,bins)
			[0, 1, 2, 0, 1, 0, 1]
			>>> bins = [1,2,4,6]
			>>> qtu.bincount(X,bins)
			[1, 2, 1, 1]
	"""

	if not isinstance(X, list) and not isinstance(X, numpy.ndarray):
		raise TypeError('X should be a list or a numpy array')

	if isinstance(X, numpy.ndarray):
		if len(X.shape) > 1:
			if X.shape[1] != 1:
				raise ValueError('X should be a vector')

	Y = numpy.array(X, dtype=int)
	if bins == None:
		minval = numpy.min(Y)
		maxval = numpy.max(Y)

		bins = range(minval, maxval+1)

	if not isinstance(bins, list):
		raise TypeError('bins should be a list')

	counts = [0]*len(bins)

	for y in Y:
		bin_index = bins.index(y)
		counts[bin_index] += 1

	assert len(counts) == len(bins)
	assert sum(counts) == len(Y)

	return counts

