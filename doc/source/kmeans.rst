


K-Means
-------

The code is based upon the Wikipedia description [wikikmeans]_. The objective of the algorithm is to minimize the intra-cluster variance, given by

.. math::
    V = \sum_{i=1}^{K} \sum_{x_j \in S_i} (x_j - \mu_i)^2

where :math:`K` is the total number of clusters, :math:`S_i` the set of points in the :math:`i`'th cluster, and :math:`\mu_i` the center of the :math:`i`'th cluster.

.. todo::

    Describe algorithm
    Initialization
    Stopping

When k-means has minimized the intra-cluster variance, it might not have found the global minimum of variance.
It is therefore a good idea to run the algorithm several times, and use the clustering result with the best intra-cluster variance.
The intra-cluster variance can be obtained from the method ``find_intra_cluster_variance``.

.. [wikikmeans] http://en.wikipedia.org/w/index.php?title=K-means_algorithm&oldid=280702005

K-Means Example
^^^^^^^^^^^^^^^

To demostrate the K-means algorithm, we will construct a simple example with three clusters with Gaussian distribution. 

.. literalinclude:: ../../examples/kmeans_2d.py


The second plot of this example will give the clustering result from the algorithm.

.. image:: figures/kmeans_2d.png


Application Programming Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pypr.clustering.kmeans
   :members:



