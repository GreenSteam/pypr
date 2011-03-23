#!/usr/bin/env python

# See: http://docs.python.org/distutils/

from distutils.core import setup

setup(name='PyPR',
      version='0.1',
      description='Python Pattern Recognition',
      author='Joan Petur Petersen',
      author_email='joanpeturpetersen@gmail.com',
      url='http://www.decision3.com/jpp',
      packages=['pypr', 'pypr/gp', 'pypr/optimization', 
                'pypr/ann', 'pypr/clustering', 'pypr/preprocessing',
                'pypr/stattest', 'pypr/helpers'],
     )

