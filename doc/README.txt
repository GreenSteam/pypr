
To compile the documents you need sphinx and numpydoc, and like many more
applications :)

It might be possible on your system to install sphinx and numpydoc using:

sudo easy_install -U sphinx
sudo easy_install -U numpydoc

For correctly displaying equations latex is also required, on Ubuntu you 
could install it using:

sudo apt-get install texlive-latex-base
sudo apt-get install texlive-latex-extra
sudo apt-get install dvipng


I get a lot of warnings when making the documentation, but I havn't found any cure yet. The warnings look like this:
/home/coffeemug/pypr/doc/source/gp.rst:: WARNING: toctree contains reference to nonexisting document 'generate'
/home/coffeemug/pypr/doc/source/gp.rst:: WARNING: toctree contains reference to nonexisting document 'regression'
/home/coffeemug/pypr/doc/source/index.rst:: WARNING: toctree contains reference to nonexisting document 'find_flat_weight_no'
/home/coffeemug/pypr/doc/source/index.rst:: WARNING: toctree contains reference to nonexisting document 'find_weight'



