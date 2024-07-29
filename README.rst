.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/stresspred.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/stresspred
    .. image:: https://readthedocs.org/projects/stresspred/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://stresspred.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/pypi/v/stresspred.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/stresspred/
    .. image:: https://img.shields.io/conda/vn/conda-forge/stresspred.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/stresspred
    .. image:: https://pepy.tech/badge/stresspred/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/stresspred
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/stresspred

.. image:: https://img.shields.io/coveralls/github/danibene/stresspred/main.svg
    :alt: Coveralls
    :target: https://coveralls.io/r/danibene/stresspred
.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

========
stresspred
========


    Template-based in-ear heartbeat extraction


Usage
==========
.. code-block:: python

    from stresspred.extraction.heartbeat_extraction import hb_extract
    # sig is a 1D numpy array
    # peak_time is a 1D numpy array with the time of each heartbeat in seconds
    peak_time = hb_extract(sig, sampling_rate=sampling_rate, method="temp")

To use a method implemented in MATLAB, you need to have MATLAB installed and
`the MATLAB engine for Python`_. After putting the MATLAB code in the
``src/matlab`` folder, you can use it as follows:

.. _the MATLAB engine for Python: https://www.mathworks.com/help/matlab/matlab-engine-for-python.html


.. code-block:: python

    peak_time = hb_extract(sig, sampling_rate=sampling_rate, method="matlab")

.. _pyscaffold-notes:

Making Changes & Contributing
=============================

This project uses `pre-commit`_, please make sure to install it before making any
changes::

    pip install pre-commit
    cd stresspred
    pre-commit install

It is a good idea to update the hooks to the latest version::

    pre-commit autoupdate

Don't forget to tell your contributors to also install and use pre-commit.

.. _pre-commit: https://pre-commit.com/

Note
====

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.
