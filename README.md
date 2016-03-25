<h1>Soundscape Analysis by Shift-Invariant Latent Components</h1>
<h2>Michael Casey - Bregman Labs, Dartmouth College</h2>

A toolkit for matrix factorization of soundscape spectrograms into independent streams of sound objects, possibly representing individual species or independent group behaviours. 

The method employs shift-invariant probabilistic latent component analysis (SIPLCA) for factorizing a time-frequency matrix (2D array) into a convolution of 2D kernels (patches) with sparse activation functions. 

Methods are based on the following:

1. Smaragdis, P, B. Raj, and M.V. Shashanka, 2008. [Sparse and shift-invariant feature extraction from non-negative data](http://paris.cs.illinois.edu/pubs/smaragdis-icassp2008.pdf). In proceedings IEEE International Conference on Audio and Speech Signal Processing, Las Vegas, Nevada, USA.
    
2. Smaragdis, P. and Raj, B. 2007. [Shift-Invariant Probabilistic Latent Component Analysis, tech report](http://paris.cs.illinois.edu/pubs/plca-report.pdf), MERL technical report, Camrbidge, MA.

3. A. C. Eldridge, M. Casey, P. Moscoso, and M. Peck (2015) [A New Method for Ecoacoustics? Toward the Extraction and Evaluation of Ecologically-Meaningful Sound Objects using Sparse Coding Methods](https://peerj.com/preprints/1407.pdf). PeerJ PrePrints, 3(e1855) 1407v2 [In Review]

<h2>Requirements</h2>
1. Numpy / Matlplotlib (the Anaconda Python distribution is recommended for these)
2. The [Bregman Audio Toolkit](http://github.com/bregmanstudio/BregmanToolkit)
3. Recommended: [scikits.audiolab v0.11+](https://pypi.python.org/pypi/scikits.audiolab/)

<h2>Installation</h2>
	git clone https://github.com/bregmanstudio/BregmanToolkit.git
	cd BregmanToolkit
	sudo python setup.py install
	cd ..
	git clone https://github.com/bregmanstudio/SoundscapeEcology
	cd SoundscapeEcology

Either run python (or jupyter notebook) in the installed SoundscapeEcology directory, or add the directory's path to your $PYTHONPATH environment variable.

To test that everything is installed correctly, launch a python shell or notebook and type:
	from bregman.suite import *
	from soundscapeecology import *

<h2>License</h2>
MIT License. See LICENSE.txt.

Please report any problems in the issue tracker.
