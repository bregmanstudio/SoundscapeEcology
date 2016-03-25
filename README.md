<h1>Soundscape Analysis by Shift-Invariant Latent Components</h1>
<h2>Michael Casey - Bregman Labs, Dartmouth College</h2>

Use matrix factorization to decompose spectrograms into independent streams of sound objects. The method employs shift-invariant probabilistic latent component analysis (SIPLCA) for factorizing a time-frequency matrix (2D array) into a convolution of 2D kernels (patches) with sparse activation functions. 

Methods are based on the following:

1. Smaragdis, P, B. Raj, and M.V. Shashanka, 2008. [Sparse and shift-invariant feature extraction from non-negative data](http://paris.cs.illinois.edu/pubs/smaragdis-icassp2008.pdf). In proceedings IEEE International Conference on Audio and Speech Signal Processing, Las Vegas, Nevada, USA.
    
2. Smaragdis, P. and Raj, B. 2007. [Shift-Invariant Probabilistic Latent Component Analysis, tech report](http://paris.cs.illinois.edu/pubs/plca-report.pdf), MERL technical report, Camrbidge, MA.

<h2>Requirements</h2>
To run this workbook you will need: Numpy, Matlplotlib, the [Bregman Audio Toolkit](http://github.com/bregmanstudio/BregmanToolkit), and the [Bregman SoundscapeEcology Toolkit](http://github.com/bregmanstudio/SoundscapeEcology).

    