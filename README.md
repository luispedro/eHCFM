# e-HCFM Processing

This is the source code to the computational pipeline for environmental high
content fluorescence microscopy data processing.

# Functionality

This code implements the computational pipeline for e-HFCM, in particular:

- Bead statistics generation
- Segmentation (2D & 3D)
- Feature computation (480 features)
- Hierarchical classification
- Vignette and hyperstack output generation (several outputs)

# Requirements

The code uses Python, in particular, the [numpy](https://http://www.numpy.org/)
stack, [mahotas](https://mahotas.rtfd.io/) for image processing,
[imread](https://http://imread.rtfd.io/) for image loading, and
[jug](https://jug.readthedocs.io/) for parallel processing and pipeline
management.

The computations in the manuscript used Python 2.7, but the code is compatible
with Python 3.

# Copyright & License

- Copyright: Bork Group, EMBL 2013-2017
- Licence: GPL version 2 (or later)

