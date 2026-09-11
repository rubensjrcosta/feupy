# FeuPy

[![Gammapy](https://img.shields.io/badge/powered%20by-Gammapy-orange.svg?style=flat)](https://gammapy.org/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

**FeuPy** is an open-source Python package for the analysis, modeling, and simulation of very-high-energy (VHE) gamma-ray observations.

Built around [Gammapy](https://gammapy.org/), FeuPy provides tools for multi-instrument spectral analysis, gamma-ray counterpart searches, observation simulations, sensitivity studies, and non-thermal emission modeling. The package is particularly designed to support studies involving current gamma-ray observatories and the Cherenkov Telescope Array Observatory (CTAO).

## Features

### Multi-instrument spectral analysis

FeuPy provides utilities for combining and analyzing gamma-ray measurements from different instruments and energy ranges using the Gammapy framework. This enables consistent spectral modeling of sources observed from GeV to multi-TeV energies.

### Catalog integration and counterpart searches

Tools are provided for identifying potential counterparts and retrieving information from astronomical catalogs, including:

* [ATNF Pulsar Catalogue (PSRCAT)](https://www.atnf.csiro.au/research/pulsar/psrcat/)
* [VERITAS VTSCat](https://github.com/VERITAS-Observatory/VERITAS-VTSCat)
* gamma-ray catalogs supported by Gammapy
* measurements collected from dedicated publications

These utilities facilitate the construction of multi-instrument datasets for gamma-ray source studies.

### CTAO performance and sensitivity studies

FeuPy provides tools for working with CTAO instrument response functions (IRFs) and evaluating the expected performance of different array configurations and observing conditions.

Applications include:

* management and validation of CTAO IRF configurations;
* spectral sensitivity calculations;
* comparison of observing configurations;
* prospective studies of CTAO observations.

### Observation simulations

FeuPy supports the simulation of gamma-ray observations using instrument response functions and predefined spectral models.

These simulations can be used to investigate source detectability, spectral reconstruction, and the expected performance of future observations.

### Non-thermal emission modeling

FeuPy includes utilities for modeling radiation produced by relativistic particle populations, with support for [Naima](https://naima.readthedocs.io/).

This allows the interpretation of gamma-ray observations in terms of leptonic and hadronic particle populations and their associated radiative processes.

## Installation

### 1. Install FeuPy

Clone the repository:

```bash
git clone https://github.com/rubensjrcosta/feupy.git
cd feupy
```

Install FeuPy and its dependencies:

```bash
pip install -e .
```

### 2. Download the data repositories

FeuPy relies on external datasets that are maintained separately from the source code. Clone the FeuPy and Gammapy data repositories:

```bash
git clone https://github.com/rubensjrcosta/feupy-data.git
git clone https://github.com/rubensjrcosta/gammapy-data.git
```

The datasets required by the current FeuPy release are located at:

```text
feupy-data/
└── feupy-datasets/
    └── 1.0/

gammapy-data/
└── gammapy-datasets/
    └── 2.0/
```

FeuPy `v0.1.0` uses **FeuPy datasets version 1.0** and **Gammapy datasets version 2.0**.

### 3. Configure the environment variables

Set `FEUPY_DATA` and `GAMMAPY_DATA` to the corresponding dataset directories.

For example:

```bash
export FEUPY_DATA=/path/to/feupy-data/feupy-datasets/1.0
export GAMMAPY_DATA=/path/to/gammapy-data/gammapy-datasets/2.0
```

You can verify the configuration with:

```bash
echo $FEUPY_DATA
echo $GAMMAPY_DATA
```

If you use a Conda environment, the variables can be stored directly in the environment:

```bash
conda env config vars set FEUPY_DATA=/path/to/feupy-data/feupy-datasets/1.0
conda env config vars set GAMMAPY_DATA=/path/to/gammapy-data/gammapy-datasets/2.0
```

Reactivate the environment after setting the variables:

```bash
conda deactivate
conda activate <your-environment>
```

## Quick start

After installation and data configuration, FeuPy can be imported in Python:

```python
import feupy
```

The configured data directories can be checked from the shell with:

```bash
echo $FEUPY_DATA
echo $GAMMAPY_DATA
```

Examples demonstrating the main analysis workflows are available in the [`examples`](examples/) directory.

## Scientific scope

FeuPy was developed to support reproducible studies of very-high-energy gamma-ray sources using measurements from current gamma-ray observatories and simulations of future CTAO observations.

The package is intended to provide a common framework for tasks such as multi-instrument spectral analysis, CTAO performance studies, observation simulations, and the physical interpretation of gamma-ray emission.

## Citation

If you use FeuPy in scientific work, please cite the software.

Citation information is provided in the [`CITATION.cff`](CITATION.cff) file.

A DOI for FeuPy will be provided through Zenodo starting with the `v0.1.0` release.

## Authors

**Rubens Costa Jr.**
**Rita C. dos Anjos**

## License

FeuPy is distributed under the BSD 3-Clause License. See [`LICENSE`](LICENSE) for details.

