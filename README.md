# FeuPy

[![Gammapy](https://img.shields.io/badge/powered%20by-Gammapy-orange.svg?style=flat)](https://gammapy.org/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

**FeuPy** is an open-source Python package for the analysis, modeling, and simulation of very-high-energy (VHE) gamma-ray data.

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

### 1. Clone FeuPy

Clone the repository:

```bash
git clone https://github.com/rubensjrcosta/feupy.git
cd feupy
```

### 2. Create the Conda environment

FeuPy provides an `environment.yml` file with the recommended development environment. The current release uses Python 3.12 and Gammapy 2.0.

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate feupy
```

The FeuPy package is installed in editable mode automatically when the environment is created.

### 3. Configure the Gammapy datasets

FeuPy uses the standard datasets distributed with Gammapy 2.0.

If you already have the Gammapy datasets installed and `GAMMAPY_DATA` configured, no additional setup is required.

Otherwise, download the official datasets:

```bash
gammapy download datasets --release 2.0
```

and set the `GAMMAPY_DATA` environment variable:

```bash
export GAMMAPY_DATA=/path/to/gammapy-datasets/2.0
```

### 4. Download the FeuPy datasets

Some FeuPy functionalities require additional datasets maintained in the [FeuPy data repository](https://github.com/rubensjrcosta/feupy-data):

```bash
git clone https://github.com/rubensjrcosta/feupy-data.git
```

FeuPy `v0.1.0` uses **FeuPy datasets version 1.0**, located at:

```text
feupy-data/
└── feupy-datasets/
    └── 1.0/
```

Set the `FEUPY_DATA` environment variable:

```bash
export FEUPY_DATA=/path/to/feupy-data/feupy-datasets/1.0
```

### 5. Conda environment variables

If you use the FeuPy Conda environment, the data paths can instead be stored directly in the environment:

```bash
conda env config vars set GAMMAPY_DATA=/path/to/gammapy-datasets/2.0
conda env config vars set FEUPY_DATA=/path/to/feupy-data/feupy-datasets/1.0
```

Reactivate the environment after setting the variables:

```bash
conda deactivate
conda activate feupy
```

Verify the configuration:

```bash
echo $GAMMAPY_DATA
echo $FEUPY_DATA
```

## Quick start

After installation and data configuration, verify the FeuPy installation:

```python
import feupy

print(feupy.__version__)
```

For example, a source catalog can be accessed with:

```python
from feupy.catalogs import SourceCatalogLHAASO

catalog = SourceCatalogLHAASO()
source = catalog[0]

print(source)
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

* **Rubens Costa Jr.**
* **Rita C. dos Anjos**

## License

FeuPy is distributed under the BSD 3-Clause License. See [`LICENSE`](LICENSE) for details.
