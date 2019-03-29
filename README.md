# ![xtensor](docs/source/xtensor-blas.svg)

[![Travis](https://travis-ci.org/QuantStack/xtensor-blas.svg?branch=master)](https://travis-ci.org/QuantStack/xtensor-blas)
[![Appveyor](https://ci.appveyor.com/api/projects/status/quf1hllkedr0rxbk?svg=true)](https://ci.appveyor.com/project/QuantStack/xtensor-blas)
[![Documentation](http://readthedocs.org/projects/xtensor-blas/badge/?version=latest)](https://xtensor-blas.readthedocs.io/en/latest/?badge=latest)
[![Binder](https://img.shields.io/badge/launch-binder-brightgreen.svg)](https://mybinder.org/v2/gh/QuantStack/xtensor/stable?filepath=notebooks/xtensor.ipynb)
[![Join the Gitter Chat](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/QuantStack/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

## Introduction

`xtensor-blas` is an extension to the xtensor library, offering bindings to BLAS and LAPACK libraries through cxxblas and cxxlapack from the [FLENS](https://github.com/michael-lehn/FLENS) project.

`xtensor-blas` currently provides non-broadcasting `dot`, `norm` (1- and 2-norm for vectors), `inverse`, `solve`,
`eig`, `cross`, `det`, `slogdet`, `matrix_rank`, `inv`, `cholesky`, `qr`, `svd` in the `xt::linalg` namespace (check the corresponding `xlinalg.hpp` header for the function signatures). The functions, and signatures, are trying to be 1-to-1 equivalent to NumPy.
Low-level functions to interface with BLAS or LAPACK with xtensor containers are also offered in the `blas` and `lapack` namespace.

`xtensor` and `xtensor-blas` require a modern C++ compiler supporting C++14. The following C++ compilers are supported:

 - On Windows platforms, Visual C++ 2015 Update 2, or more recent
 - On Unix platforms, gcc 4.9 or a recent version of Clang

## Installation

xtensor-blas is a header-only library. We provide a package for the conda package manager.

```
conda install -c conda-forge xtensor-blas
```

which will also install the core `xtensor` package.

Or you can directly install it from the sources:

```
cmake -D CMAKE_INSTALL_PREFIX=your_install_prefix
make install
```

To build the tests or actually use `xtensor-blas`, you will need binaries for

 - `openblas`
 - `lapack`

which are also available on conda-forge.

## Trying it online

To try out xtensor-blas interactively in your web browser, just click on the binder
link:

[![Binder](binder-logo.svg)](https://mybinder.org/v2/gh/QuantStack/xtensor/stable?filepath=notebooks/xtensor.ipynb)

## Documentation

To get started with using `xtensor-blas`, check out the full documentation

http://xtensor-blas.readthedocs.io/

## Dependency on `xtensor`

`xtensor-blas` depends on the `xtensor` package

| `xtensor-blas`  | `xtensor` |
|-----------------|-----------|
| master          |  ^0.20.0  |
| 0.16.0          |  ^0.20.0  |
| 0.15.2          |  ^0.19.0  |
| 0.15.1          |  ^0.19.0  |
| 0.15.0          |  ^0.19.0  |
| 0.14.0          |  ^0.18.0  |
| 0.13.x          |  ^0.17.0  |
| 0.12.x          |  ^0.17.0  |
| 0.11.x          |  ^0.16.0  |
| 0.10.x          |  ^0.15.4  |
| 0.9.x           |  ^0.15.4  |
| 0.8.x           |  ^0.15.0  |
| 0.7.x           |  ^0.14.0  |

## License

We use a shared copyright model that enables all contributors to maintain the
copyright on their contributions.

This software is licensed under the BSD-3-Clause license. See the [LICENSE](LICENSE) file for details.
