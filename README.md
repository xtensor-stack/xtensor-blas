# ![xtensor](http://quantstack.net/assets/images/xtensor-blas.svg)

[![Travis](https://travis-ci.org/QuantStack/xtensor-blas.svg?branch=master)](https://travis-ci.org/QuantStack/xtensor-blas)
[![Appveyor](https://ci.appveyor.com/api/projects/status/quf1hllkedr0rxbk?svg=true)](https://ci.appveyor.com/project/QuantStack/xtensor-blas)
[![Documentation Status](http://readthedocs.org/projects/xtensor-blas/badge/?version=latest)](https://xtensor-blas.readthedocs.io/en/latest/?badge=latest)
[![Join the Gitter Chat](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/QuantStack/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

## Introduction

----

**DISCLAIMER** Please consider this software as BETA. We have not yet performed extensive testing and therefore there might be some bugs, or numerical issues, so don't use it to control your space rockets yet! Let us know if you find any issues.

----

`xtensor-blas` is an extension to the xtensor library, offering bindings to BLAS and LAPACK libraries 
through cxxblas and cxxlapack from the [FLENS](https://github.com/michael-lehn/FLENS) project.

`xtensor-blas` currently provides non-broadcasting `dot`, `norm` (1- and 2-norm for vectors), `inverse`, `solve`,
`eig`, `cross`, `det`, `slogdet`, `matrix_rank`, `inv`, `cholesky`, `qr`, `svd` in the `xt::linalg` namespace (check the corresponding `xlinalg.hpp` header for the function signatures). The functions, and signatures, are trying to be 1-to-1 equivalent to NumPy.
Low-level functions to interface with BLAS or LAPACK with xtensor containers are also offered 
in the `blas` and `lapack` namespace.

`xtensor` and `xtensor-blas` require a modern C++ compiler supporting C++14. The following C++ compilers are supported:

 - On Windows platforms, Visual C++ 2015 Update 2, or more recent
 - On Unix platforms, gcc 4.9 or a recent version of Clang

## Installation

`xtensor-blas` is a header-only library that depends on `xtensor`, `LAPACK` and `BLAS`.
You can directly install it from the sources:

```bash
git clone https://github.com/QuantStack/xtensor-blas
cd xtensor-blas
mkdir build
cd build
cmake .. -D CMAKE_INSTALL_PREFIX=your_install_prefix  # for tests: -DBUILD_TESTS=ON
make install
```

## Dependency on `xtensor`

`xtensor-blas` depends on the `xtensor` package

| `xtensor-blas`  | `xtensor` |
|-----------------|-----------|
| master          |  ^0.10.2  |
| 0.1.0           |  ^0.10.2  |

## License

We use a shared copyright model that enables all contributors to maintain the
copyright on their contributions.

This software is licensed under the BSD-3-Clause license. See the [LICENSE](LICENSE) file for details.
