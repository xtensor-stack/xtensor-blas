.. Copyright (c) 2017, Wolf Vollprecht, Johan Mabille and Sylvain Corlay

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.


.. raw:: html

   <style>
   .rst-content .section>img {
       width: 30px;
       margin-bottom: 0;
       margin-top: 0;
       margin-right: 15px;
       margin-left: 15px;
       float: left;
   }
   </style>

Usage
=====

To use xtensor-blas functions, the `xlinalg.hpp` header has to be included.
In the `xt::linalg` namespace, many of NumPy's `np.linalg` functions are implemented. 
We make an effort to keep the interfaces very similar.

For example, calculating a determinant:

.. highlight cpp

.. code-block::
    #include "xtensor-blas/xlinalg.hpp"
    
    int main()
    {
        xt::xarray<double> a = {{1,2,3}, {4,5,6}, {7,8,9}};
        auto d = xt::linalg::det(a);
        std::cout << d << std::endl;  // 6.661338e-16
    }

We can also try to compute the same determinant using the ``slogdet`` function, which
is more robust against under- or overflows by summing up the logarithm. The slogdet
function in NumPy returns a tuple of (sign, val). In C++, we emulate the behaviour by
returning a ``std::tuple``, which can be unpacked using ``std::get<N>(tuple)``.

.. code-block::
    xt::xarray<double> a = {{1,2,3}, {4,5,6}, {7,8,9}};
    auto d = xt::linalg::slogdet(a);
    std::cout << std::get<0>(d) << ", " << std::get<1>(d) << std::endl;  // 1, -34.9450...

Returning tuples is used throughout the xlinalg package.
