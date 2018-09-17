| [Linux][lin-link] |  [Codecov][cov-link]  |
| :---------------: | :-------------------: |
| ![lin-badge]      | ![cov-badge]          |

[lin-badge]: https://travis-ci.org/phillyfan1138/fang_oost_rust.svg?branch=master "Travis build status"
[lin-link]:  https://travis-ci.org/phillyfan1138/fang_oost_rust "Travis build status"
[cov-badge]: https://codecov.io/gh/phillyfan1138/fang_oost_rust/branch/master/graph/badge.svg
[cov-link]:  https://codecov.io/gh/phillyfan1138/fang_oost_rust

# Fang-Oosterlee Library for Rust

Implements [Fang-Oosterlee](https://mpra.ub.uni-muenchen.de/8914/4/MPRA_paper_8914.pdf) algorithm in Rust.  While the algorithm originally was used for option pricing, it can be used for a variety of use cases.  For example, it can be used to compute the Value at Risk of a distribution, the density of a distribution, and the partial expectation.  

It requires a characteristic function computed at various specific intervals.  A utility function is provided which converts an analytical characteristic function into a vector. 

Documentation is at [docs.rs](https://docs.rs/fang_oost/0.13.7/fang_oost/)


## Use
Put the following in your Cargo.toml:

```toml
[dependencies]
fang_oost = "0.13"
```

Import and use:

```rust
extern crate num_complex;
extern crate fang_oost;
extern crate rayon;
use rayon::prelude::*;
use num_complex::Complex;
 
let num_x = 1024;
let num_u = 256;
let x_min = -20.0;
let x_max = 25.0;
let mu=2.0;
let sigma:f64=5.0;
let norm_cf = |u:&Complex<f64>|(u*mu+0.5*u*u*sigma*sigma).exp();
let x_domain=fang_oost::get_x_domain(num_x, x_min, x_max);
//computes discrete gaussian characteristic function
let discrete_cf=fang_oost::get_discrete_cf(num_u, x_min, x_max, &norm_cf);
let density:Vec<f64> = fang_oost::get_density(
   x_min, x_max, x_domain, &discrete_cf
).collect();
```

## Related Crates

* [Option Pricing with Fang Oost](https://crates.io/crates/fang_oost_option)
* [Distribution Utilities (including CDF, VaR, and Expected Shortfall)](https://crates.io/crates/cf_dist_utils)