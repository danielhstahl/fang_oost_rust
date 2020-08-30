//! Fang Oosterlee approach for inverting a characteristic function.
//! Some useful characteristic functions are provided in the
//! [cf_functions](https://crates.io/crates/cf_functions) repository.
//! [Link to Fang-Oosterlee paper](http://ta.twi.tudelft.nl/mf/users/oosterle/oosterlee/COS.pdf).
//!

use num_complex::Complex;
use rayon::prelude::*;
use std::f64::consts::PI;
/**
    Function to compute the difference in successive X nodes.  This can feed into the "getX" function.
    @xDiscrete number of sections to parse the X domain into
    @xMin the minimum of the X domain
    @xMax the maximum of the X domain
    @return the difference between successive x nodes
*/
fn compute_dx(x_discrete: usize, x_min: f64, x_max: f64) -> f64 {
    (x_max - x_min) / ((x_discrete as f64) - 1.0)
}

/// Function to compute the discrete U. The operation is cheap
/// and takes less ram than simply using the computeURange
/// function to create a vector.  Note that "uMin" is always
/// zero and hence is unnecessary.  This can (should?) be
/// simply an implementation of a generic "getNode" function
/// but is broken into two functions to make it explicit and be
/// more closely aligned with the Fang Oosterlee paper.
///
fn get_u(du: f64, index: usize) -> f64 {
    (index as f64) * du
}

/**
    Function to compute the discrete X.  The operation is cheap and takes less ram than simply using the computeXRange function to create a vector
    @xMin the minimum of the X domain
    @dx the difference between the nodes in X
    @index the location of the node
    @return location of discrete X
*/
fn get_x(x_min: f64, dx: f64, index: usize) -> f64 {
    x_min + (index as f64) * dx
}

/// Returns iterator over real (x) domain
/// # Examples
/// ```
/// let x_min = -20.0;
/// let x_max = 25.0;
/// let x_discrete = 10;
/// let x_range=fang_oost::get_x_domain(
///    x_discrete, x_min, x_max
/// );
/// ```
pub fn get_x_domain(
    x_discrete: usize,
    x_min: f64,
    x_max: f64,
) -> impl IndexedParallelIterator<Item = f64> {
    let dx = compute_dx(x_discrete, x_min, x_max);
    (0..x_discrete)
        .into_par_iter()
        .map(move |index| get_x(x_min, dx, index))
}

/// Function to compute the difference in successive U nodes.
/// This can feed into the "getU" function.  Note that this
/// depends on X: the U and X domains are not "independent".
fn compute_du(x_min: f64, x_max: f64) -> f64 {
    PI / (x_max - x_min)
}
/**
    Helper function to get "CP"
    @du Discrete step in u.  Can be computed using compute_du(x_min, x_max)
*/
fn compute_cp(du: f64) -> f64 {
    (2.0 * du) / PI
}

fn get_complex_u(u: f64) -> Complex<f64> {
    Complex::<f64>::new(0.0, u)
}

/// Helper function to get complex u domain
/// @num_u the number of discrete steps in complex space
/// @x_min the minimum value of the truncated x domain
/// @x_max the maximum value of the truncated x domain
/// @return discrete complex values spanning the complex space
/// # Examples
/// ```
/// let num_u = 10;
/// let x_min = -20.0;
/// let x_max = 20.0;
/// let u_domain=fang_oost::get_u_domain(
///     num_u,
///     x_min,
///     x_max
/// );
/// ```
pub fn get_u_domain(
    num_u: usize,
    x_min: f64,
    x_max: f64,
) -> impl IndexedParallelIterator<Item = Complex<f64>> {
    let du = compute_du(x_min, x_max);
    (0..num_u)
        .into_par_iter()
        .map(move |index| get_complex_u(get_u(du, index)))
}

/*Using X only makes sense for
Levy processes where log(S/K) changes
for every iteration.  This is done
separately from the Characteristic
Function for computation purposes.*/
fn convolute_extended<T>(cf_incr: &Complex<f64>, x: f64, u_im: f64, u_index: usize, vk: T) -> f64
where
    T: Fn(f64, f64, usize) -> f64,
{
    (cf_incr * (get_complex_u(u_im) * x).exp()).re * vk(u_im, x, u_index)
}
/*Convolution in standard Fourier space*/
fn convolute_real<T>(cf_incr: &Complex<f64>, x: f64, u_im: f64, u_index: usize, vk: T) -> f64
where
    T: Fn(f64, f64, usize) -> f64,
{
    cf_incr.re * vk(u_im, x, u_index)
}

fn adjust_index(index: usize) -> f64 {
    if index == 0 {
        0.5
    } else {
        1.0
    }
}
fn integrate_cf<S>(
    discrete_cf_adjusted: &[Complex<f64>],
    du: f64,
    x: f64,
    convolute: S, //this can be expensive for extended cf
) -> f64
where
    S: Fn(&Complex<f64>, f64, f64, usize) -> f64 + std::marker::Sync + std::marker::Send,
{
    discrete_cf_adjusted
        .iter()
        .enumerate()
        .map(|(index, &cf_incr)| {
            let adjusted_cf_incr = cf_incr * adjust_index(index);
            convolute(&adjusted_cf_incr, x, get_u(du, index), index)
        })
        .sum()
}

fn adjust_cf(
    fn_inv_increment: &Complex<f64>,
    u: Complex<f64>,
    x_min: f64,
    cp: f64,
) -> Complex<f64> {
    fn_inv_increment * (-u * x_min).exp() * cp
}

fn get_discrete_cf_adjusted(
    x_min: f64,
    x_max: f64,
    fn_inv_vec: &[Complex<f64>],
) -> Vec<Complex<f64>> {
    let du = compute_du(x_min, x_max);
    let cp = compute_cp(du);
    get_u_domain(fn_inv_vec.len(), x_min, x_max)
        .zip(fn_inv_vec)
        .map(move |(u, fn_inv_element)| adjust_cf(&fn_inv_element, u, x_min, cp))
        .collect()
}
/// Returns "raw" discrete cf
/// @num_u the number of discrete steps in complex space
/// @x_min the minimum value of the truncated x domain
/// @x_max the maximum value of the truncated x domain
/// @cf_fn characteristic function
/// # Examples
/// ```
/// extern crate num_complex;
/// use num_complex::Complex;
/// extern crate fang_oost;
/// # fn main(){
/// let num_u = 256;
/// let x_min = -20.0;
/// let x_max = 25.0;
/// let mu=2.0;
/// let sigma:f64=5.0;
/// let norm_cf = |u:&Complex<f64>|(u*mu+0.5*u*u*sigma*sigma).exp();
/// let discrete_cf=fang_oost::get_discrete_cf(num_u, x_min, x_max, &norm_cf);
/// # }
/// ```
pub fn get_discrete_cf<T>(num_u: usize, x_min: f64, x_max: f64, cf_fn: T) -> Vec<Complex<f64>>
where
    T: Fn(&Complex<f64>) -> Complex<f64> + std::marker::Sync + std::marker::Send,
{
    get_u_domain(num_u, x_min, x_max)
        .map(|u| cf_fn(&u))
        .collect::<Vec<_>>()
}

fn get_expectation_generic_single_element<S>(
    x_min: f64,
    x_max: f64,
    x: f64,
    fn_inv_vec: &[Complex<f64>],
    convolute: S,
) -> f64
where
    S: Fn(&Complex<f64>, f64, f64, usize) -> f64 + std::marker::Sync + std::marker::Send,
{
    let du = compute_du(x_min, x_max);
    integrate_cf(
        &get_discrete_cf_adjusted(x_min, x_max, fn_inv_vec),
        du,
        x,
        &convolute,
    )
}

/**All generic functions will be provided iterators over x
 * and iterators over RAW characteristic function.
 */
fn get_expectation_generic<'a, 'b: 'a, S, T>(
    x_min: f64,
    x_max: f64,
    x_domain_iterator: T,
    fn_inv_vec: &'b [Complex<f64>],
    convolute: S,
) -> impl IndexedParallelIterator<Item = f64> + 'a + std::marker::Sync + std::marker::Send + 'a
where
    S: Fn(&Complex<f64>, f64, f64, usize) -> f64 + std::marker::Sync + std::marker::Send + 'a,
    T: IndexedParallelIterator<Item = f64> + std::marker::Sync + std::marker::Send + 'a,
{
    let du = compute_du(x_min, x_max);
    let discrete_cf_adjusted = get_discrete_cf_adjusted(x_min, x_max, fn_inv_vec);
    x_domain_iterator.map(move |x| integrate_cf(&discrete_cf_adjusted, du, x, &convolute))
}

/// Returns expectation over equal mesh in the real domain
///
/// # Remarks
///
/// The "type" of the expectation is handled by the vk function
///
/// # Examples
/// ```
/// extern crate num_complex;
/// extern crate fang_oost;
/// use num_complex::Complex;
/// extern crate rayon;
/// use rayon::prelude::*;
/// # fn main() {
/// let mu = 2.0;
/// let sigma:f64 = 5.0;
/// let num_u = 256;
/// let num_x = 1024;
/// let x_min = -20.0;
/// let x_max = 25.0;
/// let x_domain=fang_oost::get_x_domain(num_x, x_min, x_max);
/// let norm_cf = |u:&Complex<f64>|(u*mu+0.5*u*u*sigma*sigma).exp();
/// let cf_discrete=fang_oost::get_discrete_cf(num_u, x_min, x_max, &norm_cf);
/// let result:Vec<f64>=fang_oost::get_expectation_real(
///     x_min,
///     x_max,
///     x_domain,
///     &cf_discrete,
///     |u_im, x, k|{
///         if k==0{x-x_min} else { ((x-x_min)*u_im).sin()/u_im}
///     }
/// ).collect();
/// # }
/// ```
pub fn get_expectation_real<'a, 'b: 'a, S, U>(
    x_min: f64,
    x_max: f64,
    x_domain_iterator: S,
    discrete_cf: &'b [Complex<f64>],
    vk: U,
) -> impl IndexedParallelIterator<Item = f64> + 'a
where
    S: IndexedParallelIterator<Item = f64> + std::marker::Sync + 'b,
    U: Fn(f64, f64, usize) -> f64 + std::marker::Sync + std::marker::Send + 'b,
{
    get_expectation_generic(
        x_min,
        x_max,
        x_domain_iterator,
        discrete_cf,
        move |cf, x, u, i| convolute_real(cf, x, u, i, &vk),
    )
}
/// Returns expectation over equal mesh in the real domain
/// where characteristic function depends on initial starting point
/// of a Levy process.
///
/// # Remarks
///
/// The "type" of the expectation is handled by the vk function.
/// This function is useful for Levy functions since the characteristic function
/// depends on the initial value of x.  See [fang_oost_option](https://docs.rs/crate/fang_oost_option/0.21.3/source/src/option_pricing.rs)
/// for an example.
///
/// # Examples
/// ```
/// extern crate num_complex;
/// extern crate fang_oost;
/// use num_complex::Complex;
/// extern crate rayon;
/// use rayon::prelude::*;
/// # fn main() {
/// let mu = 2.0;
/// let sigma:f64 = 5.0;
/// let num_u = 256;
/// let num_x = 1024;
/// let x_min = -20.0;
/// let x_max = 25.0;
/// let norm_cf = |u:&Complex<f64>|(u*mu+0.5*u*u*sigma*sigma).exp();
/// let x_domain=fang_oost::get_x_domain(num_x, x_min, x_max);
/// let discrete_cf=fang_oost::get_discrete_cf(num_u, x_min, x_max, &norm_cf);
/// let result:Vec<f64>=fang_oost::get_expectation_extended(
///     x_min,
///     x_max,
///     x_domain,
///     &discrete_cf,
///     |u_im, x, k|{
///         if k==0{x-x_min} else { ((x-x_min)*u_im).sin()/u_im }
///     }
/// ).collect();
/// # }
/// ```
pub fn get_expectation_extended<'a, 'b: 'a, S, U>(
    x_min: f64,
    x_max: f64,
    x_domain_iterator: S,
    discrete_cf: &'b [Complex<f64>],
    vk: U,
) -> impl IndexedParallelIterator<Item = f64> + 'a
where
    S: IndexedParallelIterator<Item = f64> + std::marker::Sync + 'b,
    U: Fn(f64, f64, usize) -> f64 + std::marker::Sync + std::marker::Send + 'b,
{
    get_expectation_generic(
        x_min,
        x_max,
        x_domain_iterator,
        discrete_cf,
        move |cf, x, u, i| convolute_extended(cf, x, u, i, &vk),
    )
}

/// Returns expectation at point supplied by the user
///
/// # Remarks
/// The endpoints of the vector should have a large enough
/// domain for accuracy.
/// The "type" of the expectation is handled by the vk function.
/// # Examples
/// ```
/// extern crate num_complex;
/// extern crate fang_oost;
/// use num_complex::Complex;
/// # fn main() {
/// let x_min = -20.0;
/// let x_max = 25.0;
/// let x = 3.0;
/// let mu=2.0;
/// let sigma:f64=5.0;
/// let num_u=128;
/// let norm_cf = |u:&Complex<f64>|(u*mu+0.5*u*u*sigma*sigma).exp();
/// let discrete_cf=fang_oost::get_discrete_cf(num_u, x_min, x_max, &norm_cf);
/// let result=fang_oost::get_expectation_single_element_real(
///    x_min, x_max, x, &discrete_cf,
///     |u_im, x, k|{
///         if k==0{x-x_min} else { ((x-x_min)*u_im).sin()/u_im }
///     }
/// );
/// # }
/// ```
pub fn get_expectation_single_element_real<'a, U>(
    x_min: f64,
    x_max: f64,
    x: f64,
    fn_inv_discrete: &[Complex<f64>],
    vk: U,
) -> f64
where
    U: Fn(f64, f64, usize) -> f64 + std::marker::Sync + std::marker::Send + 'a,
{
    get_expectation_generic_single_element(x_min, x_max, x, fn_inv_discrete, move |cf, x, u, i| {
        convolute_real(cf, x, u, i, &vk)
    })
}

/// Returns expectation at point supplied by the user
/// where characteristic function depends on initial starting point.
/// # Remarks
/// The endpoints of the vector should have a large enough
/// domain for accuracy.
/// The "type" of the expectation is handled by the vk function.
/// This function is useful for Levy functions since the characteristic function
/// depends on the initial value of x.  See [fang_oost_option](https://docs.rs/crate/fang_oost_option/0.21.3/source/src/option_pricing.rs)
/// for an example.
/// # Examples
/// ```
/// extern crate num_complex;
/// extern crate fang_oost;
/// use num_complex::Complex;
/// # fn main() {
/// let x_min = -20.0;
/// let x_max = 25.0;
/// let x = 3.0;
/// let mu=2.0;
/// let sigma:f64=5.0;
/// let num_u=128;
/// let norm_cf = |u:&Complex<f64>|(u*mu+0.5*u*u*sigma*sigma).exp();
/// let discrete_cf=fang_oost::get_discrete_cf(num_u, x_min, x_max, &norm_cf);
/// let result=fang_oost::get_expectation_single_element_extended(
///     x_min, x_max, x, &discrete_cf,
///     |u_im, x, k|{
///         if k==0{x-x_min} else { ((x-x_min)*u_im).sin()/u_im }
///     }
/// );
/// # }
/// ```
pub fn get_expectation_single_element_extended<'a, 'b: 'a, U>(
    x_min: f64,
    x_max: f64,
    x: f64,
    fn_inv_discrete: &'b [Complex<f64>],
    vk: U,
) -> f64
where
    U: Fn(f64, f64, usize) -> f64 + std::marker::Sync + std::marker::Send + 'a,
{
    get_expectation_generic_single_element(x_min, x_max, x, fn_inv_discrete, |cf, x, u, i| {
        convolute_extended(cf, x, u, i, &vk)
    })
}
/// Returns iterator over density with domain created by the function
///
/// # Examples
/// ```
/// extern crate num_complex;
/// extern crate fang_oost;
/// extern crate rayon;
/// use rayon::prelude::*;
/// use num_complex::Complex;
///
/// # fn main() {
/// let num_x = 1024;
/// let num_u = 256;
/// let x_min = -20.0;
/// let x_max = 25.0;
/// let mu=2.0;
/// let sigma:f64=5.0;
/// let norm_cf = |u:&Complex<f64>|(u*mu+0.5*u*u*sigma*sigma).exp();
/// let x_domain=fang_oost::get_x_domain(num_x, x_min, x_max);
/// let discrete_cf=fang_oost::get_discrete_cf(num_u, x_min, x_max, &norm_cf);
/// let density:Vec<f64> = fang_oost::get_density(
///    x_min, x_max, x_domain, &discrete_cf
/// ).collect();
/// # }
/// ```
pub fn get_density<'a, 'b: 'a, S>(
    x_min: f64,
    x_max: f64,
    x_domain_iterator: S,
    fn_inv_vec: &'b [Complex<f64>],
) -> impl IndexedParallelIterator<Item = f64> + 'a
where
    S: IndexedParallelIterator<Item = f64> + std::marker::Sync + 'b,
{
    get_expectation_real(
        x_min,
        x_max,
        x_domain_iterator,
        fn_inv_vec,
        move |u, x, _| (u * (x - x_min)).cos(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;
    fn vk_cdf(u: f64, x: f64, a: f64, k: usize) -> f64 {
        if k == 0 {
            x - a
        } else {
            ((x - a) * u).sin() / u
        }
    }
    #[test]
    fn test_get_x_domain() {
        assert_eq!(
            get_x_domain(5, 0.0, 1.0).collect::<Vec<f64>>(),
            vec![0.0, 0.25, 0.5, 0.75, 1.0]
        )
    }

    #[test]
    fn test_compute_inv() {
        let mu = 2.0;
        let sigma = 1.0;
        let num_x = 5;
        let num_u = 256;
        let x_min = -3.0;
        let x_max = 7.0;
        let norm_cf = |u: &Complex<f64>| (u * mu + 0.5 * u * u * sigma * sigma).exp();
        let my_x_domain = get_x_domain(num_x, x_min, x_max);
        let ref_normal: Vec<f64> = get_x_domain(num_x, x_min, x_max)
            .map(|x| {
                (-(x - mu).powi(2) / (2.0 * sigma * sigma)).exp() / (sigma * (2.0 * PI).sqrt())
            })
            .collect();
        let discrete_cf = get_discrete_cf(num_u, x_min, x_max, &norm_cf);
        let my_inverse: Vec<f64> = get_density(x_min, x_max, my_x_domain, &discrete_cf).collect();

        for (reference, estimate) in ref_normal.iter().zip(my_inverse) {
            assert_abs_diff_eq!(*reference, estimate, epsilon = 0.001);
        }
    }

    #[test]
    fn test_cdf() {
        let mu = 2.0;
        let sigma: f64 = 5.0;

        let num_x = 55;
        let num_u = 256;
        let x_min = -20.0;
        let x_max = 25.0;
        let norm_cf = |u: &Complex<f64>| (u * mu + 0.5 * u * u * sigma * sigma).exp();
        let x_domain = get_x_domain(num_x, x_min, x_max);
        let ref_normal: Vec<f64> = get_x_domain(num_x, x_min, x_max)
            .map(|x| 0.5 * statrs::function::erf::erfc(-((x - mu) / sigma) / (2.0 as f64).sqrt()))
            .collect();

        let discrete_cf = get_discrete_cf(num_u, x_min, x_max, &norm_cf);
        let result: Vec<f64> =
            get_expectation_real(x_min, x_max, x_domain, &discrete_cf, |u, x, k| {
                vk_cdf(u, x, x_min, k)
            })
            .collect();
        for (reference, estimate) in ref_normal.iter().zip(result) {
            assert_abs_diff_eq!(*reference, estimate, epsilon = 0.001);
        }
    }

    #[test]
    fn test_expectation() {
        let mu = 2.0;
        let sigma: f64 = 5.0;
        let num_u = 256;
        let x_min = -20.0;
        let x_max = 25.0;
        let norm_cf = |u: &Complex<f64>| (u * mu + 0.5 * u * u * sigma * sigma).exp();
        let discrete_cf = get_discrete_cf(num_u, x_min, x_max, &norm_cf);
        let result: f64 =
            get_expectation_single_element_real(x_min, x_max, 2.0, &discrete_cf, |u, x, k| {
                vk_cdf(u, x, x_min, k)
            });
        assert_abs_diff_eq!(0.5, result, epsilon = 0.001);
    }
}
