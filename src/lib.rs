extern crate num;
extern crate num_complex;
extern crate black_scholes;
//extern crate num_traits;
extern crate rayon;
#[macro_use]
#[cfg(test)]
extern crate approx;
use num_complex::Complex;
use num::Float;
use num::traits::{Zero, NumOps, Num};
use std::f64::consts::PI;
use rayon::prelude::*;

/**
    Function to compute the difference in successive X nodes.  This can feed into the "getX" function.
    @xDiscrete number of sections to parse the X domain into
    @xMin the minimum of the X domain
    @xMax the maximum of the X domain
    @return the difference between successive x nodes
*/
fn compute_dx(x_discrete:usize, x_min:f64, x_max:f64)->f64{
    (x_max-x_min)/((x_discrete as f64)-1.0)
}

/**
    Function to compute the discrete U.  The operation is cheap and takes less ram than simply using the computeURange function to create a vector.  Note that "uMin" is always zero and hence is unecessary.  This can (should?) be simply an implementation of a generic "getNode" function but is broken into two functions to make it explicit and be more closely aligned with the Fang Oosterlee paper.
    @du the difference between the nodes in U
    @index the location of the node
    @return location of discrete U
*/
fn get_u(dx:f64, index:usize)->f64{
    (index as f64)*dx
}
/**
    Function to compute the discrete X range per Fang Oosterlee (2007)
    @xDiscrete number of sections to parse the X domain into
    @xMin the minimum of the X domain
    @xMax the maximum of the X domain
    @return vector of discrete X values
*/
fn compute_x_range(x_discrete:usize, x_min:f64, x_max:f64)->Vec<f64>{
    let dx=compute_dx(x_discrete, x_min, x_max);
    (0..x_discrete).into_par_iter().map(|index| x_min+(index as f64)*dx).collect()
}

/**
    Function to compute the discrete X.  The operation is cheap and takes less ram than simply using the computeXRange function to create a vector
    @xMin the minimum of the X domain
    @dx the difference between the nodes in X
    @index the location of the node
    @return location of discrete X
*/
fn get_x(x_min:f64, dx: f64, index:usize)->f64{
    x_min+(index as f64)*dx
}

/**
    Function to compute the difference in successive U nodes.  This can feed into the "getU" function.  Note that this depends on X: the U and X domains are not "independent".
    @xMin the minimum of the X domain
    @xMax the maximum of the X domain
    @return the difference between successive U nodes
*/
fn compute_du(x_min:f64, x_max:f64)->f64{
    PI/(x_max-x_min)
}
/**
    Helper function to get "CP"
    @du Discrete step in u.  Can be computed using computeDU(xMin, xMax)
*/
fn compute_cp(du:f64)->f64{
    (2.0*du)/PI
}

/**
    Helps convert CF into appropriate vector for inverting
    @u Complex number.  Discretiziation of complex plane.  Can be computed by calling getU(du, index)
    @xMin Minimum of real plane
    @cp Size of integration in complex domain.  Can be computed by calling computeCP(du)
    @fnInv Characteristic function of the density.  May be computationally intense to compute (eg, if using the combined CF of millions of loans)
*/
fn format_cf_real<T>(u:&Complex<f64>, x_min:f64, cp:f64, fn_inv:T)->f64
    where T:Fn(&Complex<f64>)->Complex<f64>+std::marker::Sync+std::marker::Send
{
    (fn_inv(u)*(-u*x_min).exp()).re*cp
}

fn format_cf<T>(u:&Complex<f64>, x_min:f64, cp:f64, fn_inv:T)->Complex<f64>
    where T:Fn(&Complex<f64>)->Complex<f64>+std::marker::Sync+std::marker::Send
{
    (fn_inv(u)*(-u*x_min).exp())*cp
}

/**
    Helper function to get complex u
    @u The real valued complex component.  Can be computed using getU(du, index)
*/
fn get_complex_u(u:f64)->Complex<f64>{
    Complex::<f64>::new(0.0, u)
}

/*Using X only makes sense for 
Levy processes where log(S/K) changes 
for every iteration.  This is done 
separately from the Characteristic
Function for computation purposes.*/
fn convolute_levy<T>(cf_incr:&Complex<f64>, x:f64, u:f64, u_index:usize, vk:T)->f64
    where T:Fn(f64, f64, usize)->f64
{
    (cf_incr*(get_complex_u(u)*x).exp()).re*vk(u, x, u_index)
}
//standard convolution in fouirer space (ie, multiplication)  */
fn convolute<T>(cf_incr:f64, x:f64, u:f64, u_index:usize, vk:T)->f64
    where T:Fn(f64, f64, usize)->f64
{
    cf_incr*vk(u, x, u_index)
}

fn adjust_index_cmpl(element:&Complex<f64>, index:usize)->Complex<f64>
{
    if index==0{element*0.5} else {*element}
}
fn adjust_index_fl(element:&f64, index:usize)->f64{
    if index==0 {element*0.5} else{*element}
}

/**used when aggregating log cfs and then having to invert the results
    @xMin min of real plane
    @xMax max of real plane
    @logAndComplexCF vector of complex log values of a CF.  
    @returns actual CF for inversion
    Note that the exp(logAndComplex-u*xMin) is equivalent to 
    the computation done in formatCFReal but with vector instead
    of the function itself

*/
fn convert_log_cf_to_real_exp(x_min:f64, x_max:f64, cf_vec:&Vec<Complex<f64>>)->Vec<f64>{ 
    let du=compute_du(x_min, x_max);
    let cp=compute_cp(du);
    cf_vec.par_iter().enumerate().map(|(index, &x)|{
        (x-get_complex_u(get_u(du, index))*x_min).exp().re*cp
    }).collect()
}

/**return vector of complex elements of cf. 
This is ONLY needed where the CF depends on 
a changing "x": like for option pricing where 
x=log(S/K) and K iterates  */
fn compute_discrete_cf<T>(x_min:f64, x_max:f64, u_discrete:usize, cf:T)->Vec<Complex<f64>>
    where T: Fn(&Complex<f64>)->Complex<f64>+std::marker::Sync+std::marker::Send
{
    let du=compute_du(x_min, x_max);
    let cp=compute_cp(du);
    (0..u_discrete).into_par_iter().map(|index|{
        let u=get_complex_u(get_u(du, index));
        format_cf(&u, x_min, cp, cf)
    }).collect()
}
/**return vector of real elements of cf. 
This will work for nearly every type 
of inversion EXCEPT where the CF depends on 
a changing "x": like for option pricing where 
x=log(S/K) and K iterates  */
fn compute_discrete_cf_real<T>(x_min:f64, x_max:f64, u_discrete:usize, cf:T)->Vec<f64> 
    where T: Fn(&Complex<f64>)->Complex<f64>+std::marker::Sync+std::marker::Send
{
    let du=compute_du(x_min, x_max);
    let cp=compute_cp(du);
    (0..u_discrete).into_par_iter().map(|index|{
        let u=get_complex_u(get_u(du, index));
        format_cf_real(&u, x_min, cp, cf)
    }).collect()
}




/**
    Computes the convolution given the discretized characteristic function.  
    @xDiscrete Number of discrete points in density domain
    @xmin Minimum number in the density domain
    @xmax Maximum number in the density domain
    @discreteCF Discretized characteristic function.  This is vector of complex numbers.
    @vK Function (parameters u and x, and index)  
    @returns approximate convolution
*/
fn compute_convolution_levy<T>(x_discrete:usize, x_min:f64, x_max:f64, discrete_cf:&Vec<Complex<f64>>, vk:T)->Vec<f64>
    where T:Fn(f64, f64, usize)->f64+std::marker::Sync+std::marker::Send
{ //vk as defined in fang oosterlee
    let dx=compute_dx(x_discrete, x_min, x_max);
    let du=compute_du(x_min, x_max);
    (0..x_discrete).into_par_iter().map(|x_index|{
        let x=get_x(x_min, dx, x_index);
        discrete_cf.iter().enumerate().fold(f64::zero(), |s, (index, &cf_incr)|{
            let cf_incr_m=adjust_index_cmpl(&cf_incr, index);
            s+convolute_levy(&cf_incr_m, x, get_u(du, index), index, vk)
        })
    }).collect()
}

fn compute_convolution_vec_levy_g<T, S>(x_values:&Vec<f64>, discrete_cf:&Vec<Complex<f64>>, vk:T, extra_fn:S)->Vec<f64>
    where T:Fn(f64, f64, usize)->f64+std::marker::Sync+std::marker::Send, 
    S:Fn(f64, f64, usize)->f64+std::marker::Sync+std::marker::Send
{ //vk as defined in fang oosterlee
    //let x_min:f64=x_values.first();
    let x_min:f64=x_values[0];
    //let x_max:f64=x_values.last();
    let x_max:f64=x_values[x_values.len()-1];
    let du=compute_du(x_min, x_max);
    x_values.par_iter().enumerate().map(|(x_index, &x_value)|{
        extra_fn(
            discrete_cf.iter().enumerate().fold(f64::zero(), |s, (index, &cf_incr)|{
                let cf_incr_m=adjust_index_cmpl(&cf_incr, index);
                s+convolute_levy(&cf_incr_m, x_value, get_u(du, index), index, vk)
            }),
            x_value,
            x_index
        )  
    }).collect()
}

fn compute_convolution_vec_levy<T>(x_values:&Vec<f64>, discrete_cf:&Vec<Complex<f64>>, vk:T)->Vec<f64>
    where T:Fn(f64, f64, usize)->f64+std::marker::Sync+std::marker::Send
{
    compute_convolution_vec_levy_g(x_values, discrete_cf, vk, |v, _, _|v)
}

fn compute_convolution_at_point_levy<T>(x_value:f64, x_min:f64, x_max:f64, discrete_cf:&Vec<Complex<f64>>, vk:T)->f64
    where T:Fn(f64, f64, usize)->f64
{
    let du=compute_du(x_min, x_max);
    discrete_cf.iter().enumerate().fold(f64::zero(), |s, (index, &cf_incr)|{
        let cf_incr_m=adjust_index_cmpl(&cf_incr, index);
        s+convolute_levy(&cf_incr_m, x_value, get_u(du, index), index, vk)
    })
}



/**
    Computes the convolution given the discretized characteristic function.  
    @xDiscrete Number of discrete points in density domain
    @xmin Minimum number in the density domain
    @xmax Maximum number in the density domain
    @discreteCF Discretized characteristic function.  This is vector of complex numbers.
    @vK Function (parameters u and x, and index)  
    @returns approximate convolution
*/

fn compute_convolution<T>(x_discrete:usize, x_min:f64, x_max:f64, discrete_cf:&Vec<f64>, vk:T)->Vec<f64>
    where T:Fn(f64, f64, usize)->f64+std::marker::Sync+std::marker::Send
{
    let dx=compute_dx(x_discrete, x_min, x_max);
    let du=compute_du(x_min, x_max);
    (0..x_discrete).into_par_iter().map(|x_index|{
        let x=get_x(x_min, dx, x_index);
        discrete_cf.iter().enumerate().fold(f64::zero(), |s, (index, &cf_incr)|{
            let cf_incr_m=adjust_index_fl(&cf_incr, index);
            s+convolute(cf_incr_m, x, get_u(du, index), index, vk)
        })
    }).collect()
}


fn compute_convolution_vec<T>(x_values:&Vec<f64>, discrete_cf:&Vec<f64>,  vk:T)->Vec<f64>
    where T:Fn(f64, f64, usize)->f64+std::marker::Sync+std::marker::Send
{
    let x_min=x_values[0];
    let x_max=x_values[x_values.len()-1];
    let du=compute_du(x_min, x_max);
    x_values.par_iter().map(|&x_value|{
        discrete_cf.iter().enumerate().fold(f64::zero(), |s, (index, &cf_incr)|{
            let cf_incr_m=adjust_index_fl(&cf_incr, index);
            s+convolute(cf_incr_m, x_value, get_u(du, index), index, vk)
        })
    }).collect()
}

fn compute_convolution_at_point<T>(
    x_value:f64, 
    x_min:f64,
    x_max:f64,
    discrete_cf:&Vec<f64>,  
    vk:T)->f64
    where T:Fn(f64, f64, usize)->f64+std::marker::Sync+std::marker::Send
{
    let du=compute_du(x_min, x_max);
    discrete_cf.iter().enumerate().fold(f64::zero(), |s, (index, &cf_incr)|{
        let cf_incr_m=adjust_index_fl(&cf_incr, index);
        s+convolute(cf_incr_m, x_value, get_u(du, index), index, vk)
    })
}



/********
    FROM HERE ON are the functions that should be used by external programs
**/

/**
    Computes the density given a discretized characteristic function discreteCF (of size uDiscrete) at the discrete points xRange in xmin, xmax. See Fang Oosterlee (2007) for more information.
    @xDiscrete Number of discrete points in density domain
    @xmin Minimum number in the density domain
    @xmax Maximum number in the density domain
    @discreteCF vector of characteristic function of the density at discrete U
    @returns approximate density
*/
pub fn compute_inv_discrete(
    x_discrete: usize, x_min:f64, 
    x_max:f64, discrete_cf:&Vec<f64>
)->Vec<f64>{
    compute_convolution(x_discrete, x_min, x_max, discrete_cf, |u, x, _|{
        (u*(x-x_min)).cos()
    })
}   
/**
    Computes the density given a characteristic function fnInv at the discrete points xRange in xmin, xmax. See Fang Oosterlee (2007) for more information.
    @xDiscrete Number of discrete points in density domain
    @uDiscrete Number of discrete points in the complex domain
    @xmin Minimum number in the density domain
    @xmax Maximum number in the density domain
    @fnInv Characteristic function of the density.  May be computationally intense to compute (eg, if using the combined CF of millions of loans)
    @returns approximate density
*/

pub fn compute_inv<T>(
    x_discrete:usize, u_discrete:usize, 
    x_min:f64, x_max:f64, fn_inv:T
)->Vec<f64>
    where T:Fn(&Complex<f64>)->Complex<f64>+std::marker::Sync+std::marker::Send
{
    let discrete_cf=compute_discrete_cf_real( //Vec<f64>
        x_min, x_max, u_discrete, 
        fn_inv 
    );
    compute_inv_discrete(
        x_discrete, x_min, x_max, 
        &discrete_cf
    )

}

/**
    Computes the density given a log characteristic function at the discrete points xRange in xmin, xmax.  See Fang Oosterlee (2007) for more information.
    @xDiscrete Number of discrete points in density domain
    @xmin Minimum number in the density domain
    @xmax Maximum number in the density domain
    @logFnInv vector of log characteristic function of the density at discrete U 
    @returns approximate density
*/
pub fn compute_inv_discrete_log(
    x_discrete:usize, 
    x_min:f64, x_max:f64, 
    fn_inv_log:&Vec<Complex<f64>>
)->Vec<f64> {
    let discrete_cf_log=convert_log_cf_to_real_exp( 
        x_min, x_max, 
        fn_inv_log
    );
    compute_inv_discrete(
        x_discrete, x_min, x_max, 
        &discrete_cf_log
    )
}

/**
    Computes the expectation given a characteristic function fnInv at the discrete points xRange in xmin, xmax and functions of the expectation vK: E[f(vk)]. This is used if the CF is of a Levy process.  See Fang Oosterlee (2007) for more information.
    @xDiscrete Number of discrete points in density domain
    @uDiscrete Number of discrete points in the complex domain
    @xmin Minimum number in the density domain
    @xmax Maximum number in the density domain
    @fnInv Characteristic function of the density.  May be computationally intense to compute (eg, if using the combined CF of millions of loans)
    @vK Function (parameters u and x) or vector to multiply discrete characteristic function by.  
    @returns approximate expectation
*/
pub fn compute_expectation_levy<T, S>(
    x_discrete:usize, u_discrete:usize, 
    x_min:f64, x_max:f64, fn_inv:T, 
    vk:S
)->Vec<f64>
    where T:Fn(&Complex<f64>)->Complex<f64>+std::marker::Sync+std::marker::Send,
    S:Fn(f64, f64, usize)->f64+std::marker::Sync+std::marker::Send
{
    let discrete_cf=compute_discrete_cf(
        x_min, x_max, u_discrete,
        fn_inv
    );
    compute_convolution_levy(
        x_discrete, x_min, x_max, 
        &discrete_cf,
        vk
    )
}


/**
    Computes the expectation given a discretized characteristic function discreteCF (of size uDiscrete) at the discrete points xRange in xmin, xmax and functions of the expectation vK: E[f(vk)]. This is used if the CF is of a Levy process.  See Fang Oosterlee (2007) for more information.
    @xDiscrete Number of discrete points in density domain
    @uDiscrete Number of discrete points in the complex domain
    @xmin Minimum number in the density domain
    @xmax Maximum number in the density domain
    @discreteCF vector of characteristic function of the density at discrete U
    @vK Function (parameters u and x) or vector to multiply discrete characteristic function by.  
    @returns approximate expectation
*/
pub fn compute_expectation_levy_discrete<S>(
    x_discrete:usize, x_min:f64, x_max:f64,
    discrete_cf:&Vec<Complex<f64>>,
    vk:S
)->Vec<f64>
    where S:Fn(f64, f64, usize)->f64+std::marker::Sync+std::marker::Send
{
    compute_convolution_levy(
        x_discrete, x_min, x_max,
        discrete_cf, vk
    )
}


 /**
    Computes the expectation given a characteristic function fnInv at the discrete points xRange in xmin, xmax and functions of the expectation vK: E[f(vk)]. Only used for non-Levy processes.  See Fang Oosterlee (2007) for more information.
    @xDiscrete Number of discrete points in density domain
    @uDiscrete Number of discrete points in the complex domain
    @xmin Minimum number in the density domain
    @xmax Maximum number in the density domain
    @fnInv Characteristic function of the density.  May be computationally intense to compute (eg, if using the combined CF of millions of loans)
    @vK Function (parameters u and x) or vector to multiply discrete characteristic function by.  
    @returns approximate expectation
*/

pub fn compute_expectation<T, S>(
    x_discrete:usize,
    u_discrete:usize,
    x_min:f64, x_max:f64,
    fn_inv:T,
    vk:S
)->Vec<f64>
    where T:Fn(&Complex<f64>)->Complex<f64>+std::marker::Sync+std::marker::Send,
    S:Fn(f64, f64, usize)->f64+std::marker::Sync+std::marker::Send
{
    let discrete_cf=compute_discrete_cf_real(
        x_min, x_max,
        u_discrete,
        fn_inv
    );
    compute_convolution(
        x_discrete,
        x_min, x_max,
        &discrete_cf,
        vk
    )
}

/**
    Computes the expectation given a discretized characteristic function discreteCF (of size uDiscrete) at the discrete points xRange in xmin, xmax and functions of the expectation vK: E[f(vk)]. Only used for non-Levy processes.  See Fang Oosterlee (2007) for more information.
    @xDiscrete Number of discrete points in density domain
    @uDiscrete Number of discrete points in the complex domain
    @xmin Minimum number in the density domain
    @xmax Maximum number in the density domain
    @discreteCF vector of characteristic function of the density at discrete U
    @vK Function (parameters u and x) or vector to multiply discrete characteristic function by.  
    @returns approximate expectation
*/

pub fn compute_expectation_discrete<T, S>(
    x_discrete:usize,
    x_min:f64, x_max:f64,
    fn_inv:&Vec<f64>, vk:S
)->Vec<f64>
    where S:Fn(f64, f64, usize)->f64+std::marker::Sync+std::marker::Send
{
    compute_convolution(
        x_discrete,
        x_min, x_max,
        fn_inv, vk
    )
}

 /**
    Computes the expectation given a characteristic function fnInv at the vector of discrete points xValues and functions of the expectation vK: E[f(vk)]. This is used if the CF is of a Levy process.  See Fang Oosterlee (2007) for more information.
    @xValues Array of x values to compute the function at.
    @uDiscrete Number of discrete points in the complex domain
    @fnInv Characteristic function of the density.  May be computationally intense to compute (eg, if using the combined CF of millions of loans)
    @vK Function (parameters u and x) or vector to multiply discrete characteristic function by.  
    @extraFn Function (parameters expectation, x, and index) to multiply result by.  If not provided, defaults to raw expectation.
    @returns approximate expectation
*/

pub fn compute_expectation_vec_levy_g<T, S, U>(
    x_values:&Vec<f64>,
    u_discrete:usize,
    fn_inv:T,
    vk:S, extra_fn:U
)->Vec<f64>
    where T:Fn(&Complex<f64>)->Complex<f64>+std::marker::Sync+std::marker::Send,
    S:Fn(f64, f64, usize)->f64+std::marker::Sync+std::marker::Send,
    U:Fn(f64, f64, usize)->f64+std::marker::Sync+std::marker::Send
{
    let discrete_cf=compute_discrete_cf(
        x_values[0],
        x_values[x_values.len()-1],
        u_discrete,
        fn_inv
    );
    compute_convolution_vec_levy_g(
        x_values, 
        &discrete_cf,
        vk, extra_fn
    )
}
/**
    Computes the expectation given a characteristic function fnInv at the vector of discrete points xValues and functions of the expectation vK: E[f(vk)]. This is used if the CF is of a Levy process.  See Fang Oosterlee (2007) for more information.
    @xValues Array of x values to compute the function at.
    @uDiscrete Number of discrete points in the complex domain
    @fnInv Characteristic function of the density.  May be computationally intense to compute (eg, if using the combined CF of millions of loans)
    @vK Function (parameters u and x) or vector to multiply discrete characteristic function by.  
    @returns approximate expectation
*/

pub fn compute_expectation_vec_levy<T, S>(
    x_values:&Vec<f64>,
    u_discrete:usize,
    fn_inv:T,
    vk:S
)->Vec<f64>
    where T:Fn(&Complex<f64>)->Complex<f64>+std::marker::Sync+std::marker::Send,
    S:Fn(f64, f64, usize)->f64+std::marker::Sync+std::marker::Send
{
    compute_expectation_vec_levy_g(
        x_values, 
        u_discrete,
        fn_inv, vk, |v, _, _|v
    )
}

 /**
    Computes the expectation given a discretized characteristic function discreteCF (of size uDiscrete) at the vector of discrete points xValues and functions of the expectation vK: E[f(vk)]. This is used if the CF is of a Levy process.  See Fang Oosterlee (2007) for more information.
    @xValues x values to compute the function at.
    @uDiscrete Number of discrete points in the complex domain
    @discreteCF vector of characteristic function of the density at discrete U
    @vK Function (parameters u and x) or vector to multiply discrete characteristic function by.  
    @returns approximate expectation
*/
pub fn compute_expectation_vec_levy_discrete<S>(
    x_values: &Vec<f64>,
    discrete_cf: &Vec<Complex<f64>>,
    vk:S
)->Vec<f64>
    where S:Fn(f64, f64, usize)->f64+std::marker::Sync+std::marker::Send
{
    compute_convolution_vec_levy(
        x_values, discrete_cf, vk
    )
}

/**
    Computes the expectation given a characteristic function fnInv at the array of discrete points in xValues and functions of the expectation vK: E[f(vk)]. This is used if the CF is not for a Levy process.  See Fang Oosterlee (2007) for more information.
    @xValues Array of x values to compute the function at.
    @uDiscrete Number of discrete points in the complex domain
    @fnInv Characteristic function of the density.  May be computationally intense to compute (eg, if using the combined CF of millions of loans)
    @vK Function (parameters u and x) or vector to multiply discrete characteristic function by.  
    @returns approximate expectation
*/

pub fn compute_expectation_vec<T, S>(
    x_values:&Vec<f64>,
    u_discrete:usize,
    fn_inv:T, vk:S
)->Vec<f64>
    where T:Fn(&Complex<f64>)->Complex<f64>+std::marker::Sync+std::marker::Send,
    S:Fn(f64, f64, usize)->f64+std::marker::Sync+std::marker::Send
{
    let discrete_cf=compute_discrete_cf_real(
        x_values[0],
        x_values[x_values.len()-1],
        u_discrete,
        fn_inv
    );
    compute_convolution_vec(
        x_values,
        &discrete_cf,
        vk
    )
}

/**
    Computes the expectation given a discretized characteristic function discreteCF (of size uDiscrete) at the discrete points in xValues and functions of the expectation vK: E[f(vk)]. This is used if the CF is not for a Levy process.  See Fang Oosterlee (2007) for more information.
    @xValues x values to compute the function at.
    @uDiscrete Number of discrete points in the complex domain
    @discreteCF vector of characteristic function of the density at discrete U
    @vK Function (parameters u and x) or vector to multiply discrete characteristic function by.  
    @returns approximate expectation
*/
pub fn compute_expectation_vec_discrete<S>(
    x_values:&Vec<f64>,
    discrete_cf:&Vec<f64 >,
    vk:S
) -> Vec<f64>
    where S:Fn(f64, f64, usize)->f64+std::marker::Sync+std::marker::Send
{
    compute_convolution_vec(
        x_values, 
        discrete_cf,
        vk
    )
}

pub fn compute_expectation_at_point_levy<T, S>(
    x_value:f64,
    x_min:f64,
    x_max:f64,
    u_discrete:usize,
    fn_inv:T,
    vk:S
) -> f64 
    where T:Fn(&Complex<f64>)->Complex<f64>+std::marker::Sync+std::marker::Send,
    S:Fn(f64, f64, usize)->f64+std::marker::Sync+std::marker::Send
{
    let discrete_cf=compute_discrete_cf(
        x_min,
        x_max,
        u_discrete,
        fn_inv
    );
    compute_convolution_at_point_levy(
        x_value,
        x_min,
        x_max,
        &discrete_cf,
        vk
    )
}

pub fn compute_expectation_at_point_levy_discrete<T, S>(
    x_value:f64,
    x_min:f64, 
    x_max:f64,
    fn_inv:&Vec<Complex<f64>>,
    vk:S
)->f64
    where T:Fn(&Complex<f64>)->Complex<f64>+std::marker::Sync+std::marker::Send,
    S:Fn(f64, f64, usize)->f64+std::marker::Sync+std::marker::Send
{
    compute_convolution_at_point_levy(
        x_value,
        x_min,
        x_max,
        fn_inv,
        vk
    )
}

pub fn compute_expectation_at_point<T, S>(
    x_value:f64,
    x_min:f64,
    x_max:f64,
    u_discrete:usize,
    fn_inv:T,
    vk:S
)->f64
    where T:Fn(&Complex<f64>)->Complex<f64>+std::marker::Sync+std::marker::Send,
    S:Fn(f64, f64, usize)->f64+std::marker::Sync+std::marker::Send
{
    let discrete_cf=compute_discrete_cf_real(
        x_min, x_max,
        u_discrete,
        fn_inv
    );
    compute_convolution_at_point(
        x_value,
        x_min,
        x_max,
        &discrete_cf,
        vk
    )
}
pub fn compute_expectation_at_point_discrete<T, S>(
    x_value:f64,
    x_min:f64,
    x_max:f64,
    fn_inv:&Vec<f64>,
    vk:S
)->f64
    where T:Fn(&Complex<f64>)->Complex<f64>+std::marker::Sync+std::marker::Send,
    S:Fn(f64, f64, usize)->f64+std::marker::Sync+std::marker::Send
{
    compute_convolution_at_point(
        x_value,
        x_min, 
        x_max,
        fn_inv,
        vk
    )
}



#[cfg(test)]
mod tests {
    use super::*;
    fn vk_cdf(
        x:f64, u:f64, a:f64, b:f64, k:usize
    ){
        if k==0{x-a} else { ((x-a)*u).sin()/u }
    }
    #[test]
    fn test_compute_x_range(){
        assert_eq!(
            compute_x_range(5, 0.0, 1.0),
            vec![0.0, 0.25, 0.5, 0.75, 1.0]
        )
    }

    #[test]
    fn test_compute_inv(){
        let mu=2.0;
        let sigma=1.0;
        let num_x=5;
        let num_u=256;
        let x_min=-3.0;
        let x_max=7.0;
        let norm_cf=|&u|(u*mu+0.5*u*u*sigma*sigma).exp();

        let ref_normal=compute_x_range(num_x, x_min, x_max).iter().map(|x|{
            ((x-mu).pow(2.0)/(2.0*sigma*sigma)).exp()/(sigma*(2*PI).sqrt())
        }).collect();
        
        let my_inverse=compute_inv(num_x, num_u, x_min, x_max, norm_cf);
        
        for (index, x) in ref_normal.enumerate(){
            assert_eq!(x, my_inverse[index]);
        }
    }

}
