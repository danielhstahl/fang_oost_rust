extern crate num;
extern crate num_complex;
extern crate black_scholes;
extern crate rayon;
#[macro_use]
#[cfg(test)]
extern crate approx;
use num_complex::Complex;
use num::traits::{Zero};
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
fn convolute<T>(cf_incr:&Complex<f64>, x:f64, u:f64, u_index:usize, vk:T)->f64
    where T:Fn(f64, f64, usize)->f64
{
    (cf_incr*(get_complex_u(u)*x).exp()).re*vk(u, x, u_index)
}


fn adjust_index_cmpl(element:&Complex<f64>, index:usize)->Complex<f64>
{
    if index==0{element*0.5} else {*element}
}

pub fn get_discrete_cf<T>(
    num_u:usize,
    x_min:f64,
    x_max:f64,
    fn_inv:T
)->Vec<Complex<f64>>
where T:Fn(&Complex<f64>)->Complex<f64>+std::marker::Sync+std::marker::Send,
{
    let du=compute_du(x_min, x_max);
    let cp=compute_cp(du);
    (0..num_u).into_par_iter().map(|index|{
        let u=get_complex_u(get_u(du, index));
        fn_inv(&u)*(-u*x_min).exp()*cp //potentially expensive...want to perform this once...which means that we then need to actually make this into a vector not an iterator.  
    }).collect()
}

/**
 * Generates expectation over equal mesh
 * in the real domain.  The "type" of
 * expectation is handled by the vk
 * function.  
 * @num_x Number of discrete steps in 
 * the real domain.
 * @num_u Number of discrete steps in
 * the complex domain.
 * @x_min Lower truncation of the real
 * domain.
 * @x_max Upper truncation of the 
 * complex domain.
 * @fn_inv Characteristic function.
 * @vk Function which controls what kind
 * of expectation (the integrand).
 */
pub fn get_expectation_x<T, U>(
    num_x:usize,
    num_u:usize,
    x_min:f64,
    x_max:f64,
    fn_inv:T,
    vk:U
)->Vec<f64>
    where T:Fn(&Complex<f64>)->Complex<f64>+std::marker::Sync+std::marker::Send,
    U:Fn(f64, f64, usize)->f64+std::marker::Sync+std::marker::Send
{
    let dx=compute_dx(num_x, x_min, x_max);
    let du=compute_du(x_min, x_max);
    let cp=compute_cp(du);
    //get discrete cf
    let cf_discrete=get_discrete_cf(
        num_u, 
        x_min, x_max, fn_inv
    );
    //for every x, iterate over discrete cf
    (0..num_x).into_par_iter().map(|x_index|{
        let x=get_x(x_min, dx, x_index);
        cf_discrete.iter().enumerate().fold(f64::zero(), |s, (index, cf_incr)|{
            let cf_incr_m=adjust_index_cmpl(&cf_incr, index);
            s+convolute(&cf_incr_m, x, get_u(du, index), index, &vk)
        })
    }).collect()
}

/**
 * Generates expectation over real domain
 * provided by the input "x".  The "type" 
 * of expectation is handled by the vk
 * function.  
 * @num_u Number of discrete steps in
 * the complex domain.
 * @x Vector of elements in real domain.
 * @fn_inv Characteristic function.
 * @vk Function which controls what kind
 * of expectation (the integrand).
 */
pub fn get_expectation_discrete<T, U>(
    num_u:usize,
    x:&Vec<f64>,
    fn_inv:T,
    vk:U
)->Vec<f64>
    where T:Fn(&Complex<f64>)->Complex<f64>+std::marker::Sync+std::marker::Send,
    U:Fn(f64, f64, usize)->f64+std::marker::Sync+std::marker::Send
{
    let x_max=*x.last().unwrap();
    let x_min=*x.first().unwrap();
    let du=compute_du(x_min, x_max);
    let cp=compute_cp(du);
    //get discrete cf
    let cf_discrete=get_discrete_cf(
        num_u, 
        x_min, x_max, fn_inv
    );
    //for every x, iterate over discrete cf
    x.par_iter().map(|&x_value|{
        cf_discrete.iter().enumerate().fold(f64::zero(), |s, (index, cf_incr)|{
            let cf_incr_m=adjust_index_cmpl(&cf_incr, index);
            s+convolute(&cf_incr_m, x_value, get_u(du, index), index, &vk)
        })
    }).collect()
}

pub fn get_density<T>(
    num_u:usize,
    x:&Vec<f64>,
    fn_inv:T
)->Vec<f64>
    where T:Fn(&Complex<f64>)->Complex<f64>+std::marker::Sync+std::marker::Send,
{
    let x_min=x.first().unwrap();
    get_expectation_discrete(
        num_u, 
        &x, 
        fn_inv,
        |u, x, _|(u*(x-x_min)).cos()
    )
}

pub fn get_density_x<T>(
    num_x:usize,
    num_u:usize,
    x_min:f64,
    x_max:f64,
    fn_inv:T
)->Vec<f64>
    where T:Fn(&Complex<f64>)->Complex<f64>+std::marker::Sync+std::marker::Send,
{
    get_expectation_x(
        num_x,
        num_u,
        x_min,
        x_max,
        fn_inv,
        |u, x, _|(u*(x-x_min)).cos()
    )
}




#[cfg(test)]
mod tests {
    use super::*;
    fn vk_cdf(
        x:f64, u:f64, a:f64, b:f64, k:usize
    )->f64{
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
        let norm_cf=|u:&Complex<f64>|(u*mu+0.5*u*u*sigma*sigma).exp();

        let ref_normal:Vec<f64>=compute_x_range(num_x, x_min, x_max).iter().map(|x|{
            (-(x-mu).powi(2)/(2.0*sigma*sigma)).exp()/(sigma*(2.0*PI).sqrt())
        }).collect();
        
        let my_inverse=get_density_x(num_x, num_u, x_min, x_max, norm_cf);
        
        for (index, x) in ref_normal.iter().enumerate(){
            assert_abs_diff_eq!(*x, my_inverse[index], epsilon=0.001);
        }
    }


}
