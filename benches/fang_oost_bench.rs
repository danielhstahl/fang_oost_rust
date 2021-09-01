#[macro_use]
extern crate criterion;
extern crate fang_oost;
extern crate num_complex;
use criterion::Criterion;
use num_complex::Complex;
use rayon::prelude::*;

fn bench_fang_oost_normal_density(c: &mut Criterion) {
    let mu = 2.0;
    let sigma = 1.0;
    let num_x = 5;
    let num_u = 256;
    let x_min = -3.0;
    let x_max = 7.0;
    let norm_cf = move |u: &Complex<f64>| (u * mu + 0.5 * u * u * sigma * sigma).exp();
    c.bench_function("normal density", move |b| {
        b.iter(move || {
            let my_x_domain = fang_oost::get_x_domain(num_x, x_min, x_max);
            let discrete_cf = fang_oost::get_discrete_cf(num_u, x_min, x_max, &norm_cf);
            let _my_inverse: Vec<fang_oost::GraphElement> = fang_oost::get_expectation_real(
                x_min,
                x_max,
                my_x_domain,
                &discrete_cf,
                move |u, x, _| (u * (x - x_min)).cos(),
            )
            .collect();
        })
    });
}

criterion_group!(benches, bench_fang_oost_normal_density);
criterion_main!(benches);
