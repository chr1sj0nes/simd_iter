#![feature(portable_simd)]

use std::simd::Simd;

use criterion::{criterion_group, criterion_main, Criterion};
use rand::Rng;

use simd_iter::{NumVectorReductions, SimdIterable};

fn criterion_benchmark(c: &mut Criterion) {
    let mut xs = vec![0.; 10_000_000];
    rand::thread_rng().fill(xs.as_mut_slice());
    let mut ys = vec![0.; 10_000_000];
    rand::thread_rng().fill(ys.as_mut_slice());

    c.bench_function("sum", |b| b.iter(|| xs.iter().sum::<f64>()));
    c.bench_function("vec sum", |b| b.iter(|| xs.simd_iter().scalar_sum()));
    c.bench_function("dot", |b| {
        b.iter(|| xs.iter().zip(&ys).map(|(x, y)| x * y).sum::<f64>())
    });
    c.bench_function("vec dot", |b| {
        b.iter(|| {
            xs.simd_iter()
                .zip(ys.simd_iter())
                .map(|(xv, yv)| xv * yv)
                .reduce(std::ops::Add::add)
                .map(Simd::<f64, 32>::horizontal_sum)
                .unwrap()
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
