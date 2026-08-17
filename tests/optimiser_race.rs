//! Which bound-constrained optimiser converges better on Asterism's problems.
//!
//! Every fit in this package goes through one library, so which one is not a
//! detail. The first choice was made by measurement: a hand-written projected
//! BFGS was raced against `lbfgsb-rs-pure` and lost, and the reasoning was
//! written down. The second change -- to `rcompat-lbfgsb` -- was made in a
//! commit with a one-line title and an empty body, so nothing recorded why.
//!
//! This measures it rather than leaving it to a commit message. `lbfgsb-rs-pure`
//! is a development dependency only, present so this comparison can be rerun,
//! and it is not part of the shipped crate.
//!
//! Run with:
//!
//!     cargo test --release --test optimiser_race -- --nocapture

use asterism::ComponentModel;
use lbfgsb_rs_pure::LBFGSB;
use nalgebra::{DMatrix, DVector};
use rcompat_lbfgsb::{Bounds, OptimControl, optim_lbfgsb_with_gradient};

/// Sibling pairs in households of four, drawn from the model.
fn problem() -> (ComponentModel, DVector<f64>) {
    let pairs = 200;
    let n = 2 * pairs;
    let mut a = DMatrix::<f64>::identity(n, n);
    for pair in 0..pairs {
        a[(2 * pair, 2 * pair + 1)] = 0.5;
        a[(2 * pair + 1, 2 * pair)] = 0.5;
    }
    let mut h = DMatrix::<f64>::identity(n, n);
    for household in 0..n / 4 {
        for i in 0..4 {
            for j in 0..4 {
                h[(household * 4 + i, household * 4 + j)] = 1.0;
            }
        }
    }
    let design = DMatrix::from_element(n, 1, 1.0);

    let mut seed = 20_260_812u64;
    let mut next = || {
        let mut total = 0.0;
        for _ in 0..3 {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            total += ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5;
        }
        total * 2.0
    };
    let half = 0.5f64.sqrt();
    let mut y = DVector::<f64>::zeros(n);
    for household in 0..n / 4 {
        let shared = next();
        for i in 0..4 {
            y[household * 4 + i] += 0.2f64.sqrt() * shared;
        }
    }
    for pair in 0..pairs {
        let common = next();
        for i in 0..2 {
            let genetic = half * common + half * next();
            y[2 * pair + i] += 0.4f64.sqrt() * genetic + 0.4f64.sqrt() * next();
        }
    }
    (ComponentModel::build(&[a, h], &design).expect("valid"), y)
}

#[test]
fn the_two_optimisers_are_raced_on_the_same_problem() {
    let (model, y) = problem();
    let starts: Vec<Vec<f64>> = vec![
        vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        vec![0.05, 0.05, 0.9],
        vec![0.6, 0.05, 0.35],
        vec![0.1, 0.6, 0.3],
    ];
    let lower = vec![0.0; 3];
    let upper = vec![f64::INFINITY; 3];

    println!("\nOne trait, three components, REML. Four starts, the same objective.\n");
    println!(
        "{:<28}{:>18}{:>16}{:>12}",
        "start", "negative loglik", "|projected g|", "evaluations"
    );

    let mut wins = [0usize; 2];
    for start in &starts {
        // rcompat-lbfgsb, the library the crate ships with.
        let mut calls_r = 0usize;
        let value_r = |c: &[f64]| {
            calls_r += 1;
            model
                .objective_for_test(c, &y, true, false)
                .map_or(1e30, |(v, _)| v)
        };
        let gradient_r = |c: &[f64]| {
            model
                .objective_for_test(c, &y, true, true)
                .map_or_else(|| vec![0.0; 3], |(_, g)| g)
        };
        let bounds = Bounds::new(lower.clone(), upper.clone()).expect("bounds");
        let mut control = OptimControl::default_for_dimension(3);
        control.maxit = 500;
        control.fnscale = 1.0;
        control.parscale = vec![1.0; 3];
        control.factr = 0.0;
        control.pgtol = 1e-9;
        control.lmm = 3;
        let solution =
            optim_lbfgsb_with_gradient(start.clone(), bounds, value_r, gradient_r, control)
                .expect("rcompat runs");
        let (value_rc, gradient_rc) = model
            .objective_for_test(&solution.par, &y, true, true)
            .expect("evaluates");
        let projected_rc = projected(&solution.par, &gradient_rc);

        // lbfgsb-rs-pure, the library it replaced.
        let mut x = start.clone();
        let mut calls_p = 0usize;
        let mut solver = LBFGSB::new(3);
        let mut f_and_grad = |c: &[f64]| {
            calls_p += 1;
            model
                .objective_for_test(c, &y, true, true)
                .map_or((1e30, vec![0.0; 3]), |(v, g)| (v, g))
        };
        let pure = solver.minimize(&mut x, &lower, &upper, &mut f_and_grad);
        let (value_pu, projected_pu, calls_pu) = match pure {
            Ok(sol) => {
                let point = sol.x.clone();
                let (v, g) = model
                    .objective_for_test(&point, &y, true, true)
                    .expect("evaluates");
                (v, projected(&point, &g), calls_p)
            }
            Err(_) => (f64::INFINITY, f64::INFINITY, calls_p),
        };

        println!(
            "{:<28}{:>18.6}{:>16.2e}{:>12}",
            format!("{start:?} rcompat"),
            value_rc,
            projected_rc,
            calls_r
        );
        println!(
            "{:<28}{:>18.6}{:>16.2e}{:>12}",
            "  pure", value_pu, projected_pu, calls_pu
        );

        // **The criterion is whether it reached a stationary point, not which
        // objective is lower.** Scoring on the objective alone rewards a search
        // for being less bad at a point neither has solved, which is how the
        // first reading of this table made the two look even. A run that stops
        // far from stationary has not found an optimum whatever value it
        // reports.
        if projected_rc < 1e-6 {
            wins[0] += 1;
        }
        if projected_pu < 1e-6 {
            wins[1] += 1;
        }
    }
    println!(
        "\nreached a stationary point (projected gradient below 1e-6): \
         rcompat {} of {}, pure {} of {}",
        wins[0],
        starts.len(),
        wins[1],
        starts.len()
    );
    println!(
        "A search that stops far from stationary has not found an optimum, whatever\n\
         value it reports. The fits use several starts and keep the best, so a start\n\
         that goes nowhere costs time rather than correctness.\n"
    );
    assert!(
        wins[0] > 0,
        "the shipped optimiser reached no stationary point on any start"
    );
}

/// The projected gradient: a parameter resting on its bound is converged when
/// its derivative pushes outward, which an unprojected norm calls a failure.
fn projected(x: &[f64], gradient: &[f64]) -> f64 {
    x.iter()
        .zip(gradient)
        .map(|(v, g)| if *v <= 0.0 { g.min(0.0) } else { *g })
        .fold(0.0f64, |worst, g| worst.max(g.abs()))
}
