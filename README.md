# Gauss-Markov Process Simulator

**A Rust library for simulating and analyzing the Gauss-Markov (Ornstein-Uhlenbeck) process** — the canonical mean-reverting stochastic differential equation `dx = -θ(x-μ)dt + σdW` — with Euler-Maruyama integration, parameter estimation, and closed-form stationary statistics.

## Why It Matters

The Ornstein-Uhlenbeck (OU) process is the mathematical foundation of mean-reverting models across science and finance: interest rate models (Vasicek model), neuronal membrane potential (leaky integrate-and-fire), temperature anomalies in climate modeling, and velocity autocorrelation in Brownian dynamics. Unlike geometric Brownian motion, the OU process is stationary and ergodic — it has a well-defined equilibrium distribution and autocorrelation function. This library provides both the simulation engine (to generate sample paths) and the estimation routines (to fit OU parameters from observed data), making it a complete toolkit for time-series analysis.

## How It Works

**Simulation** uses the Euler-Maruyama method — the simplest numerical SDE integrator. For each time step `dt`, the update is `x_{n+1} = x_n - θ(x_n - μ)dt + σ√dt · Z` where `Z ~ N(0,1)` is a standard normal random variable generated via Box-Muller transform from a user-supplied uniform RNG. The `√dt` scaling comes from the quadratic variation of Brownian motion.

**Closed-form statistics**: The stationary variance is `σ²/(2θ)`, the autocorrelation at lag τ is `exp(-θ|τ|)`, and the stationary distribution is `N(μ, σ²/(2θ))`. These are computed analytically — **O(1)** — without needing to run the simulation.

**Parameter estimation**: `estimate_theta` uses the sample autocorrelation at lag 1: `θ ≈ -ln(ρ(Δt))/Δt`. `estimate_mu` is the sample mean. `estimate_sigma` uses the relation `σ² = 2θ·Var(X)`. These are method-of-moments estimators — **O(n)** in the sample size.

## Quick Start

```rust
use gauss_markov::{GaussMarkovParams, simulate, estimate_theta, estimate_mu};

fn main() {
    // Define an OU process: θ=1.0, μ=5.0, σ=0.5
    let params = GaussMarkovParams::new(1.0, 5.0, 0.5).unwrap();

    // Simulate from t=0 to t=10 with dt=0.01
    let mut rng_state = 0u64;
    let result = simulate(
        params,
        0.0,    // x0
        0.0,    // t_start
        10.0,   // t_end
        0.01,   // dt
        &mut || {
            // Simple LCG random number generator
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (rng_state >> 11) as f64 / (1u64 << 53) as f64
        },
    );

    println!("Generated {} samples", result.values.len());

    // Estimate parameters from the simulated data
    let theta_hat = estimate_theta(&result.values, 0.01);
    let mu_hat = estimate_mu(&result.values);
    println!("Estimated θ = {:.3} (true: 1.0)", theta_hat);
    println!("Estimated μ = {:.3} (true: 5.0)", mu_hat);

    // Closed-form stationary statistics
    println!("Stationary std: {:.4}", params.stationary_std());
    println!("Autocorrelation at τ=1: {:.4}", params.autocorrelation(1.0));
}
```

## API

| Type / Function | Complexity | Description |
|---|---|---|
| `GaussMarkovParams::new(θ, μ, σ)` | **O(1)** | Validated parameter construction |
| `simulate(params, x0, t0, t1, dt, rng)` | **O(n)** | Euler-Maruyama simulation |
| `estimate_theta(values, dt)` | **O(n)** | Estimate θ from lag-1 autocorrelation |
| `estimate_mu(values)` | **O(n)** | Sample mean |
| `estimate_sigma(values, θ)` | **O(n)** | Estimate σ from stationary variance |
| `stationary_variance()` | **O(1)** | `σ²/(2θ)` |
| `autocorrelation(τ)` | **O(1)** | `exp(-θ|τ|)` |
| `stationary_pdf(x)` | **O(1)** | Gaussian density at x |

## Architecture Notes

Part of the SuperInstance stochastic modeling suite. Companion crates include `fredholm-equation` (integral equations) and `hermite-polynomial` (quadrature). See the [Architecture Guide](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## License

MIT
