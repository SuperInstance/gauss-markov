# gauss-markov

A Rust library for simulating and analyzing the Gauss-Markov process (also known as the Ornstein-Uhlenbeck process). Implements the stochastic differential equation `dx = -θ(x - μ)dt + σdW` via the Euler-Maruyama method, along with parameter estimation and theoretical statistical properties (stationary distribution, autocorrelation function, PDF).

## Why It Matters

The Gauss-Markov process is the canonical model for **mean-reverting noise**. It appears in physics (Brownian motion in a harmonic potential — the Langevin equation), finance (Vasicek interest-rate model), engineering (sensor drift modeling, kalman filter process noise), and biology (ion channel gating). Unlike pure Brownian motion, which wanders without bound, the OU process is pulled toward a long-term mean μ at rate θ, making it stationary and ergodic. This makes it the natural model for any system that fluctuates around equilibrium with memory.

The key insight: the OU process is the unique Gauss-Markov process — it is simultaneously Gaussian-distributed and Markovian (memoryless given the current state). This dual property is why it serves as the process noise model in the Kalman-Bucy filter, the workhorse of estimation theory.

## How It Works

### Stochastic Differential Equation

The continuous-time SDE is:

```
dx(t) = -θ(x(t) - μ) dt + σ dW(t)
```

where:
- **θ > 0**: Mean reversion rate (how fast the process returns to μ)
- **μ**: Long-term equilibrium level
- **σ > 0**: Volatility (diffusion coefficient)
- **W(t)**: Standard Wiener process (Brownian motion)

### Euler-Maruyama Discretization

The numerical scheme discretizes time into steps of size Δt:

```
x_{n+1} = x_n - θ(x_n - μ)Δt + σ√(Δt) · Z_n
```

where Z_n ~ N(0, 1) are i.i.d. standard normal random variables generated via the Box-Muller transform:

```
Z = √(-2 ln U₁) · cos(2π U₂)
```

**Time complexity**: O(N) for N = ⌈(t_end - t_start) / dt⌉ steps.

**Space complexity**: O(N) for storing the trajectory.

### Stationary Distribution

The OU process has a closed-form stationary distribution:

```
X_∞ ~ N(μ, σ²/(2θ))
```

The stationary variance `σ²/(2θ)` follows from the fluctuation-dissipation theorem: stronger mean reversion (large θ) reduces variance; stronger noise (large σ) increases it.

### Autocorrelation Function

```
ρ(τ) = E[(X(t) - μ)(X(t+τ) - μ)] / Var(X) = exp(-θ|τ|)
```

The autocorrelation decays exponentially with timescale τ_c = 1/θ. This is the **memory** of the process — after τ_c, past values are effectively forgotten.

### Parameter Estimation

Three estimators from sampled data:

| Parameter | Estimator | Formula |
|-----------|-----------|---------|
| θ | From lag-1 autocorrelation | `θ̂ = -ln(ρ̂(Δt)) / Δt` |
| μ | Sample mean | `μ̂ = (1/n) Σ xᵢ` |
| σ | From stationary variance | `σ̂ = √(2θ̂ · Var(X))` |

The autocorrelation estimator derives from the fact that for the discrete-time OU process sampled at interval Δt:

```
ρ(Δt) = exp(-θΔt)  ⟹  θ = -ln(ρ) / Δt
```

### Stability Condition

The Euler-Maruyama scheme is stable when θΔt < 2 (the deterministic part doesn't overshoot). For typical parameters (θ ~ 1, Δt ~ 0.01), this is easily satisfied.

## Quick Start

```rust
use gauss_markov::{GaussMarkovParams, simulate};

let params = GaussMarkovParams::new(1.0, 0.0, 0.5).unwrap();
let mut rng = || { /* uniform [0,1) generator */ };

let result = simulate(params, 0.0, 0.0, 10.0, 0.01, &mut rng);
println!("Trajectory length: {}", result.values.len());
println!("Sample mean: {:.3}", result.values.iter().sum::<f64>() / result.values.len() as f64);
```

## API

### `GaussMarkovParams`

| Method | Returns | Description |
|--------|---------|-------------|
| `new(theta, mu, sigma)` | `Result<Self, &str>` | Validated constructor (θ > 0, σ > 0) |
| `stationary_variance()` | `f64` | σ²/(2θ) |
| `stationary_std()` | `f64` | √(σ²/(2θ)) |
| `autocorrelation(tau)` | `f64` | exp(−θ|τ|) |
| `stationary_pdf(x)` | `f64` | Normal PDF at x |

### Free Functions

| Function | Description |
|----------|-------------|
| `simulate(params, x0, t_start, t_end, dt, rng)` | Euler-Maruyama trajectory |
| `estimate_theta(values, dt)` | ML estimate of θ from samples |
| `estimate_mu(values)` | Sample mean |
| `estimate_sigma(values, theta)` | σ from stationary variance |

## Architecture Notes

The OU process embodies **γ + η = C** in estimation theory. The process noise (**η**) represents unmodeled dynamics perturbing the system. The Kalman filter (**γ**) uses the OU process model to predict how uncertainty evolves, producing optimal state estimates (**C**). The autocorrelation function ρ(τ) = e^{−θτ} is the transfer function: it tells the filter how much past information to retain. Longer memory (smaller θ) means more smoothing; shorter memory (larger θ) means more responsiveness. This tradeoff between noise rejection and responsiveness is the fundamental tension in feedback control.

## References

- **Ornstein-Uhlenbeck process**: Uhlenbeck, G. E., & Ornstein, L. S. "On the theory of the Brownian motion." *Physical Review* 36.5 (1930): 823.
- **Euler-Maruyama method**: Kloeden, P. E., & Platen, E. *Numerical Solution of Stochastic Differential Equations.* Springer, 1992.
- **Fluctuation-dissipation theorem**: Kubo, R. "The fluctuation-dissipation theorem." *Reports on Progress in Physics* 29.1 (1966): 255.
- **Vasicek model**: Vasicek, O. "An equilibrium characterization of the term structure." *Journal of Financial Economics* 5.2 (1977): 177–188.
- **Parameter estimation for OU process**: Kessler, M. "Estimation of an ergodic diffusion from discrete observations." *Scandinavian Journal of Statistics* 24.2 (1997): 211–229.

## License

MIT
