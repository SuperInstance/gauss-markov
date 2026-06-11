//! Gauss-Markov process simulation.
//!
//! Implements the Gauss-Markov (Ornstein-Uhlenbeck) process:
//! dx = -θ(x - μ)dt + σdW
//!
//! Provides simulation, parameter estimation, and statistical analysis.

use std::f64::consts::PI;

/// Parameters defining a Gauss-Markov process.
#[derive(Debug, Clone, Copy)]
pub struct GaussMarkovParams {
    /// Mean reversion rate (θ > 0)
    pub theta: f64,
    /// Long-term mean (μ)
    pub mu: f64,
    /// Volatility / diffusion coefficient (σ > 0)
    pub sigma: f64,
}

impl GaussMarkovParams {
    /// Create new parameters with validation.
    pub fn new(theta: f64, mu: f64, sigma: f64) -> Result<Self, &'static str> {
        if theta <= 0.0 {
            return Err("Theta must be positive");
        }
        if sigma <= 0.0 {
            return Err("Sigma must be positive");
        }
        Ok(Self { theta, mu, sigma })
    }

    /// Theoretical stationary variance: σ² / (2θ)
    pub fn stationary_variance(&self) -> f64 {
        self.sigma * self.sigma / (2.0 * self.theta)
    }

    /// Theoretical stationary standard deviation.
    pub fn stationary_std(&self) -> f64 {
        self.stationary_variance().sqrt()
    }

    /// Theoretical autocorrelation at lag τ: exp(-θ|τ|)
    pub fn autocorrelation(&self, tau: f64) -> f64 {
        (-self.theta * tau.abs()).exp()
    }

    /// Theoretical stationary probability density at x.
    pub fn stationary_pdf(&self, x: f64) -> f64 {
        let var = self.stationary_variance();
        let coeff = 1.0 / (2.0 * PI * var).sqrt();
        coeff * (-(x - self.mu).powi(2) / (2.0 * var)).exp()
    }
}

/// Result of simulating a Gauss-Markov process.
#[derive(Debug, Clone)]
pub struct SimulationResult {
    pub times: Vec<f64>,
    pub values: Vec<f64>,
    pub params: GaussMarkovParams,
}

/// Simulate a Gauss-Markov process using Euler-Maruyama method.
pub fn simulate(params: GaussMarkovParams, x0: f64, t_start: f64, t_end: f64, dt: f64, rng: &mut impl FnMut() -> f64) -> SimulationResult {
    let n = ((t_end - t_start) / dt).ceil() as usize;
    let mut times = Vec::with_capacity(n + 1);
    let mut values = Vec::with_capacity(n + 1);

    let mut x = x0;
    let mut t = t_start;
    let sqrt_dt = dt.sqrt();

    times.push(t);
    values.push(x);

    while t < t_end - 1e-12 {
        let dW = sqrt_dt * standard_normal(rng);
        x += -params.theta * (x - params.mu) * dt + params.sigma * dW;
        t += dt;
        times.push(t);
        values.push(x);
    }

    SimulationResult { times, values, params }
}

/// Estimate θ from sample autocorrelation at lag dt:
/// θ ≈ -ln(ρ(dt)) / dt
pub fn estimate_theta(values: &[f64], dt: f64) -> f64 {
    let rho = sample_autocorrelation(values, 1);
    if rho <= 0.0 || rho >= 1.0 {
        return 0.0;
    }
    -rho.ln() / dt
}

/// Estimate μ as sample mean.
pub fn estimate_mu(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Estimate σ² from stationary variance: σ² = 2θ·Var(X)
pub fn estimate_sigma(values: &[f64], theta: f64) -> f64 {
    let mu = estimate_mu(values);
    let var = values.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / values.len() as f64;
    (2.0 * theta * var).sqrt()
}

/// Compute sample autocorrelation at given lag (in steps).
fn sample_autocorrelation(values: &[f64], lag: usize) -> f64 {
    let n = values.len();
    if lag >= n { return 0.0; }
    let mean = values.iter().sum::<f64>() / n as f64;
    let var: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    if var < 1e-16 { return 0.0; }
    let cov: f64 = (0..n - lag).map(|i| (values[i] - mean) * (values[i + lag] - mean)).sum::<f64>() / n as f64;
    cov / var
}

/// Box-Muller transform: generate standard normal from uniform RNG.
fn standard_normal(rng: &mut impl FnMut() -> f64) -> f64 {
    let u1 = rng();
    let u2 = rng();
    let u1 = if u1 < 1e-10 { 1e-10 } else { u1 };
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_params_validation() {
        assert!(GaussMarkovParams::new(1.0, 0.0, 1.0).is_ok());
        assert!(GaussMarkovParams::new(-1.0, 0.0, 1.0).is_err());
        assert!(GaussMarkovParams::new(1.0, 0.0, -1.0).is_err());
    }

    #[test]
    fn test_stationary_stats() {
        let p = GaussMarkovParams::new(2.0, 5.0, 3.0).unwrap();
        assert!((p.stationary_variance() - 9.0 / 4.0).abs() < 1e-10);
        assert!((p.autocorrelation(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_simulation_length() {
        let p = GaussMarkovParams::new(1.0, 0.0, 1.0).unwrap();
        let mut rng = || { ((std::time::SystemTime::now().elapsed().unwrap_or_default().subsec_nanos() as u64 ^ std::hint::black_box(0)) as f64) % 1.0 };
        let result = simulate(p, 0.0, 0.0, 1.0, 0.01, &mut || {
            let x = rng();
            x.max(1e-10).min(1.0 - 1e-10)
        });
        assert_eq!(result.times.len(), result.values.len());
        assert!(result.times.len() > 90);
    }
}
