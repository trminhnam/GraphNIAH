"""Power-law analysis of a discrete degree distribution.

Follows Clauset, Shalizi, and Newman (2009), "Power-law distributions in empirical
data", SIAM Review 51(4). A distribution is given as counts: ``degrees`` (sorted
positive integers) and ``counts`` (papers with each degree). Only degrees
a >= ``minimum`` enter a fit, so every law below is normalized over a >= minimum.

- fit_power_law: maximum-likelihood exponent of P(a) = a^-gamma / zeta(gamma, minimum).
- goodness_of_fit: KS distance and its bootstrap p-value (plausible if p > 0.1).
- compare_alternatives: log-likelihood ratio tests against a log-normal, an
  exponential, and a power law with exponential cutoff.
- analyze_power_law: all of the above, a scan over the minimum degree, and the
  minimum that makes the fit closest to the data (smallest KS distance).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import erfc, logsumexp, zeta
from scipy.stats import chi2

# Synthetic samples use exact probabilities up to this many degrees above the
# minimum, and the continuous approximation of Clauset et al. (Appendix D) beyond.
EXACT_SAMPLING_DEGREES = 10**4
# The alternatives are normalized over minimum <= a <= 100 x the largest observed
# degree (between 10^5 and 10^7); the cutoff rate is at least 10 / that end, so the
# left-out tail of the power law with cutoff is below e^-10 of its mass.
GRID_END_FACTOR = 100
SMALLEST_GRID_END = 10**5
# An alternative whose log-probabilities differ from the power law's by less than
# this at every observed degree has converged to the power law (a log-normal does
# as sigma grows); the two then cannot be told apart, so no test is reported.
SAME_FIT_TOLERANCE = 1e-2
# p < SIGNIFICANCE rejects the power law (goodness of fit) or decides a
# likelihood-ratio test, as in Clauset, Shalizi, and Newman (2009).
SIGNIFICANCE = 0.1
# The minimum-degree scan stops when fewer than SCAN_MIN_PAPERS papers, or less than
# SCAN_MIN_SHARE of the papers at its first minimum, remain in the tail.
SCAN_MIN_PAPERS = 50
SCAN_MIN_SHARE = 0.01


def tail(
    degrees: np.ndarray, counts: np.ndarray, minimum: int
) -> tuple[np.ndarray, np.ndarray]:
    """Degrees >= minimum that have at least one paper, and their counts."""
    kept = (degrees >= minimum) & (counts > 0)
    return degrees[kept], counts[kept]


def fit_power_law(support: np.ndarray, counts: np.ndarray, minimum: int) -> float:
    """Exponent gamma of the discrete power law P(a) = a^-gamma / zeta(gamma, minimum).

    Fitted by maximum likelihood to the counts of degrees a >= minimum. The
    Hurwitz zeta function normalizes the law over a >= minimum.
    """
    kept = support >= minimum
    total = counts[kept].sum()
    log_sum = (counts[kept] * np.log(support[kept])).sum()
    result = minimize_scalar(
        lambda gamma: total * np.log(zeta(gamma, minimum)) + gamma * log_sum,
        bounds=(1.01, 10.0),
        method="bounded",
    )
    return float(result.x)


def standard_error(gamma: float, papers: float, minimum: int) -> float:
    """Standard error of gamma from the Fisher information n * (log zeta)''."""
    step = 1e-4
    curvature = (
        np.log(zeta(gamma + step, minimum))
        - 2 * np.log(zeta(gamma, minimum))
        + np.log(zeta(gamma - step, minimum))
    ) / step**2
    return float(1 / np.sqrt(papers * curvature))


def ks_distance(
    degrees: np.ndarray, counts: np.ndarray, gamma: float, minimum: int
) -> float:
    """Largest gap between the empirical CDF and the fitted power-law CDF.

    Between two observed degrees the empirical CDF is flat and the fitted CDF
    increases, so the largest gap over all integers a >= minimum lies at an
    observed degree or just before one; only those points are evaluated.
    """

    def model(values: np.ndarray) -> np.ndarray:
        return 1 - zeta(gamma, values + 1) / zeta(gamma, minimum)

    empirical = np.cumsum(counts) / counts.sum()
    previous = np.concatenate([[0.0], empirical[:-1]])
    before = degrees - 1
    valid = before >= minimum
    return float(
        max(
            np.abs(empirical - model(degrees)).max(),
            np.abs(previous[valid] - model(before[valid])).max(initial=0.0),
        )
    )


def power_law_sampler(gamma: float, minimum: int):
    """Function (rng, papers) -> (degrees, counts) drawing papers from the law.

    Degrees up to minimum + EXACT_SAMPLING_DEGREES use their exact probabilities;
    each draw beyond uses x = floor((start - 1/2) (1 - u)^(-1 / (gamma - 1)) + 1/2)
    (Clauset, Shalizi, and Newman 2009, Appendix D), which is very accurate there.
    """
    grid = np.arange(minimum, minimum + EXACT_SAMPLING_DEGREES + 1)
    normalizer = zeta(gamma, minimum)
    probabilities = np.append(
        grid.astype(float) ** -gamma / normalizer,
        zeta(gamma, grid[-1] + 1) / normalizer,
    )
    probabilities /= probabilities.sum()
    start = grid[-1] + 1

    def sample(rng: np.random.Generator, papers: int) -> tuple[np.ndarray, np.ndarray]:
        counts = rng.multinomial(papers, probabilities)
        kept = counts[:-1] > 0
        degrees, kept_counts = grid[kept], counts[:-1][kept]
        if counts[-1]:
            draws = (start - 0.5) * (1 - rng.random(counts[-1])) ** (-1 / (gamma - 1))
            far, far_counts = np.unique(
                np.floor(np.minimum(draws + 0.5, 1e12)).astype(np.int64),
                return_counts=True,
            )
            degrees = np.concatenate([degrees, far])
            kept_counts = np.concatenate([kept_counts, far_counts])
        return degrees, kept_counts

    return sample


def goodness_of_fit(
    degrees: np.ndarray,
    counts: np.ndarray,
    minimum: int,
    repeats: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """KS distance of the fit and its parametric-bootstrap p-value.

    Each synthetic sample has the same number of papers, drawn from the fitted
    law; it is refitted and its KS distance computed. The p-value is the share of
    synthetic distances at least as large as the observed one. This treats
    papers as independent draws, as in Clauset, Shalizi, and Newman (2009).
    """
    gamma = fit_power_law(degrees, counts, minimum)
    observed = ks_distance(degrees, counts, gamma, minimum)
    sample = power_law_sampler(gamma, minimum)
    papers = int(counts.sum())
    synthetic = np.empty(repeats)
    for index in range(repeats):
        synthetic_degrees, synthetic_counts = sample(rng, papers)
        refit = fit_power_law(synthetic_degrees, synthetic_counts, minimum)
        synthetic[index] = ks_distance(
            synthetic_degrees, synthetic_counts, refit, minimum
        )
    p_value = float(np.mean(synthetic >= observed))
    return {
        "ks_distance": observed,
        "p_value": p_value,
        "bootstrap_repeats": repeats,
        "synthetic_ks_median": float(np.median(synthetic)),
        "synthetic_ks_95th_percentile": float(np.percentile(synthetic, 95)),
        "power_law_plausible": p_value > SIGNIFICANCE,
    }


def alternative_grid(degrees: np.ndarray, minimum: int) -> np.ndarray:
    """Degrees over which the alternatives are normalized, far beyond the data."""
    end = max(SMALLEST_GRID_END, min(GRID_END_FACTOR * int(degrees.max()), 10**7))
    return np.arange(minimum, end + 1)


def power_law_log_pmf(degrees: np.ndarray, gamma: float, minimum: int) -> np.ndarray:
    return -gamma * np.log(degrees) - np.log(zeta(gamma, minimum))


def fit_exponential(
    degrees: np.ndarray, counts: np.ndarray, minimum: int
) -> tuple[dict[str, float], np.ndarray]:
    """Discrete exponential P(a) = (1 - q) q^(a - minimum); closed-form MLE of q."""
    excess = (counts * (degrees - minimum)).sum() / counts.sum()
    q = excess / (1 + excess)
    log_pmf = np.log1p(-q) + (degrees - minimum) * np.log(q)
    return {"rate": float(-np.log(q))}, log_pmf


def fit_log_normal(
    degrees: np.ndarray, counts: np.ndarray, minimum: int
) -> tuple[dict[str, float | None], np.ndarray]:
    """Discrete log-normal: P(a) proportional to exp(-(ln a - mu)^2 / (2 sigma^2)) / a.

    Fitted in the form log P(a) = slope ln a + curvature (ln a)^2 + const, with
    curvature = -1 / (2 sigma^2) <= 0 and slope = mu / sigma^2 - 1. The negative
    log-likelihood is then convex. Curvature 0 is the limit sigma -> infinity, a
    power law; mu and sigma are reported as None there.
    """
    grid_log = np.log(alternative_grid(degrees, minimum))
    grid_square = grid_log**2
    log_degrees = np.log(degrees)
    total = counts.sum()
    first, second = (counts * log_degrees).sum(), (counts * log_degrees**2).sum()

    def negative_log_likelihood(values: np.ndarray) -> tuple[float, np.ndarray]:
        slope, curvature = values
        log_weights = slope * grid_log + curvature * grid_square
        normalizer = logsumexp(log_weights)
        weights = np.exp(log_weights - normalizer)
        value = -(slope * first + curvature * second) + total * normalizer
        gradient = np.array(
            [
                -first + total * (weights * grid_log).sum(),
                -second + total * (weights * grid_square).sum(),
            ]
        )
        return value, gradient

    gamma = fit_power_law(degrees, counts, minimum)
    result = minimize(
        negative_log_likelihood,
        x0=[-gamma, -0.01],
        jac=True,
        method="L-BFGS-B",
        bounds=[(-50.0, 50.0), (-50.0, 0.0)],
    )
    slope, curvature = (float(value) for value in result.x)
    log_pmf = slope * log_degrees + curvature * log_degrees**2 - logsumexp(
        slope * grid_log + curvature * grid_square
    )
    if curvature < 0:
        sigma = float(np.sqrt(-1 / (2 * curvature)))
        parameters = {"mu": sigma**2 * (slope + 1), "sigma": sigma}
    else:
        parameters = {"mu": None, "sigma": None}
    return {**parameters, "slope": slope, "curvature": curvature}, log_pmf

def fit_power_law_with_cutoff(
    degrees: np.ndarray, counts: np.ndarray, minimum: int, gamma: float
) -> tuple[dict[str, float], np.ndarray]:
    """P(a) proportional to a^-gamma e^(-rate a); the power law is the limit rate -> 0.

    The negative log-likelihood is convex in (gamma, rate). The rate is at least
    10 / (end of the normalization grid), so the left-out tail stays below e^-10.
    """
    grid = alternative_grid(degrees, minimum)
    grid_log = np.log(grid)
    log_degrees = np.log(degrees)
    total = counts.sum()
    first, second = (counts * log_degrees).sum(), (counts * degrees).sum()

    def negative_log_likelihood(values: np.ndarray) -> tuple[float, np.ndarray]:
        exponent, rate = values
        log_weights = -exponent * grid_log - rate * grid
        normalizer = logsumexp(log_weights)
        weights = np.exp(log_weights - normalizer)
        value = exponent * first + rate * second + total * normalizer
        gradient = np.array(
            [
                first - total * (weights * grid_log).sum(),
                second - total * (weights * grid).sum(),
            ]
        )
        return value, gradient

    result = minimize(
        negative_log_likelihood,
        x0=[gamma, max(10 / grid[-1], 1e-3)],
        jac=True,
        method="L-BFGS-B",
        bounds=[(-10.0, 10.0), (10 / grid[-1], 10.0)],
    )
    exponent, rate = (float(value) for value in result.x)
    normalizer = logsumexp(-exponent * grid_log - rate * grid)
    log_pmf = -exponent * log_degrees - rate * degrees - normalizer
    return {"gamma": exponent, "rate": rate}, log_pmf

def vuong_test(
    counts: np.ndarray, power_law: np.ndarray, alternative: np.ndarray
) -> dict[str, float]:
    """Log-likelihood ratio R (positive favors the power law) and its p-value.

    The p-value is two-sided under the normal approximation of Vuong (1989), as
    in Section 5 of Clauset, Shalizi, and Newman (2009); each paper is one term.
    """
    difference = power_law - alternative
    papers = counts.sum()
    ratio = float((counts * difference).sum())
    variance = (counts * (difference - ratio / papers) ** 2).sum() / papers
    if variance == 0:
        return {"log_likelihood_ratio": ratio, "normalized_ratio": 0.0, "p_value": 1.0}
    normalized = ratio / np.sqrt(papers * variance)
    return {
        "log_likelihood_ratio": ratio,
        "normalized_ratio": float(normalized),
        "p_value": float(erfc(abs(normalized) / np.sqrt(2))),
    }


def favored(ratio: float, p_value: float, alternative: str) -> str:
    if p_value >= SIGNIFICANCE:
        return "neither (not significant)"
    return "power law" if ratio > 0 else alternative


def compare_alternatives(
    degrees: np.ndarray, counts: np.ndarray, minimum: int, gamma: float
) -> dict[str, Any]:
    """Likelihood-ratio tests of the fitted power law against three alternatives.

    Log-normal and exponential are not nested in the power law, so they use the
    Vuong test. The power law with cutoff contains the power law (rate -> 0), so
    it uses the nested test: 2 x (log-likelihood gain) against chi-squared with
    one degree of freedom; a small p-value means the cutoff fits significantly
    better.
    """
    power_law = power_law_log_pmf(degrees, gamma, minimum)
    results = {}
    for name, (parameters, log_pmf) in (
        ("log_normal", fit_log_normal(degrees, counts, minimum)),
        ("exponential", fit_exponential(degrees, counts, minimum)),
    ):
        largest_gap = float(np.abs(log_pmf - power_law).max())
        if largest_gap < SAME_FIT_TOLERANCE:
            # The Vuong statistic divides by the spread of the differences, which
            # is then close to zero, so its p-value would be meaningless.
            results[name] = {
                "parameters": parameters,
                "largest_log_probability_gap": largest_gap,
                "log_likelihood_ratio": float((counts * (power_law - log_pmf)).sum()),
                "normalized_ratio": None,
                "p_value": None,
                "favored": "neither (the fit converges to the power law)",
            }
            continue
        test = vuong_test(counts, power_law, log_pmf)
        results[name] = {
            "parameters": parameters,
            "largest_log_probability_gap": largest_gap,
            **test,
            "favored": favored(
                test["log_likelihood_ratio"], test["p_value"], name.replace("_", "-")
            ),
        }
    parameters, log_pmf = fit_power_law_with_cutoff(degrees, counts, minimum, gamma)
    gain = max(float((counts * (log_pmf - power_law)).sum()), 0.0)
    p_value = float(chi2.sf(2 * gain, df=1))
    results["power_law_with_cutoff"] = {
        "parameters": parameters,
        "log_likelihood_ratio": -gain,
        "p_value": p_value,
        "favored": "power law with cutoff"
        if p_value < SIGNIFICANCE
        else "neither (not significant)",
    }
    return results


def scan_minimum(
    degrees: np.ndarray, counts: np.ndarray, start: int
) -> list[dict[str, float]]:
    """Fit and KS distance for each minimum degree from start, while the tail has
    at least SCAN_MIN_PAPERS papers, SCAN_MIN_SHARE of the papers at start, and
    two distinct degrees."""
    rows = []
    smallest = max(SCAN_MIN_PAPERS, SCAN_MIN_SHARE * tail(degrees, counts, start)[1].sum())
    for minimum in range(start, int(degrees.max()) + 1):
        kept_degrees, kept_counts = tail(degrees, counts, minimum)
        if kept_counts.sum() < smallest or len(kept_degrees) < 2:
            break
        gamma = fit_power_law(kept_degrees, kept_counts, minimum)
        rows.append(
            {
                "minimum_degree": minimum,
                "papers": int(kept_counts.sum()),
                "gamma": gamma,
                "ks_distance": ks_distance(kept_degrees, kept_counts, gamma, minimum),
            }
        )
    return rows


def fit_at_minimum(
    degrees: np.ndarray,
    counts: np.ndarray,
    minimum: int,
    repeats: int,
    rng: np.random.Generator,
    replicates: np.ndarray | None = None,
) -> dict[str, Any]:
    """Every result for one minimum degree; ``replicates`` (one row of counts per
    bootstrap resample, aligned with ``degrees``) gives the 95% interval of gamma."""
    kept_degrees, kept_counts = tail(degrees, counts, minimum)
    papers = int(kept_counts.sum())
    if papers < 10 or len(kept_degrees) < 3:
        return {
            "minimum_degree": minimum,
            "papers": papers,
            "skipped": "fewer than 10 papers or 3 distinct degrees above the minimum",
        }
    gamma = fit_power_law(kept_degrees, kept_counts, minimum)
    result: dict[str, Any] = {
        "minimum_degree": minimum,
        "papers": papers,
        "distinct_degrees": len(kept_degrees),
        "largest_degree": int(kept_degrees.max()),
        "gamma": gamma,
        "gamma_standard_error": standard_error(gamma, papers, minimum),
    }
    if replicates is not None:
        kept = degrees >= minimum
        values = np.array(
            [
                fit_power_law(degrees[kept], row[kept], minimum)
                for row in replicates
                if row[kept].sum() > 0
            ]
        )
        low, high = np.percentile(values, [2.5, 97.5])
        result["gamma_95_interval"] = [float(low), float(high)]
        result["interval_repeats"] = len(values)
    result["goodness_of_fit"] = goodness_of_fit(
        kept_degrees, kept_counts, minimum, repeats, rng
    )
    result["alternatives"] = compare_alternatives(
        kept_degrees, kept_counts, minimum, gamma
    )
    return result


def analyze_power_law(
    degrees: np.ndarray,
    counts: np.ndarray,
    minimum: int,
    repeats: int,
    rng: np.random.Generator,
    replicates: np.ndarray | None = None,
) -> dict[str, Any]:
    """The fit at ``minimum`` and at the KS-optimal minimum, and the scan between.

    The KS-optimal minimum is the one whose fit has the smallest KS distance
    (Clauset, Shalizi, and Newman 2009, Section 3.3); its p-value keeps that
    minimum fixed, so it is optimistic compared with refitting it per sample.
    """
    scan = scan_minimum(degrees, counts, minimum)
    result: dict[str, Any] = {
        "at_minimum": fit_at_minimum(degrees, counts, minimum, repeats, rng, replicates),
        "minimum_scan": scan,
    }
    if scan:
        best = min(scan, key=lambda row: row["ks_distance"])["minimum_degree"]
        result["ks_optimal_minimum"] = best
        if best != minimum:
            result["at_ks_optimal_minimum"] = fit_at_minimum(
                degrees, counts, best, repeats, rng, replicates
            )
    return result
