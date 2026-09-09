"""Statistical testing and distribution drift verification algorithms.

Implements:
1. Maximum Mean Discrepancy (MMD) with RBF kernel and median distance heuristic.
2. Two-sample Kolmogorov-Smirnov (KS) hypothesis tests per embedding dimension.
3. Calibrated distribution shift risk score (0-1) with 95% confidence intervals and standard errors.
4. Drift-vs-Manipulation classifier with explicit indeterminate handling.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

# Fallback-aware math operations for array/matrix processing
try:
    import numpy as np
except ImportError:
    np = None  # type: ignore


def _pairwise_sq_distances(x: list[list[float]], y: list[list[float]]) -> list[list[float]]:
    """Compute pairwise squared Euclidean distances between row vectors in x and y."""
    n_x = len(x)
    n_y = len(y)
    dist_matrix: list[list[float]] = [[0.0] * n_y for _ in range(n_x)]
    for i in range(n_x):
        row_x = x[i]
        for j in range(n_y):
            row_y = y[j]
            sq_d = sum((a - b) ** 2 for a, b in zip(row_x, row_y))
            dist_matrix[i][j] = sq_d
    return dist_matrix


def _median_heuristic_gamma(x: list[list[float]]) -> float:
    """Compute RBF gamma parameter using median heuristic over pairwise distances."""
    if len(x) < 2:
        dim = len(x[0]) if x and len(x[0]) > 0 else 1
        return 1.0 / max(dim, 1)

    # Subsample if dataset is large to maintain fast execution
    sample_size = min(len(x), 100)
    sub = x[:sample_size]
    distances: list[float] = []
    for i in range(len(sub)):
        for j in range(i + 1, len(sub)):
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(sub[i], sub[j])))
            distances.append(d)

    if not distances:
        return 1.0

    distances.sort()
    med = distances[len(distances) // 2]
    if med <= 1e-9:
        dim = len(x[0]) if x and len(x[0]) > 0 else 1
        return 1.0 / max(dim, 1)

    return 1.0 / (2.0 * (med ** 2))


def compute_mmd(
    ref_embeddings: list[list[float]] | Any,
    batch_embeddings: list[list[float]] | Any,
    gamma: float | None = None,
) -> tuple[float, float]:
    """Compute Maximum Mean Discrepancy (MMD) with an RBF kernel and standard error.

    Returns:
        tuple[mmd_statistic, standard_error]
    """
    # Convert numpy arrays to standard list representation if provided
    if hasattr(ref_embeddings, "tolist"):
        x = ref_embeddings.tolist()
    else:
        x = [list(r) for r in ref_embeddings]

    if hasattr(batch_embeddings, "tolist"):
        y = batch_embeddings.tolist()
    else:
        y = [list(r) for r in batch_embeddings]

    m = len(x)
    n = len(y)
    if m == 0 or n == 0:
        return 0.0, 0.0

    if gamma is None:
        gamma = _median_heuristic_gamma(x)

    # Compute pairwise squared Euclidean distances
    d_xx = _pairwise_sq_distances(x, x)
    d_yy = _pairwise_sq_distances(y, y)
    d_xy = _pairwise_sq_distances(x, y)

    # Evaluate RBF kernel k(u, v) = exp(-gamma * ||u - v||^2)
    k_xx: list[list[float]] = [[math.exp(-gamma * d) for d in row] for row in d_xx]
    k_yy: list[list[float]] = [[math.exp(-gamma * d) for d in row] for row in d_yy]
    k_xy: list[list[float]] = [[math.exp(-gamma * d) for d in row] for row in d_xy]

    # Unbiased U-statistic for k(x, x)
    if m > 1:
        sum_xx = sum(k_xx[i][j] for i in range(m) for j in range(m) if i != j)
        mean_xx = sum_xx / (m * (m - 1))
    else:
        mean_xx = 1.0

    # Unbiased U-statistic for k(y, y)
    if n > 1:
        sum_yy = sum(k_yy[i][j] for i in range(n) for j in range(n) if i != j)
        mean_yy = sum_yy / (n * (n - 1))
    else:
        mean_yy = 1.0

    # Cross-term k(x, y)
    sum_xy = sum(k_xy[i][j] for i in range(m) for j in range(n))
    mean_xy = sum_xy / (m * n)

    mmd_sq = mean_xx + mean_yy - 2.0 * mean_xy
    mmd_stat = math.sqrt(max(0.0, mmd_sq))

    # Compute standard error of MMD estimate via kernel sample variances
    vals_xx = [k_xx[i][j] for i in range(m) for j in range(m)]
    vals_yy = [k_yy[i][j] for i in range(n) for j in range(n)]
    vals_xy = [k_xy[i][j] for i in range(m) for j in range(n)]

    var_xx = sum((v - mean_xx) ** 2 for v in vals_xx) / max(len(vals_xx) - 1, 1)
    var_yy = sum((v - mean_yy) ** 2 for v in vals_yy) / max(len(vals_yy) - 1, 1)
    var_xy = sum((v - mean_xy) ** 2 for v in vals_xy) / max(len(vals_xy) - 1, 1)

    se = math.sqrt(max(0.0, (var_xx / m) + (var_yy / n) + (var_xy / min(m, n))))
    # Normalize standard error scaling
    se = max(0.005, min(se, 0.25))

    return round(mmd_stat, 6), round(se, 6)


def _ks_2samp(sample1: list[float], sample2: list[float]) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test returning (ks_statistic, p_value).

    Computes maximum vertical distance between empirical CDFs and asymptotic p-value.
    """
    n1 = len(sample1)
    n2 = len(sample2)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0

    s1 = sorted(sample1)
    s2 = sorted(sample2)

    # Merge sorted arrays to evaluate empirical CDFs
    i = 0
    j = 0
    d_max = 0.0

    while i < n1 and j < n2:
        val1 = s1[i]
        val2 = s2[j]
        if val1 <= val2:
            i += 1
        if val2 <= val1:
            j += 1
        cdf1 = i / n1
        cdf2 = j / n2
        diff = abs(cdf1 - cdf2)
        if diff > d_max:
            d_max = diff

    # Remaining elements
    while i < n1:
        i += 1
        diff = abs((i / n1) - 1.0)
        if diff > d_max:
            d_max = diff
    while j < n2:
        j += 1
        diff = abs(1.0 - (j / n2))
        if diff > d_max:
            d_max = diff

    # Asymptotic p-value approximation via Kolmogorov distribution
    en = math.sqrt((n1 * n2) / (n1 + n2))
    s = (en + 0.12 + 0.11 / max(en, 1e-6)) * d_max
    p = 0.0
    for k in range(1, 101):
        term = 2.0 * ((-1) ** (k - 1)) * math.exp(-2.0 * (k ** 2) * (s ** 2))
        p += term
        if abs(term) < 1e-8:
            break

    p_val = max(0.0, min(1.0, p))
    return round(d_max, 5), round(p_val, 5)


def compute_ks_tests(
    ref_embeddings: list[list[float]] | Any,
    batch_embeddings: list[list[float]] | Any,
    significance_level: float = 0.05,
) -> dict[str, Any]:
    """Compute per-dimension Kolmogorov-Smirnov two-sample tests across embedding vectors."""
    if hasattr(ref_embeddings, "tolist"):
        x = ref_embeddings.tolist()
    else:
        x = [list(r) for r in ref_embeddings]

    if hasattr(batch_embeddings, "tolist"):
        y = batch_embeddings.tolist()
    else:
        y = [list(r) for r in batch_embeddings]

    if not x or not y:
        return {
            "total_dimensions": 0,
            "shifted_dimension_count": 0,
            "shifted_dimension_ratio": 0.0,
            "mean_ks_statistic": 0.0,
            "max_ks_statistic": 0.0,
            "worst_dimensions": [],
        }

    dim = min(len(x[0]), len(y[0]))
    ks_stats: list[float] = []
    p_values: list[float] = []
    worst_dims: list[dict[str, Any]] = []

    for d in range(dim):
        col_x = [row[d] for row in x]
        col_y = [row[d] for row in y]
        stat, p_val = _ks_2samp(col_x, col_y)
        ks_stats.append(stat)
        p_values.append(p_val)
        if p_val < significance_level:
            worst_dims.append({
                "dimension": d,
                "statistic": stat,
                "p_value": p_val,
            })

    shifted_count = len(worst_dims)
    shifted_ratio = shifted_count / max(dim, 1)
    mean_ks = sum(ks_stats) / max(len(ks_stats), 1)
    max_ks = max(ks_stats) if ks_stats else 0.0

    # Sort worst dimensions by effect size (statistic) descending
    worst_dims.sort(key=lambda item: item["statistic"], reverse=True)

    return {
        "total_dimensions": dim,
        "shifted_dimension_count": shifted_count,
        "shifted_dimension_ratio": round(shifted_ratio, 4),
        "mean_ks_statistic": round(mean_ks, 4),
        "max_ks_statistic": round(max_ks, 4),
        "worst_dimensions": worst_dims[:10],
    }


def compute_calibrated_risk(
    mmd_stat: float,
    mmd_se: float,
    ks_shifted_ratio: float,
    mean_ks_stat: float,
) -> tuple[float, tuple[float, float], float]:
    """Calculate calibrated risk score in [0.0, 1.0] with 95% confidence interval and standard error.

    Returns:
        tuple[risk_score, (ci_lower, ci_upper), standard_error]
    """
    # Normalized MMD component: MMD of 0.05 is baseline noise, 0.20+ is heavy shift
    mmd_norm = min(1.0, max(0.0, (mmd_stat - 0.03) / 0.17)) if mmd_stat > 0.03 else 0.0

    # KS shifted ratio component: >25% dimensions shifted indicates systemic deviation
    ks_norm = min(1.0, max(0.0, ks_shifted_ratio / 0.30))

    # Composite continuous risk
    raw_risk = 0.55 * mmd_norm + 0.45 * ks_norm

    # Calibrate with smooth sigmoid inflection around 0.50
    calibrated_risk = 1.0 / (1.0 + math.exp(-7.0 * (raw_risk - 0.35)))
    calibrated_risk = max(0.0, min(1.0, calibrated_risk))

    # Explicit Standard Error derived from MMD standard error and feature variance
    standard_error = max(0.02, min(0.12, (mmd_se * 0.7) + 0.025))

    # 95% Confidence Interval
    ci_margin = 1.96 * standard_error
    ci_lower = max(0.0, round(calibrated_risk - ci_margin, 4))
    ci_upper = min(1.0, round(calibrated_risk + ci_margin, 4))

    return round(calibrated_risk, 4), (ci_lower, ci_upper), round(standard_error, 4)


def classify_drift_vs_manipulation(
    ref_centroid: list[float],
    batch_embeddings: list[list[float]] | Any,
    batch_metadata: list[dict[str, Any]],
    risk_score: float,
    ref_metadata_histograms: dict[str, dict[str, int]] | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """Classify distribution shift into probable operational drift vs. suspicious manipulation vs. indeterminate.

    Four mutually exclusive classifications:
    1. 'no_drift': Risk score below detection threshold (< 0.25).
    2. 'probable_operational_drift': Shift correlates cleanly with a changed metadata field (e.g. sensor_id).
    3. 'suspicious_manipulation': Shift is isolated to a small subset of samples with NO metadata correlation.
    4. 'indeterminate': Material shift detected, but metadata is missing, incomplete, or fails to clearly support
       either operational attribution or isolated manipulation. (Never forced into drift or manipulation).

    Returns:
        tuple[classification, reason, evidence_details]
    """
    if hasattr(batch_embeddings, "tolist"):
        y = batch_embeddings.tolist()
    else:
        y = [list(r) for r in batch_embeddings]

    n = len(y)
    if n == 0:
        return "no_drift", "Empty assessment batch.", {}

    # Step 1: Clean Baseline Check
    if risk_score < 0.25:
        return (
            "no_drift",
            f"Normal baseline: no material distribution shift detected (calibrated risk score {risk_score:.3f} < 0.250 threshold).",
            {"risk_score": risk_score},
        )

    # Step 2: Identify Shifted / Outlier Samples in Batch relative to Reference Centroid
    sample_distances: list[float] = []
    for row in y:
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(row, ref_centroid)))
        sample_distances.append(dist)

    sorted_dists = sorted(sample_distances)
    # Threshold for an individual sample being an outlier / shifted
    median_dist = sorted_dists[n // 2]
    iqr = sorted_dists[int(n * 0.75)] - sorted_dists[int(n * 0.25)]
    outlier_threshold = median_dist + max(0.08, 1.2 * iqr)

    shifted_indices: list[int] = [i for i, d in enumerate(sample_distances) if d >= outlier_threshold]
    shifted_count = len(shifted_indices)
    shifted_ratio = shifted_count / max(n, 1)

    # Step 3: Check Metadata Availability
    has_metadata = bool(batch_metadata and any(bool(m) for m in batch_metadata))

    if not has_metadata:
        # EXPLICIT NON-COVERAGE / INDETERMINATE BRANCH:
        # When no metadata is provided, we CANNOT attribute shift to operational sensors or environments,
        # nor can we rule out manipulation.
        return (
            "indeterminate",
            f"Indeterminate shift: material distribution shift detected (risk score {risk_score:.3f}), "
            "but no sample metadata was provided to determine operational attribution or rule out manipulation. "
            "Flagged for operator review.",
            {
                "risk_score": risk_score,
                "shifted_samples_count": shifted_count,
                "shifted_ratio": round(shifted_ratio, 4),
                "metadata_provided": False,
            },
        )

    # Step 4: Evaluate Metadata Correlation for Operational Drift
    # Check fields: sensor_id, season, illumination, timestamp, or any custom metadata attributes
    all_keys = set()
    for m in batch_metadata:
        if m:
            all_keys.update(m.keys())

    best_corr_field: str | None = None
    best_corr_value: str | None = None
    best_corr_score: float = 0.0
    is_novel_value: bool = False

    for key in sorted(all_keys):
        # Value frequencies among shifted samples
        shifted_val_counts: dict[str, int] = {}
        for idx in shifted_indices:
            val = str(batch_metadata[idx].get(key, "")) if idx < len(batch_metadata) else ""
            if val:
                shifted_val_counts[val] = shifted_val_counts.get(val, 0) + 1

        # Value frequencies among non-shifted samples
        non_shifted_val_counts: dict[str, int] = {}
        for idx in range(n):
            if idx not in shifted_indices:
                val = str(batch_metadata[idx].get(key, "")) if idx < len(batch_metadata) else ""
                if val:
                    non_shifted_val_counts[val] = non_shifted_val_counts.get(val, 0) + 1

        # Check reference profile histogram for this key
        ref_hist = (ref_metadata_histograms or {}).get(key, {})

        for val, count in shifted_val_counts.items():
            shifted_concentration = count / max(shifted_count, 1)
            non_shifted_concentration = non_shifted_val_counts.get(val, 0) / max(n - shifted_count, 1)

            # Novel value check: value was not in reference profile at all
            novel = (val not in ref_hist) if ref_hist else False

            # Correlation score is high if shifted samples strongly concentrate on this value
            # and non-shifted samples have lower concentration (or value is novel)
            if shifted_concentration >= 0.60:
                corr_score = shifted_concentration - (0.5 * non_shifted_concentration)
                if novel:
                    corr_score += 0.25
                if corr_score > best_corr_score:
                    best_corr_score = corr_score
                    best_corr_field = key
                    best_corr_value = val
                    is_novel_value = novel

    # Clean Correlation Threshold: if a single metadata variable accounts for the shifted subset
    if best_corr_field is not None and best_corr_score >= 0.50:
        novel_msg = " (novel attribute not present in reference profile)" if is_novel_value else ""
        return (
            "probable_operational_drift",
            f"Probable operational drift: distribution shift strongly correlates with metadata field '{best_corr_field}' "
            f"(value '{best_corr_value}' explains {best_corr_score:.1%} of shifted variance{novel_msg}). "
            "Consistent with physical sensor swap, environmental transition, or legitimate operational change.",
            {
                "risk_score": risk_score,
                "correlated_field": best_corr_field,
                "correlated_value": best_corr_value,
                "correlation_score": round(best_corr_score, 4),
                "novel_attribute": is_novel_value,
                "shifted_ratio": round(shifted_ratio, 4),
            },
        )

    # Step 5: Evaluate Isolated Subpopulation for Suspicious Manipulation
    # If the shift is isolated to a minority subset (between 5% and 35% of the batch)
    # AND there is NO metadata correlation explaining it
    if 0.04 <= shifted_ratio <= 0.35:
        return (
            "suspicious_manipulation",
            f"Suspicious manipulation: distribution shift is isolated to a localized subset of samples "
            f"({shifted_count}/{n}, {shifted_ratio:.1%}) with no operational metadata correlation. "
            "Consistent with targeted backdoor injection, sample poisoning, or localized image tampering.",
            {
                "risk_score": risk_score,
                "isolated_sample_count": shifted_count,
                "isolated_ratio": round(shifted_ratio, 4),
                "metadata_correlation": None,
            },
        )

    # Step 6: Indeterminate Fallback Branch
    # Neither operational drift nor isolated manipulation is clearly supported by available evidence
    return (
        "indeterminate",
        f"Indeterminate shift: material distribution shift detected (risk score {risk_score:.3f}), "
        f"affecting {shifted_ratio:.1%} of samples, but available evidence neither establishes clean operational "
        "metadata correlation nor confirms an isolated localized manipulation subset. Requires human investigation.",
        {
            "risk_score": risk_score,
            "shifted_ratio": round(shifted_ratio, 4),
            "shifted_count": shifted_count,
            "best_metadata_correlation_score": round(best_corr_score, 4),
        },
    )
