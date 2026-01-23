# E-Commerce A/B Testing Analysis: ASOS Digital Experiments

Statistical analysis of real e-commerce experiments using Welch's t-test and multiple testing corrections to demonstrate proper false discovery rate control in large-scale experimentation.

## Dataset Context

This project analyzes the **ASOS Digital Experiments Dataset**, a publicly available research dataset released by ASOS.com in collaboration with Imperial College London. The dataset contains real A/B test results from a major global fashion retailer.

**Dataset Details:**
- **Source**: ASOS.com (global online fashion retailer)
- **Time Period**: 2019-2020  
- **Scale**: Experiments involving hundreds of thousands to millions of users
- **Publication**: Released alongside "Datasets for Online Controlled Experiments" (NeurIPS 2021)
- **DOI**: [10.17605/OSF.IO/64JSB](https://osf.io/64jsb/)

The dataset represents real business experiments that ASOS actually deployed to customers, making this analysis particularly valuable for understanding multiple testing challenges in production e-commerce environments.

## Project Overview

This project analyzes **78 real experiments** across **4 business metrics**, testing **396 experiment-variant-metric combinations** to demonstrate proper statistical methodology for controlling false discovery rates in large-scale A/B testing programs.

**Research Focus**: Comparing Benjamini-Hochberg (FDR control) vs Bonferroni (FWER control) correction methods on real industry data.

**Statistical Methods**: 
- Welch's t-test for unequal variances
- Benjamini-Hochberg procedure for False Discovery Rate control
- Bonferroni correction for Family-Wise Error Rate control

## Key Findings

### Multiple Testing Correction Impact
- **67 significant discoveries** (16.9%) using BH correction (FDR ≤ 0.05)
- **42 significant discoveries** (10.6%) using Bonferroni correction (FWER ≤ 0.05)
- **37% reduction** in discoveries when using more conservative Bonferroni vs BH

### Experiment-Level Results
- **27 experiments** (35%) showed at least one significant result
- **Largest treatment effect**: +1.28 (experiment `81761c`, variant 2, metric 4)  
- **Average effect size**: +0.056 across BH-significant results

### Cross-Metric Consistency
All four business metrics showed similar discovery rates (~16-18%), suggesting:
1. Balanced experimental impact across different KPIs
2. Consistent statistical methodology across metrics
3. No evidence of metric-specific bias in the correction procedures

## Why This Analysis Matters

**For Data Scientists**: Demonstrates the practical impact of multiple testing corrections on real business decisions. The 37% difference between BH and Bonferroni methods shows how correction choice affects experiment conclusions.

**For E-commerce Teams**: Shows realistic effect sizes and significance rates from a major retailer, providing benchmarks for experimentation programs.

**For Statisticians**: Validates multiple testing theory on production data, showing that real experiments follow expected statistical patterns.

## Visualizations

### P-value Distributions
![P-value Distributions](analysis/outputs/01_pvalue_distributions.png)

Comparison of raw vs BH-adjusted p-value distributions. The concentration of raw p-values near zero reflects real experiments with genuine treatment effects, while the BH adjustment shows proper FDR control.

### BH Decision Boundary
![BH Curve](analysis/outputs/02_bh_curve.png)

Visual representation of the Benjamini-Hochberg procedure. Points below the line represent discoveries at FDR ≤ 0.05.

### Effect vs Significance (Volcano Plot)
![Effect vs FDR](analysis/outputs/03_effect_vs_fdr.png)

Treatment effects plotted against statistical significance. Red points indicate FDR-significant discoveries, showing the relationship between effect size and detectability.

### Discoveries by Metric
![Discoveries by Metric](analysis/outputs/04_discoveries_by_metric.png)

Comparison of BH vs Bonferroni discoveries across ASOS's four business metrics, demonstrating consistent correction method impact.

### Top Performing Experiments
![Top Experiments](analysis/outputs/05_top_experiments.png)

Experiments ranked by number of significant discoveries across metrics. Multiple significant results suggest robust treatment effects.

### Treatment Effect Stability Over Time
![Top Discoveries Over Time](analysis/outputs/06_top_discoveries_over_time.png)

Temporal evolution of treatment effects for top discoveries, assessing whether effects remain consistent throughout experiment duration.

## Technical Implementation

### Dependencies
```bash
pip install -r requirements.txt
```

### Run Analysis
```bash
python analysis/multiple_testing_correction.py
```

Generates all visualizations and saves results to `analysis/outputs/welch_latest_multiple_testing.csv`.

### Testing
```bash
pytest tests/
```

## Dataset Citation

If you use this analysis or dataset, please cite:

```
Liu, C. H. B., Cardoso, A., Couturier, P., & McCoy, E. J. (2021). 
Datasets for Online Controlled Experiments. 
NeurIPS Datasets and Benchmarks Track.
```

## Limitations & Considerations

1. **Business Context**: Results reflect ASOS's specific business model, user base, and experimental practices
2. **Time Period**: Data from 2019-2020 may not reflect current e-commerce patterns
3. **Metric Anonymization**: Business metrics are anonymized, limiting interpretation of practical significance
4. **Selection Bias**: Dataset includes only completed experiments, potentially over-representing successful tests

## Future Extensions

- **Bayesian Analysis**: Apply Bayesian multiple testing procedures
- **Sequential Testing**: Analyze experiments with adaptive stopping rules  
- **Effect Size Meta-Analysis**: Pool effect sizes across similar experiment types
- **Power Analysis**: Estimate required sample sizes for different effect magnitudes
