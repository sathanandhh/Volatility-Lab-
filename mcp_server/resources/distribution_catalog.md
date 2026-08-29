# Innovation Distribution Catalog

Each distribution models the standardized residuals ε_t / σ_t. The
choice affects VaR/ES tail behavior.

## Normal (Gaussian)
- Skewness: 0, Excess kurtosis: 0
- Parameters: none
- Use when: residuals pass JB/SW (rare in finance)
- Tail: thin — underestimates extremes

## Student-t
- Skewness: 0, Excess kurtosis: > 0 (controlled by ν)
- Parameters: ν (degrees of freedom)
- Use when: residuals fail normality but are symmetric
- Tail: heavier than normal; ν typically 4-10 for equity returns
- VaR adjustment: t_ν^{-1}(α) × √((ν-2)/ν)

## Generalized Error Distribution (GED)
- Skewness: 0, Kurtosis: parameter-controlled
- Parameters: ν (shape)
- Use when: tails heavier or lighter than Student-t
- Tail: ν < 2 heavier than normal, ν > 2 lighter

## Skew-Student-t (skewt)
- Skewness: ≠ 0, Excess kurtosis: > 0
- Parameters: ν (df), λ (skew)
- Use when: residuals are both skewed and heavy-tailed
- Tail: asymmetric — important for negatively skewed equity returns

## Johnson SU (JSU)
- Skewness: parameter-controlled
- Parameters: γ (shape tail), δ (shape), ξ (location), λ (scale)
- Use when: very heavy tails with skew
- Tail: more flexible than skew-t; can model extreme skew

## Generalized Hyperbolic (GHD)
- Five-parameter family; includes many others as special cases
- Parameters: α, β, μ, δ, λ
- Use when: maximum flexibility; NIG is a special case
- Tail: very flexible; computationally heavier

## Selection rule

Run `optimize.distribution` — the optimizer fits each candidate and
picks the lowest AIC. The pre-flight normality check provides the
initial hint: if blocked, Normal will almost always lose.

## Distribution → VaR method mapping

| Distribution | VaR formula |
|---|---|
| Normal | μ + σ Φ^{-1}(α) |
| Student-t | μ + σ t_ν^{-1}(α) √((ν-2)/ν) |
| GED | μ + σ GED^{-1}(α) |
| Skew-t | μ + σ SkewT^{-1}(α) |
| JSU | μ + σ JSU^{-1}(α) |
