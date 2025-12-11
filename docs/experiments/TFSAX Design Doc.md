# Design Document: SAX + Simple Trend Feature Encoding

## Overview
This document outlines a minimal hybrid symbolic representation for univariate time series data that combines **Symbolic Aggregate Approximation (SAX)** from the `pyts` library with a simple **trend-based feature**. The design aims to maintain SAX’s dimensionality reduction and symbolic interpretability while incorporating directional change information.

---

## Objectives
- **Preserve SAX’s efficiency** in representing time series as symbolic words.  
- **Capture basic trend dynamics** (upward, downward, flat) without heavy computation.  
- **Enable integration** into classification pipelines, particularly RNN-based models.

---

## Inputs and Outputs
**Inputs:**
- Time series \( T = (t_1, t_2, \ldots, t_n) \)
- Number of segments \( w \)
- Alphabet size \( \alpha \)
- Trend sensitivity threshold \( \tau \)

**Outputs:**
- \( W = (w_1, \dots, w_w) \): SAX word
- \( U = (u_1, \dots, u_w) \): Trend word
- Combined representation \( R = \{W, U\} \)

---

## System Design

### Step 1: SAX Transformation
1. Normalize \( T \) to zero mean and unit variance.
2. Use `pyts.approximation.SymbolicAggregateApproximation` to transform \( T \) into \( W \).
3. Each \( w_i \) encodes the average behavior of segment \( S^{(i)} \).

### Step 2: Trend Feature Extraction
1. Partition \( T \) into \( w \) equal segments.
2. For each segment \( S^{(i)} \), compute:
   \[
   \Delta_i = s^{(i)}_{\text{end}} - s^{(i)}_{\text{start}}
   \]
3. Discretize \(\Delta_i\) into a simple 3-symbol alphabet:
   - \( u_i = \texttt{'U'} \) if \( \Delta_i > \tau \)
   - \( u_i = \texttt{'D'} \) if \( \Delta_i < -\tau \)
   - \( u_i = \texttt{'F'} \) otherwise

4. Construct the trend word \( U = (u_1, \dots, u_w) \).

---

## Example
Given:
- \( w = 8 \), \( \alpha = 5 \), \( \tau = 0.01 \)
- Input time series \( T \) of length 128.

**Output:**
- \( W = \texttt{‘cbacdebb’} \)
- \( U = \texttt{‘UDFUUFFD’} \)

This hybrid encoding now contains symbolic mean and trend descriptors per segment.

---

## Complexity
- **Time:** \( O(n) \) (linear with respect to time series length)
- **Space:** \( O(w) \) (SAX + trend symbols)

---

## Extensions
- Use slope or regression coefficients instead of endpoint difference for more robust trends.
- Replace hard threshold \( \tau \) with adaptive quantile-based bins.
- Combine \( W \) and \( U \) into a unified symbol set for use in RNN model.

---

## Summary
This method offers a lightweight and interpretable extension to SAX by integrating local trend information. It provides a foundational step toward more expressive symbolic time series encodings such as TFSAX, while remaining computationally minimal and easy to implement in Python.
