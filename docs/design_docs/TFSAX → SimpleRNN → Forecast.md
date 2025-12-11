# Planning: TFSAX → SimpleRNN → Forecast 🔮

Goal: design the *data flow* and *logic flow* from a time series \(T\) through a TFSAX-style encoder into `tf.keras.layers.SimpleRNN`, and then to a forecast. No code yet, just clean shapes and responsibilities.

---

## 1. Constraints from `SimpleRNN` 🧱

From the TensorFlow / Keras docs, `SimpleRNN` expects input as a 3D tensor:  
\[
\text{inputs} \in \mathbb{R}^{(\text{batch}, \text{timesteps}, \text{features})}
\]
where:

- `batch` = number of training sequences in a batch
- `timesteps` = sequence length
- `features` = number of features per time step

Example from docs: inputs of shape `(32, 10, 8)` → 32 samples, 10 time steps, 8 features per step. :contentReference[oaicite:0]{index=0}  

So whatever TFSAX gives us, it **must be convertible** to this numeric 3D format.

---

## 2. Data Structures Around TFSAX 🧬

Let’s define the TFSAX-like encoder (your SAX + trend variant) more concretely in terms of I/O.

### 2.1 Input to TFSAX

We will work at the level of *windows* of the time series.

- Global raw time series:
  \[
  T = (t_1, t_2, \dots, t_N)
  \]
- For supervised learning we build sliding windows:
  - Input window length \(L_\text{in}\)
  - Forecast horizon \(L_\text{out}\)

For each training example \(k\):

- Input window:
  \[
  T^{(k)}_\text{in} = (t_{s_k}, \dots, t_{s_k + L_\text{in} - 1})
  \]
- Target (depends on forecast design, see §4.3):
  - Either next raw values:  
    \[
    Y^{(k)} = (t_{s_k + L_\text{in}}, \dots, t_{s_k + L_\text{in} + L_\text{out} - 1})
    \]
  - Or next symbol(s) / segment(s).

**This window \(T^{(k)}_\text{in}\)** is what we feed into our TFSAX encoder.

### 2.2 Output of TFSAX

Your TFSAX-like encoder (SAX + trend) works by segmenting the window into \(w\) segments and outputting symbols.

Conceptually:

- Input: 1D float array of length \(L_\text{in}\)
- Hyperparams:
  - number of segments \(w\)
  - SAX alphabet size \(\alpha\)
  - trend alphabet size \(\beta\) (e.g. 3: U/D/F, or more angular bins)
- Internal:
  - SAX transformer (`pyts.SymbolicAggregateApproximation`) for mean symbols
  - Trend encoder for U/D/F (or multi-bin slope)

**Symbolic output:**

- Mean symbols:  
  \[
  W = (w_1, \dots, w_w), \quad w_i \in \{0, \dots, \alpha - 1\}
  \]
  (integer IDs for SAX symbols)
- Trend symbols:  
  \[
  U = (u_1, \dots, u_w), \quad u_i \in \{0, \dots, \beta - 1\}
  \]

**Numerical feature output for a single window** (what RNN actually sees):

We need to convert \((W, U)\) into a numeric matrix of shape:
\[
X^{(k)} \in \mathbb{R}^{(w, d_\text{feat})}
\]

Two main design choices:

1. **One-hot encoding (simple, no embedding layers):**
   - Mean one-hot: \(\alpha\)-dim vector
   - Trend one-hot: \(\beta\)-dim vector
   - Concatenate:  
     \[
     d_\text{feat} = \alpha + \beta
     \]
   - Each timestep \(i\) (segment) becomes a vector:
     \[
     x^{(k)}_i = \text{one\_hot}(w_i) \Vert \text{one\_hot}(u_i)
     \]

2. **Index-based with Embedding layers (more flexible, “NLP-style”):**
   - Keep \(w_i, u_i\) as integer IDs
   - Feed them to `Embedding` layers:
     - `Embedding(α, d_mean)`
     - `Embedding(β, d_trend)`
   - Concatenate their outputs to get:
     \[
     d_\text{feat} = d_\text{mean} + d_\text{trend}
     \]
   - This shifts part of the logic into the Keras model, but conceptually the RNN still sees a float matrix of shape \((w, d_\text{feat})\).

Either way, **TFSAX’s conceptual numerical output per window** is:

- `X_window`: shape `(timesteps = w, features = d_feat)`

At dataset level:

- Stack all windows:
  \[
  X \in \mathbb{R}^{(\text{num\_windows},\ w,\ d_\text{feat})}
  \]
which matches `SimpleRNN`’s expected input layout.

---

## 3. End-to-End Logic Flow 🧠➡️📈

Let’s describe the **algorithmic flow** in stages: preprocessing, encoding, model, forecasting.

### 3.1 Preprocessing Stage

1. **Collect raw series** \(T\) (train, val, test splits).
2. (Optional) **Normalize/standardize** globally or per series (e.g. z-score).
3. Choose hyperparams:
   - \(L_\text{in}\): input window length
   - \(L_\text{out}\): forecast horizon
   - \(w\): number of TFSAX segments per window
   - \(\alpha\): SAX alphabet size
   - \(\beta\): trend alphabet size
4. Create sliding windows:
   - For training:
     - Build pairs \((T^{(k)}_\text{in}, Y^{(k)})\) for all valid starting positions.
   - Similarly for validation / test.

**Data structure after this stage:**

- `T_in_list`: list/array of shape `(num_windows, L_in)` (float)
- `Y_list`: either numeric targets or symbolic targets depending on forecast design.

### 3.2 TFSAX Encoding Stage

For each input window \(T^{(k)}_\text{in}\):

1. **SAX encoding:**
   - Use `pyts` SAX to get segment means → symbol IDs
   - Output: \(W^{(k)} \in \{0, \dots, \alpha-1\}^w\)

2. **Trend encoding:**
   - Split window into same \(w\) segments.
   - For each segment \(S^{(i)}\):
     - Compute simple trend metric (e.g. end–start difference, or slope)
     - Bin into trend symbol ID \(u_i \in \{0, \dots, \beta-1\}\)
   - Output: \(U^{(k)} \in \{0, \dots, \beta-1\}^w\)

3. **Numeric feature construction:**
   - Either:
     - One-hot encode and concatenate (pure preprocessing), or
     - Leave as IDs and rely on Embeddings (model-level decision).
   - Result:  
     \[
     X^{(k)} \in \mathbb{R}^{(w, d_\text{feat})}
     \]

Stack all:
- \( X \in \mathbb{R}^{(\text{num\_windows}, w, d_\text{feat})} \)
- Targets packed into:
  - \( Y \in \mathbb{R}^{(\text{num\_windows}, L_\text{out})} \) (numeric forecasting), or
  - a symbolic structure if forecasting symbols.

This \(X\) is what feeds into `SimpleRNN`.

---

## 4. SimpleRNN Forecaster Role 🎯

We’re still in planning mode, but we need to be clear what the RNN is actually asked to do.

### 4.1 Input to SimpleRNN

- Tensor shape:
  \[
  X_\text{batch} \in \mathbb{R}^{(B, w, d_\text{feat})}
  \]
  where \(B\) is the batch size.

- In Keras terms:  
  - `input_shape = (w, d_feat)` for a non-stateful model (batch dim inferred).

### 4.2 Output from SimpleRNN

`SimpleRNN(units=u, return_sequences=...)`:

- If `return_sequences=False` (default) → output shape `(B, u)` (last hidden state only). :contentReference[oaicite:1]{index=1}  
- If `return_sequences=True` → output shape `(B, w, u)`.

For forecasting, we’ll almost certainly:

- Use `return_sequences=False`, then:
  - Add a `Dense(...)` head to map from \(\mathbb{R}^u\) to:
    - \(\mathbb{R}^{L_\text{out}}\) for numeric multi-step forecast, or
    - \(\mathbb{R}^{|\mathcal{V}|}\) for classification over next symbol(s).

The **end-to-end contract**:

- Input batch: segment-level symbolic features \(X\)
- RNN: encodes temporal pattern of segments into a fixed-size vector
- Dense head: decodes that vector into future values or symbols.

### 4.3 Forecast Target Design Options

We need to choose what the model predicts:

1. **Numeric regression (likely simplest for time series):**
   - Target \(Y^{(k)} \in \mathbb{R}^{L_\text{out}}\) are raw future values (possibly normalized).
   - Loss: MSE/MAE.
   - Pro: easy to interpret; doesn’t require decoding symbols.
   - Con: you use symbolic features as inputs but still live in numeric space at the output.

2. **Symbolic forecasting (predict next segment’s TFSAX code):**
   - Target: next SAX symbol, trend symbol, or both.
   - Loss: categorical cross-entropy.
   - Requires an additional *decoding* step to get numeric forecast (e.g. mapping symbol→mean+trend and reconstructing a segment).
   - This is more “symbolic time series” in spirit, but more work.

The design doc for the forecaster can lock one of these in; for now we just acknowledge both are structurally supported.

---

## 5. Inference / Forecasting Flow 🛰️

At prediction time:

1. **Take the latest window** of raw series:
   \[
   T_\text{in}^{\text{(latest)}} = (t_{N - L_\text{in} + 1}, \dots, t_N)
   \]
2. **Apply the same preprocessing** as during training:
   - Normalization
   - TFSAX segmentation → \(W^{\text{(latest)}}, U^{\text{(latest)}}\)
   - Numeric feature construction → \(X^{\text{(latest)}} \in \mathbb{R}^{(w, d_\text{feat})}\)
3. **Add batch dimension:**
   - Shape becomes \((1, w, d_\text{feat})\)
4. **Feed into trained SimpleRNN model.**
5. **Decode forecast:**
   - If regression: directly interpret predicted numeric values (optionally inverse-normalize).
   - If symbolic: decode predicted symbols → approximate future segment(s).

---

## 6. Algorithmic Logic Summary (Block View) 🧩

High-level block diagram (conceptual):

1. **Raw Data Block**
   - Input: full time series \(T\)
   - Output: sliding windows \((T^{(k)}_\text{in}, Y^{(k)})\)

2. **TFSAX Encoding Block**
   - Input: \(T^{(k)}_\text{in}\)
   - Internal: SAX + trend discretization
   - Output: numeric feature tensor \(X^{(k)} \in \mathbb{R}^{(w, d_\text{feat})}\)

3. **Dataset Assembly Block**
   - Input: \(\{X^{(k)}, Y^{(k)}\}_{k=1}^{K}\)
   - Output:  
     - `X_train` / `X_val` / `X_test` with shape \((K_{\cdot}, w, d_\text{feat})\)  
     - `Y_train` / `Y_val` / `Y_test`

4. **SimpleRNN Model Block**
   - Input: tensor of shape \((B, w, d_\text{feat})\)
   - Internal: `SimpleRNN(units=...)` → `Dense(...)` forecaster head
   - Output: forecast(s) with shape \((B, L_\text{out})\) or classification logits.

5. **Inference Block**
   - Same as training path but without gradient updates.

---

This gives us a clean separation of concerns:  
TFSAX handles **symbolic, segmented representation**, and SimpleRNN handles **sequence modelling over those segments**. The next design doc can now focus purely on the SimpleRNN forecaster: its architecture, loss, training schedule, and evaluation, assuming the input is already a neatly packed tensor \((\text{batch}, w, d_\text{feat})\) from the TFSAX block.
