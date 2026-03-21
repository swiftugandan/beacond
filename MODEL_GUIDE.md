# ML Model Guide

Beacon's ML embedding backend replaces or augments the built-in 192-dim spectral signatures with learned embeddings from a pre-trained audio model. This guide walks through the full workflow — obtaining, converting, loading, and running a model — using YAMNet as the reference example.

---

## Model Requirements

The ONNX model must satisfy this contract:

| | Shape | Type |
|---|---|---|
| **Input** | `[1, num_samples]` | float32 mono audio at 16 kHz |
| **Output** | `[1, embedding_dim]` | float32 embedding vector |

Models that output per-frame embeddings `[1, num_frames, embedding_dim]` are also supported — Beacon average-pools across frames automatically.

Most audio embedding models (YAMNet, OpenL3, VGGish) conform to this after export.

---

## Step 1: Set Up a Python Environment

TensorFlow does not yet support Python 3.13+, so use Python 3.11.

```bash
python3.11 -m venv .venv-tf
source .venv-tf/bin/activate

pip install tensorflow tensorflow-hub tf2onnx onnx==1.16.0
```

> **Note**: Pin `onnx==1.16.0` to avoid version conflicts with `ml-dtypes` that occur in newer releases.

---

## Step 2: Obtain YAMNet from TensorFlow Hub

YAMNet is a MobileNet v1-based audio event classifier trained on the AudioSet corpus (521 sound classes, 2M+ clips). It produces 1024-dimensional embeddings that capture rich audio semantics — far more robust to noise and environmental variation than handcrafted spectral features.

YAMNet's raw TF Hub signature returns three outputs (class scores, per-frame embeddings, spectrogram). We need a thin wrapper that takes raw audio waveform in and produces a single average-pooled embedding vector out.

Save the following as `convert_yamnet.py`:

```python
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# Load YAMNet from TF Hub
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

class YAMNetEmbedder(tf.Module):
    """Wrapper that average-pools per-frame embeddings into one vector."""

    def __init__(self):
        super().__init__()
        self.yamnet = yamnet

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
    def __call__(self, waveform):
        scores, embeddings, spectrogram = self.yamnet(waveform)
        # embeddings shape: [num_frames, 1024]
        # Average across frames -> [1024]
        avg_embedding = tf.reduce_mean(embeddings, axis=0)
        # Add batch dimension -> [1, 1024]
        return tf.expand_dims(avg_embedding, axis=0)

# Export as SavedModel
model = YAMNetEmbedder()
tf.saved_model.save(model, "yamnet_embedder_saved")

# Verify
test_audio = np.random.randn(16000).astype(np.float32)
result = model(test_audio)
print(f"Output shape: {result.shape}")  # Expected: (1, 1024)
```

```bash
python convert_yamnet.py
```

This downloads the YAMNet weights from TF Hub (~20 MB) and saves a wrapped SavedModel to `yamnet_embedder_saved/`.

---

## Step 3: Convert to ONNX

```bash
python -m tf2onnx.convert \
  --saved-model yamnet_embedder_saved \
  --output yamnet_embedder.onnx \
  --opset 15
```

This produces `yamnet_embedder.onnx` (~13 MB). Move it into the project:

```bash
mkdir -p models
mv yamnet_embedder.onnx models/
```

The `models/` directory and `*.onnx` files are in `.gitignore` — each deployment converts or downloads its own copy.

You can verify the model with Python:

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("models/yamnet_embedder.onnx")
input_name = session.get_inputs()[0].name
test_audio = np.random.randn(1, 16000).astype(np.float32)
result = session.run(None, {input_name: test_audio})
print(f"Output shape: {result[0].shape}")  # Expected: (1, 1024)
```

---

## Step 4: Install ONNX Runtime

Beacon links ONNX Runtime dynamically at startup via the `ort` crate's `load-dynamic` feature. The shared library must be installed on the system and discoverable via `ORT_DYLIB_PATH`.

**macOS (Homebrew)**:

```bash
brew install onnxruntime

# Intel Mac
export ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.dylib

# Apple Silicon
export ORT_DYLIB_PATH=/opt/homebrew/lib/libonnxruntime.dylib
```

**Debian / Ubuntu**:

```bash
apt install libonnxruntime-dev
export ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so
```

**Manual install** (any platform):

Download a release from [github.com/microsoft/onnxruntime/releases](https://github.com/microsoft/onnxruntime/releases), extract it, and point `ORT_DYLIB_PATH` at the `.dylib` (macOS) or `.so` (Linux) file.

Add the `export` line to your shell profile (`~/.zshrc`, `~/.bashrc`) so it persists.

---

## Step 5: Build Beacon with ML Support

```bash
cargo build --release --features ml-embeddings
```

This compiles in the `ort` and `ndarray` dependencies. Without `--features ml-embeddings`, the embedding module is excluded entirely and ONNX Runtime is not required.

---

## Step 6: Run

```bash
beacond daemon --monitor --model-path models/yamnet_embedder.onnx
```

On startup you should see:

```
Loaded audio embedding model: models/yamnet_embedder.onnx (1024D embeddings)
```

All zones registered while the model is loaded get 1024-dim ML signatures. If the daemon is started without `--model-path`, zones get 192-dim spectral signatures instead. Both types coexist in the same database — Beacon's dimension-mismatch guard skips incomparable signatures during detection.

---

## Step 7: Verify End-to-End

Register a zone and run detection:

```bash
# Register from mic
beacond zone add kitchen --record --audible --duration 5

# Detect (should match even with ambient noise variation)
beacond zone detect --audible --duration 3
```

Or via the daemon TCP protocol:

```bash
# Register a zone (response includes "signature_dim": 1024)
echo '{"type":"zone_add","name":"office","path":"/path/to/office.wav","frequency_mode":"audible"}' \
  | nc localhost 18923

# Detect (response includes "method": "signature" or "fingerprint")
echo '{"type":"zone_detect","path":"/path/to/test.wav","frequency_mode":"audible"}' \
  | nc localhost 18923
```

The `method` field in the detection response tells you which matching strategy won:

- `"signature"` — ML embedding cosine similarity (or spectral, if no model loaded)
- `"fingerprint"` — hash-based Shazam-style matching

Both run in parallel on every detection; the highest-confidence result wins.

---

## Using Other Models

Any ONNX model that satisfies the input/output contract described above will work. The conversion process is the same for all models: wrap to produce `[1, embedding_dim]` from `[1, num_samples]`, export to ONNX, point `--model-path` at the file.

| Model | Embedding Dim | Size | Notes |
|-------|--------------|------|-------|
| **YAMNet** | 1024 | ~13 MB | Best general-purpose; trained on AudioSet (521 classes) |
| **OpenL3** | 512 or 6144 | ~5–50 MB | Music/environment focused; choose embedding size at export |
| **VGGish** | 128 | ~70 MB | Lightweight embeddings; older architecture |
| **PANNs (CNN14)** | 2048 | ~300 MB | Highest accuracy; significantly larger model |

---

## Troubleshooting

**"Failed to load ONNX model"**: Check that `ORT_DYLIB_PATH` points to the ONNX Runtime shared library and the model file path is correct.

**"Session lock poisoned"**: The ONNX session crashed during a previous inference. Restart the daemon.

**Embedding failed, falling back to spectral**: The model rejected the input (e.g., zero-length audio). Check the audio file. The daemon continues with spectral signatures as a fallback.

**TensorFlow not available on Python 3.13**: Use Python 3.11. Create a separate venv as shown in Step 1.

**onnx/ml-dtypes version conflict**: Pin `onnx==1.16.0` as shown in Step 1.

**Detection returns `"method": "fingerprint"` instead of `"signature"`**: The fingerprint match had higher confidence than the ML signature match. This is normal — both methods compete and the best result wins. If you want to verify ML signatures are working, test with a variant of the registered audio (different duration, slight noise) where fingerprints won't match but embeddings will.
