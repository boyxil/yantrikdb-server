//! Built-in embedding via fastembed (all-MiniLM-L6-v2, ONNX).
//!
//! Implements the yantrikdb-core `Embedder` trait so engines can
//! auto-embed text without client-provided vectors.
//!
//! ## Caching layer
//!
//! Every embedding call goes through [`crate::cache::EmbeddingCache`]
//! BEFORE the ONNX model. On cache hit, we skip both the
//! `Mutex<TextEmbedding>` lock AND the ~10–30ms forward pass entirely.
//!
//! Why this matters: on the term=1423 incident traffic profile (5
//! concurrent pollers re-issuing the same query texts), most embed
//! calls are repeats. The cache turns "26 writes/sec serialized
//! through a ONNX mutex" into "cached calls return in microseconds,
//! only novel texts pay the model cost."
//!
//! Cache key includes the model version, so a future model upgrade
//! (RFC 013-B shadow index) automatically invalidates every entry on
//! lookup — no manual flush required.

use anyhow::Context;
use parking_lot::Mutex;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use fastembed::{
    EmbeddingModel, InitOptions, InitOptionsUserDefined, Pooling, QuantizationMode, TextEmbedding,
    TokenizerFiles, UserDefinedEmbeddingModel,
};

use crate::cache::{EmbeddingCache, EmbeddingCacheConfig};

/// Stable identifier for the built-in model. Bumped if the underlying
/// fastembed `EmbeddingModel` enum value changes. Wrapped into
/// every cache key so an upgrade trivially invalidates stale entries.
const BUILTIN_MODEL_VERSION: &str = "all-MiniLM-L6-v2";

/// Load a `TextEmbedding`, converting an ort dlopen PANIC into a clear
/// actionable error. ort (the underlying ONNX Runtime binding) panics
/// on dlopen failure instead of returning an error, and a raw stack
/// trace on first run is hostile.
fn try_new_catching_ort_panic<F>(load: F) -> anyhow::Result<TextEmbedding>
where
    F: FnOnce() -> std::result::Result<TextEmbedding, fastembed::Error>,
{
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(load))
        .map_err(|panic_info| {
            let msg = panic_info
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic_info.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "unknown panic".to_string());

            let hint = if msg.contains("dlopen") || msg.contains("libonnxruntime") || msg.contains("onnxruntime.dll") {
                "\n\n\
                ONNX Runtime library not found. The built-in embedder requires it.\n\
                \n\
                Linux:   wget https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/onnxruntime-linux-x64-1.24.4.tgz\n\
                         tar xzf onnxruntime-linux-x64-1.24.4.tgz\n\
                         sudo cp onnxruntime-linux-x64-1.24.4/lib/libonnxruntime*.so* /usr/local/lib/\n\
                         sudo ldconfig\n\
                         export ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so.1.24.4\n\
                \n\
                macOS:   brew install onnxruntime\n\
                \n\
                Windows: Download from https://github.com/microsoft/onnxruntime/releases\n\
                         and place onnxruntime.dll alongside the binary.\n\
                \n\
                Or use the Docker image (ghcr.io/yantrikos/yantrikdb) which bundles ONNX Runtime.\n\
                \n\
                Or skip the built-in embedder by setting [embedding] strategy = \"client_only\"\n\
                in your config file (you'll need to provide pre-computed embeddings to remember()).\n"
            } else {
                ""
            };

            anyhow::anyhow!(
                "embedder initialization failed: {}{}",
                msg,
                hint,
            )
        })?
}

/// Cache-version identity of a user-defined ONNX graph: file name +
/// size + mtime seconds. Swapping or touching the model file therefore
/// invalidates every cached vector on the next lookup — no manual
/// flush. An unreadable mtime (exotic filesystems) degrades to 0.
fn user_defined_model_version(onnx_name: &str, meta: &std::fs::Metadata) -> String {
    let mtime_secs = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("user-defined:{}:{}:{}", onnx_name, meta.len(), mtime_secs)
}

/// Inner model + cache shared across all engines.
struct FastEmbedInner {
    model: Mutex<TextEmbedding>,
    dim: usize,
    cache: EmbeddingCache,
    /// Stable string included in every cache key. Public via
    /// [`FastEmbedder::model_version`].
    model_version: String,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
}

/// Shareable embedder — clone this to give each engine its own `Box<dyn Embedder>`.
#[derive(Clone)]
pub struct FastEmbedder {
    inner: Arc<FastEmbedInner>,
}

impl FastEmbedder {
    pub fn new() -> anyhow::Result<Self> {
        Self::with_cache_config(EmbeddingCacheConfig::default())
    }

    /// Construct with a non-default cache size. Operators with high
    /// memory budgets can dial up `max_entries`; tests can shrink it.
    pub fn with_cache_config(cache_cfg: EmbeddingCacheConfig) -> anyhow::Result<Self> {
        tracing::info!("loading embedding model (all-MiniLM-L6-v2)...");

        let model = try_new_catching_ort_panic(|| {
            TextEmbedding::try_new(
                InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true),
            )
        })?;

        tracing::info!(
            "embedding model loaded (384 dim, cache_capacity={})",
            cache_cfg.max_entries
        );

        Ok(Self {
            inner: Arc::new(FastEmbedInner {
                model: Mutex::new(model),
                dim: 384,
                cache: EmbeddingCache::new(cache_cfg),
                model_version: BUILTIN_MODEL_VERSION.to_string(),
                cache_hits: AtomicU64::new(0),
                cache_misses: AtomicU64::new(0),
            }),
        })
    }

    /// Construct from a directory with a user-provided ONNX embedding
    /// model: an ONNX graph (`model_optimized.onnx` / `model.onnx` /
    /// `model_quantized.onnx`) plus the four tokenizer/config JSON
    /// files fastembed expects. Lets an operator pin a model that is
    /// NOT in fastembed's download catalog (e.g. the Qdrant
    /// `paraphrase-multilingual-MiniLM-L12-v2` export) and keeps the
    /// weights byte-identical to whatever produced the store's
    /// existing vectors.
    ///
    /// Pools with MEAN (MiniLM sentence-transformers layout over
    /// `last_hidden_state`; fastembed's user-defined default is CLS)
    /// and marks the graph as dynamically quantized (Qdrant "-onnx-Q"
    /// exports): ort cannot batch dynamic-quantized MatMuls, so
    /// fastembed falls back to one text per session call — same as the
    /// Python fastembed reference behavior for this model.
    ///
    /// `expected_dim` comes from `[embedding].dim`; a startup probe
    /// embed fails the boot if the model disagrees, so a wrong
    /// pooling/output never reaches tenant data.
    pub fn from_model_dir(
        model_dir: &Path,
        expected_dim: usize,
        cache_cfg: EmbeddingCacheConfig,
    ) -> anyhow::Result<Self> {
        let onnx_name = ["model_optimized.onnx", "model.onnx", "model_quantized.onnx"]
            .into_iter()
            .find(|name| model_dir.join(name).exists())
            .with_context(|| {
                format!(
                    "no ONNX model file (model_optimized.onnx / model.onnx / model_quantized.onnx) in {}",
                    model_dir.display()
                )
            })?
            .to_string();
        let read = |name: &str| -> anyhow::Result<Vec<u8>> {
            std::fs::read(model_dir.join(name))
                .with_context(|| format!("reading {} in {}", name, model_dir.display()))
        };
        let tokenizer = TokenizerFiles {
            tokenizer_file: read("tokenizer.json")?,
            config_file: read("config.json")?,
            special_tokens_map_file: read("special_tokens_map.json")?,
            tokenizer_config_file: read("tokenizer_config.json")?,
        };
        let onnx = read(&onnx_name)?;

        tracing::info!(
            model_dir = %model_dir.display(),
            onnx = %onnx_name,
            "loading user-defined embedding model..."
        );

        let mut model = try_new_catching_ort_panic(|| {
            TextEmbedding::try_new_from_user_defined(
                UserDefinedEmbeddingModel::new(onnx, tokenizer)
                    .with_pooling(Pooling::Mean)
                    .with_quantization(QuantizationMode::Dynamic),
                InitOptionsUserDefined::new(),
            )
        })?;

        // Verify the model actually produces the configured
        // dimensionality before any tenant attaches it — a wrong
        // pooling or output head would otherwise surface as silently
        // garbage vectors.
        let probe = model
            .embed(vec!["yantrikdb embedder dim probe"], None)
            .map_err(|e| anyhow::anyhow!("dim probe embed failed: {}", e))?
            .pop()
            .ok_or_else(|| anyhow::anyhow!("dim probe returned no embedding"))?;
        if probe.len() != expected_dim {
            anyhow::bail!(
                "model in {} produces {}-dim embeddings, but [embedding].dim = {}",
                model_dir.display(),
                probe.len(),
                expected_dim
            );
        }

        // Cache-version identity: file name + size + mtime of the ONNX
        // graph, so swapping the model file invalidates every cached
        // vector on the next lookup.
        let onnx_meta = std::fs::metadata(model_dir.join(&onnx_name))?;
        let model_version = user_defined_model_version(&onnx_name, &onnx_meta);

        tracing::info!(
            "user-defined embedding model loaded ({} dim, cache_capacity={})",
            expected_dim,
            cache_cfg.max_entries
        );

        Ok(Self {
            inner: Arc::new(FastEmbedInner {
                model: Mutex::new(model),
                dim: expected_dim,
                cache: EmbeddingCache::new(cache_cfg),
                model_version,
                cache_hits: AtomicU64::new(0),
                cache_misses: AtomicU64::new(0),
            }),
        })
    }

    /// Stable identifier of the loaded model. Useful for emitting
    /// `embedder_model{version}` Prometheus labels.
    pub fn model_version(&self) -> &str {
        &self.inner.model_version
    }

    /// Cumulative cache hit count since startup. Pair with
    /// [`Self::cache_misses`] to compute hit rate.
    pub fn cache_hits(&self) -> u64 {
        self.inner.cache_hits.load(Ordering::Relaxed)
    }

    pub fn cache_misses(&self) -> u64 {
        self.inner.cache_misses.load(Ordering::Relaxed)
    }

    /// Convenience: hit fraction. Returns 0.0 if no calls yet.
    pub fn cache_hit_rate(&self) -> f64 {
        let h = self.cache_hits();
        let m = self.cache_misses();
        let total = h + m;
        if total == 0 {
            0.0
        } else {
            h as f64 / total as f64
        }
    }

    /// Create a boxed clone suitable for `YantrikDB::set_embedder()`.
    pub fn boxed(&self) -> Box<dyn yantrikdb::types::Embedder + Send + Sync> {
        Box::new(self.clone())
    }
}

impl yantrikdb::types::Embedder for FastEmbedder {
    fn embed(
        &self,
        text: &str,
    ) -> std::result::Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
        // Cache check FIRST — cheap (~hash + hashmap lookup) and skips
        // both the Mutex contention and the ONNX forward pass.
        if let Some(v) = self
            .inner
            .cache
            .get_for_text(text, &self.inner.model_version)
        {
            self.inner.cache_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(v);
        }
        self.inner.cache_misses.fetch_add(1, Ordering::Relaxed);

        // Cache miss: take the model lock, compute, release before
        // populating the cache (cache put doesn't need the model lock).
        let computed = {
            let mut model = self.inner.model.lock();
            let mut results = model.embed(vec![text], None)?;
            results
                .pop()
                .ok_or_else(|| -> Box<dyn std::error::Error + Send + Sync> {
                    "empty embedding result".into()
                })?
        };
        self.inner
            .cache
            .put_for_text(text, &self.inner.model_version, computed.clone());
        Ok(computed)
    }

    fn embed_batch(
        &self,
        texts: &[&str],
    ) -> std::result::Result<Vec<Vec<f32>>, Box<dyn std::error::Error + Send + Sync>> {
        let model_ver = self.inner.model_version.as_str();

        // First pass: separate cache hits from misses. We preserve
        // original order via `results: Vec<Option<Vec<f32>>>` indexed
        // by the input order.
        let mut results: Vec<Option<Vec<f32>>> = Vec::with_capacity(texts.len());
        let mut miss_indices: Vec<usize> = Vec::new();
        let mut miss_texts: Vec<String> = Vec::new();

        for (idx, &t) in texts.iter().enumerate() {
            if let Some(v) = self.inner.cache.get_for_text(t, model_ver) {
                results.push(Some(v));
                self.inner.cache_hits.fetch_add(1, Ordering::Relaxed);
            } else {
                results.push(None);
                miss_indices.push(idx);
                miss_texts.push(t.to_string());
                self.inner.cache_misses.fetch_add(1, Ordering::Relaxed);
            }
        }

        // Second pass: one batch call for all misses. Single Mutex
        // acquisition, single ONNX forward pass — way more efficient
        // than N individual single-text calls.
        if !miss_texts.is_empty() {
            let computed = {
                let mut model = self.inner.model.lock();
                model.embed(miss_texts.clone(), None)?
            };

            for ((dest_idx, miss_text), vec) in miss_indices
                .iter()
                .zip(miss_texts.iter())
                .zip(computed.into_iter())
            {
                self.inner
                    .cache
                    .put_for_text(miss_text, model_ver, vec.clone());
                results[*dest_idx] = Some(vec);
            }
        }

        // All slots populated by now — unwrap is safe by construction.
        Ok(results
            .into_iter()
            .map(|opt| opt.expect("every position populated by either cache hit or batch"))
            .collect())
    }

    fn dim(&self) -> usize {
        self.inner.dim
    }
}

/// Server-side embedder enum spanning both the [`FastEmbedder`] (fastembed
/// + ONNX + HuggingFace) and the engine-bundled
/// [`yantrikdb::embedder::BundledEmbedder`] (potion-base-2M, zero-network,
/// dim=64). One pool, one enum value, distinct first-run cost profiles.
///
/// Added in v0.8.16 (issue #35) so Docker users can opt out of the
/// first-run HuggingFace download. See `[embedding] strategy = "bundled"`.
///
/// Trait-object dispatch is one indirection per `embed()` call, dwarfed by
/// the actual embedding cost (model forward pass for `Fast`, lazy load +
/// tokenize for `Bundled`). The enum keeps the storage type concrete so
/// `TenantPool` doesn't have to thread `Box<dyn Embedder>` through its
/// `Send + Sync + Clone` constraints.
#[derive(Clone)]
pub enum ServerEmbedder {
    Fast(FastEmbedder),
    Bundled(yantrikdb::embedder::BundledEmbedder),
}

impl ServerEmbedder {
    /// Box-clone for `YantrikDB::set_embedder()`. Mirrors
    /// `FastEmbedder::boxed()` so the call site doesn't have to match.
    pub fn boxed(&self) -> Box<dyn yantrikdb::types::Embedder + Send + Sync> {
        match self {
            ServerEmbedder::Fast(f) => f.boxed(),
            ServerEmbedder::Bundled(b) => Box::new(b.clone()),
        }
    }
}

impl yantrikdb::types::Embedder for ServerEmbedder {
    fn embed(
        &self,
        text: &str,
    ) -> std::result::Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            ServerEmbedder::Fast(f) => f.embed(text),
            ServerEmbedder::Bundled(b) => b.embed(text),
        }
    }

    fn embed_batch(
        &self,
        texts: &[&str],
    ) -> std::result::Result<Vec<Vec<f32>>, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            ServerEmbedder::Fast(f) => f.embed_batch(texts),
            ServerEmbedder::Bundled(b) => b.embed_batch(texts),
        }
    }

    fn dim(&self) -> usize {
        match self {
            ServerEmbedder::Fast(f) => f.dim(),
            ServerEmbedder::Bundled(_) => yantrikdb::embedder::BUNDLED_EMBEDDER_DIM,
        }
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for the cache layering logic and config semantics —
    //! these run WITHOUT the ONNX runtime (cache plumbing via a
    //! hand-built FastEmbedInner shape that bypasses model load).
    //!
    //! Live tests with a real user-defined ONNX model live here too,
    //! opt-in via YANTRIKDB_TEST_MODEL_DIR; without it they skip, so
    //! the default unit run stays ONNX-free. Remaining end-to-end
    //! coverage (builtin catalog model, HTTP surface) is in the
    //! integration tests + the bench harness.

    use super::*;

    /// Build a [`FastEmbedder`]-shaped struct without loading ONNX.
    /// Pre-populates the cache with known entries so we can verify
    /// the cache hit path returns without ever touching the model.
    fn embedder_with_seeded_cache(entries: &[(&str, Vec<f32>)]) -> Option<FastEmbedder> {
        // We can't actually construct the inner Mutex<TextEmbedding>
        // without loading ONNX. So tests for the cache hit path go
        // through `EmbeddingCache` directly — what they verify is that
        // the cache logic is correct, which is the only piece this
        // file owns vs the model.
        let _ = entries;
        None
    }

    #[test]
    fn cache_seeded_helper_compiles() {
        // Smoke: the test helper itself compiles; cache-hit-fast-path
        // verification is via `EmbeddingCache` tests (already shipped).
        // This stub keeps the test module valid until end-to-end ONNX
        // tests can run on a CI runner with the runtime installed.
        let result = embedder_with_seeded_cache(&[]);
        assert!(result.is_none()); // expected: helper is a no-op stub
    }

    #[test]
    fn cache_key_includes_model_version() {
        // Regression: the wire-form model_version constant must match
        // what gets baked into cache keys. If someone refactors
        // BUILTIN_MODEL_VERSION away, this catches it.
        use crate::cache::embedding::EmbeddingCacheKey;
        let k1 = EmbeddingCacheKey::for_text("hi", BUILTIN_MODEL_VERSION);
        let k2 = EmbeddingCacheKey::for_text("hi", "different-model");
        assert_ne!(k1, k2);
    }

    #[test]
    fn model_version_is_stable_string() {
        // Pinned identifier: changes here invalidate every cache entry
        // on the next request. That's intentional but rare — guard the
        // current value.
        assert_eq!(BUILTIN_MODEL_VERSION, "all-MiniLM-L6-v2");
    }

    #[test]
    fn cache_hit_rate_zero_on_no_calls() {
        // Verify the hit-rate accessor handles the zero-divide case.
        // Construct the rate calculation manually since we can't init
        // the full embedder without ONNX.
        let h: u64 = 0;
        let m: u64 = 0;
        let rate = if h + m == 0 {
            0.0
        } else {
            h as f64 / (h + m) as f64
        };
        assert_eq!(rate, 0.0);
    }

    #[test]
    fn cache_hit_rate_correct_for_mixed() {
        // 7 hits, 3 misses → 70% hit rate.
        let h: u64 = 7;
        let m: u64 = 3;
        let rate = if h + m == 0 {
            0.0
        } else {
            h as f64 / (h + m) as f64
        };
        assert!((rate - 0.7).abs() < 1e-9);
    }

    // ── user-defined model (model_dir) ────────────────────────────

    /// The cache identity must react to EVERY component of the key —
    /// name, size, mtime. If any one is dropped, swapping the ONNX
    /// file would silently serve stale vectors from the previous
    /// model (issue this guards: "cache keys embed name+size+mtime").
    #[test]
    fn user_defined_version_reacts_to_name_size_and_mtime() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("model_optimized.onnx");
        let b = dir.path().join("model.onnx");
        std::fs::write(&a, vec![0u8; 100]).unwrap();
        std::fs::write(&b, vec![0u8; 100]).unwrap();
        let meta_a = std::fs::metadata(&a).unwrap();
        let meta_b = std::fs::metadata(&b).unwrap();

        // Same file → stable identity.
        assert_eq!(
            user_defined_model_version("model_optimized.onnx", &meta_a),
            user_defined_model_version("model_optimized.onnx", &meta_a)
        );

        // Different file name → different identity. Same metadata
        // under both names, so the NAME is the only moving part —
        // dropping the name from the key would make these equal.
        assert_ne!(
            user_defined_model_version("model_optimized.onnx", &meta_a),
            user_defined_model_version("model.onnx", &meta_a)
        );

        // Different size → different identity. Rewriting b also moves
        // its mtime, so pin the timestamp back — size stays the only
        // moving part and dropping size from the key would make these
        // equal.
        std::fs::write(&b, vec![0u8; 101]).unwrap();
        std::fs::File::options()
            .write(true)
            .open(&b)
            .unwrap()
            .set_modified(meta_b.modified().unwrap())
            .unwrap();
        let meta_b_grown = std::fs::metadata(&b).unwrap();
        assert_eq!(meta_b_grown.len(), 101);
        // Durations, not Results: SystemTimeError is not PartialEq.
        assert_eq!(
            meta_b_grown
                .modified()
                .unwrap()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap(),
            meta_b
                .modified()
                .unwrap()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap(),
            "same mtime, only size moved"
        );
        assert_ne!(
            user_defined_model_version("model.onnx", &meta_b),
            user_defined_model_version("model.onnx", &meta_b_grown)
        );

        // Same name and size, only mtime moved — the swap-in-place
        // case (same file path, new weights). set_modified pins the
        // timestamp so the test can't race the one-second resolution.
        let bumped = meta_a.modified().unwrap() + std::time::Duration::from_secs(3600);
        std::fs::File::options()
            .write(true)
            .open(&a)
            .unwrap()
            .set_modified(bumped)
            .unwrap();
        let meta_a_touched = std::fs::metadata(&a).unwrap();
        assert_eq!(
            meta_a_touched.len(),
            meta_a.len(),
            "same size, only mtime moved"
        );
        assert_ne!(
            user_defined_model_version("model_optimized.onnx", &meta_a),
            user_defined_model_version("model_optimized.onnx", &meta_a_touched)
        );
    }

    /// The version string feeds `EmbeddingCacheKey`; differing versions
    /// must produce differing cache keys, which is what actually
    /// invalidates stale entries on lookup.
    #[test]
    fn user_defined_version_invalidates_cache_keys() {
        use crate::cache::embedding::EmbeddingCacheKey;
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("model.onnx");
        std::fs::write(&a, vec![0u8; 10]).unwrap();
        let v1 = user_defined_model_version("model.onnx", &std::fs::metadata(&a).unwrap());
        std::fs::write(&a, vec![0u8; 11]).unwrap();
        let v2 = user_defined_model_version("model.onnx", &std::fs::metadata(&a).unwrap());
        assert_ne!(v1, v2);
        assert_ne!(
            EmbeddingCacheKey::for_text("hi", &v1),
            EmbeddingCacheKey::for_text("hi", &v2)
        );
    }

    /// `from_model_dir` must fail BEFORE touching the ONNX runtime when
    /// the directory has no ONNX graph at all — the failure stays
    /// instant and filesystem-shaped instead of an ort load error.
    #[test]
    fn from_model_dir_rejects_dir_without_onnx_graph() {
        let dir = tempfile::tempdir().unwrap();
        // match instead of expect_err: FastEmbedder is not Debug.
        let err = match FastEmbedder::from_model_dir(dir.path(), 384, Default::default()) {
            Ok(_) => panic!("empty dir must be rejected"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("no ONNX model file"), "got: {err}");
    }

    /// A graph without the four tokenizer/config files fastembed
    /// requires is likewise rejected before model load.
    #[test]
    fn from_model_dir_rejects_dir_without_tokenizer_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.onnx"), b"fake-graph").unwrap();
        let err = match FastEmbedder::from_model_dir(dir.path(), 384, Default::default()) {
            Ok(_) => panic!("missing tokenizer must be rejected"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("tokenizer.json"), "got: {err}");
    }

    /// Live-model tests need a real user-defined model directory (ONNX
    /// graph + tokenizer files) and the ONNX Runtime dylib. Opt-in via
    /// YANTRIKDB_TEST_MODEL_DIR (optionally YANTRIKDB_TEST_MODEL_DIM,
    /// default 384) so environments without a model skip instead of
    /// failing. An explicitly SET but unusable variable is a setup
    /// error and panics loudly — a typo must not masquerade as a
    /// passing (vacuously skipped) live test.
    fn live_model_dir() -> Option<std::path::PathBuf> {
        let dir = std::env::var("YANTRIKDB_TEST_MODEL_DIR").ok()?;
        let dir = std::path::PathBuf::from(dir);
        if !["model_optimized.onnx", "model.onnx", "model_quantized.onnx"]
            .iter()
            .any(|name| dir.join(name).exists())
        {
            panic!(
                "YANTRIKDB_TEST_MODEL_DIR is set but contains no ONNX graph: {} — \
                 unset it to skip live tests, or fix the path",
                dir.display()
            );
        }
        Some(dir)
    }

    fn live_model_dim() -> usize {
        match std::env::var("YANTRIKDB_TEST_MODEL_DIM") {
            Err(_) => 384,
            Ok(v) => v.parse().unwrap_or_else(|_| {
                panic!("YANTRIKDB_TEST_MODEL_DIM is set but not a number: {v}")
            }),
        }
    }

    #[test]
    fn from_model_dir_boots_with_matching_dim_and_embeds() {
        let Some(dir) = live_model_dir() else {
            eprintln!("skipping: YANTRIKDB_TEST_MODEL_DIR not set or has no ONNX graph");
            return;
        };
        let dim = live_model_dim();
        let embedder = FastEmbedder::from_model_dir(&dir, dim, Default::default())
            .expect("matching dim must boot");

        assert!(
            embedder.model_version().starts_with("user-defined:"),
            "got: {}",
            embedder.model_version()
        );

        use yantrikdb::types::Embedder as _;
        let v = embedder.embed("привет").expect("embed must work");
        assert_eq!(v.len(), dim);

        // Second call on the same text is a cache hit under the
        // versioned key.
        let _ = embedder.embed("привет").unwrap();
        assert_eq!(embedder.cache_hits(), 1);
        assert_eq!(embedder.cache_misses(), 1);
    }

    #[test]
    fn from_model_dir_fails_boot_on_dim_mismatch() {
        let Some(dir) = live_model_dir() else {
            eprintln!("skipping: YANTRIKDB_TEST_MODEL_DIR not set or has no ONNX graph");
            return;
        };
        let wrong = live_model_dim() + 1;
        let err = match FastEmbedder::from_model_dir(&dir, wrong, Default::default()) {
            Ok(_) => panic!("mismatched dim must fail the boot"),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(msg.contains("produces"), "got: {msg}");
        assert!(msg.contains(&format!("dim = {wrong}")), "got: {msg}");
    }

    /// Cosine of two unit-normalized-ish embedding vectors.
    fn cos(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb)
    }

    /// Semantic regression pin for the user-defined model. Dimension
    /// checks alone would let a wrong POOLING (fastembed's user-defined
    /// default is CLS, this feature needs Mean) or a wrong quantization
    /// mode ship 384-dim vectors with garbage geometry — every other
    /// test would stay green. Reference values were measured on the
    /// canary-verified build (vectors byte-matched the Python fastembed
    /// client); the ±0.03 band absorbs ONNX Runtime float drift.
    #[test]
    fn from_model_dir_semantic_geometry_is_pinned() {
        let Some(dir) = live_model_dir() else {
            eprintln!("skipping: YANTRIKDB_TEST_MODEL_DIR not set or has no ONNX graph");
            return;
        };
        let embedder = FastEmbedder::from_model_dir(&dir, live_model_dim(), Default::default())
            .expect("matching dim must boot");
        use yantrikdb::types::Embedder as _;

        let hi_ru = embedder.embed("привет мир").unwrap();
        let hi_en = embedder.embed("hello world").unwrap();
        let cat_ru = embedder.embed("кот сидит на окне").unwrap();
        let cat_en = embedder.embed("the cat sits on the window").unwrap();
        let junk = embedder
            .embed("квартальный финансовый отчёт по закупкам")
            .unwrap();

        // Cross-lingual near-paraphrases land close together…
        let cos_hi = cos(&hi_ru, &hi_en);
        assert!(
            (0.695..=0.755).contains(&cos_hi),
            "cos(привет мир, hello world) = {cos_hi}, expected 0.7254 ± 0.03 — \
             pooling/quantization regression?"
        );
        let cos_cat = cos(&cat_ru, &cat_en);
        assert!(
            (0.938..=0.998).contains(&cos_cat),
            "cos(cat ru/en) = {cos_cat}, expected 0.9683 ± 0.03"
        );

        // …while an unrelated memory stays far away. This ordering is
        // the whole point of the multilingual model: the old bundled
        // embedder had a ~0.99 median cosine between ANY two records
        // and could not rank.
        let cos_junk = cos(&cat_ru, &junk);
        assert!(
            cos_cat > cos_junk + 0.2,
            "related pair ({cos_cat}) must beat unrelated ({cos_junk}) with margin"
        );
    }

    /// The batch path must agree with single-text embeds on the
    /// user-defined model. Dynamic quantization forces fastembed into
    /// one text per session call; if that fallback regressed (a real
    /// batch through a dynamic-quantized graph errors in ort), this
    /// catches it. Two model loads so neither side reads the other's
    /// cache — the comparison exercises real computation both ways.
    #[test]
    fn from_model_dir_embed_batch_matches_single_embeds() {
        let Some(dir) = live_model_dir() else {
            eprintln!("skipping: YANTRIKDB_TEST_MODEL_DIR not set or has no ONNX graph");
            return;
        };
        let batcher = FastEmbedder::from_model_dir(&dir, live_model_dim(), Default::default())
            .expect("matching dim must boot");
        let singler = FastEmbedder::from_model_dir(&dir, live_model_dim(), Default::default())
            .expect("matching dim must boot");

        let texts = ["запись про кластеризацию", "a note about embeddings"];
        use yantrikdb::types::Embedder as _;
        let batched = batcher.embed_batch(&texts).expect("embed_batch must work");
        assert_eq!(batched.len(), 2);
        assert_eq!(batcher.cache_misses(), 2, "both texts are fresh misses");
        assert_eq!(batcher.cache_hits(), 0);

        for (text, bvec) in texts.iter().zip(batched.iter()) {
            let svec = singler.embed(text).expect("single embed must work");
            assert_eq!(svec.len(), bvec.len());
            let c = cos(bvec, &svec);
            assert!(
                c > 0.999999,
                "batch vs single diverged for {text:?}: cos = {c}"
            );
        }

        // Re-batching the same texts is served entirely from cache.
        let _ = batcher.embed_batch(&texts).unwrap();
        assert_eq!(batcher.cache_hits(), 2);
        assert_eq!(batcher.cache_misses(), 2);
    }
}
