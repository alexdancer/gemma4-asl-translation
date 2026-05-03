#!/usr/bin/env python3
"""
Unit tests for Cactus model export functionality

Tests:
- Quantization correctness
- Model shape consistency
- Config JSON validity
- Size constraints
- Inference benchmarking
- Checkpoint compatibility
"""

import unittest
import json
import struct
import sys
import types
import torch
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Import exporter
from src.mobile.cactus_export import (
    QuantizationConfig,
    CactusConfig,
    QuantizationCalibrator,
    CactusModelExporter,
    UnsupportedCheckpointError,
    main,
)


class TinyConfig:
    vocab_size = 256000
    hidden_size = 2304
    num_hidden_layers = 18
    num_attention_heads = 8
    max_position_embeddings = 2048
    use_cache = True


class TinyCheckpointModel:
    def __init__(self):
        generator = torch.Generator().manual_seed(42)
        self.config = TinyConfig()
        self.state_dict_data = {
            "pose_projection.weight": torch.randn(64, 32, generator=generator),
            "pose_projection.bias": torch.randn(64, generator=generator),
            "decoder.layers.0.input_layernorm.weight": torch.randn(64, generator=generator),
            "decoder.layers.0.self_attn.q_proj.weight": torch.randn(64, 64, generator=generator),
            "decoder.layers.0.mlp.up_proj.weight": torch.randn(128, 64, generator=generator),
            "decoder.layers.0.step": torch.tensor(1, dtype=torch.int64),
        }

    def state_dict(self):
        return self.state_dict_data


class TinyTokenizer:
    vocab_size = 256000
    model_max_length = 2048
    eos_token = "<eos>"
    bos_token = "<bos>"
    unk_token = "<unk>"
    pad_token = "<pad>"


def tiny_checkpoint_pair():
    return TinyCheckpointModel(), TinyTokenizer()


def create_checkpoint_dir(parent: str | Path) -> Path:
    checkpoint_dir = Path(parent) / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "trainer_state.pt").write_bytes(b"task-a-test-checkpoint")
    return checkpoint_dir


def read_cactus_bundle(path: Path):
    data = path.read_bytes()
    assert data[:8] == b"CACTUSI4"
    manifest_len = struct.unpack("<I", data[8:12])[0]
    manifest_start = 12
    manifest_end = manifest_start + manifest_len
    manifest = json.loads(data[manifest_start:manifest_end].decode("utf-8"))
    offset = manifest_end
    records = []

    for _ in manifest["tensors"]:
        name_len = struct.unpack("<I", data[offset : offset + 4])[0]
        offset += 4
        name = data[offset : offset + name_len].decode("utf-8")
        offset += name_len
        scale_len = struct.unpack("<I", data[offset : offset + 4])[0]
        offset += 4
        scale_bytes = data[offset : offset + scale_len]
        offset += scale_len
        packed_len = struct.unpack("<I", data[offset : offset + 4])[0]
        offset += 4
        packed_bytes = data[offset : offset + packed_len]
        offset += packed_len
        records.append((name, scale_bytes, packed_bytes))

    assert offset == len(data)
    return manifest, records


class TestQuantizationCalibrator(unittest.TestCase):
    """Test INT4 quantization logic"""
    
    def setUp(self):
        self.config = QuantizationConfig()
        self.calibrator = QuantizationCalibrator(self.config)
    
    def test_quantize_weights_shape_consistency(self):
        """Verify quantized weights maintain shape"""
        weights = torch.randn(2304, 2304)
        quantized = self.calibrator.quantize_weights(weights, "test_layer")
        
        self.assertEqual(quantized.shape, weights.shape)
        self.assertEqual(quantized.dtype, torch.int8)
    
    def test_quantize_weights_range(self):
        """Verify quantized values are in INT4 range"""
        weights = torch.randn(128, 256)
        quantized = self.calibrator.quantize_weights(weights, "test_layer")
        
        self.assertLessEqual(quantized.max().item(), 7)
        self.assertGreaterEqual(quantized.min().item(), -8)
    
    def test_calibration_report_generation(self):
        """Test calibration report creation"""
        weights = torch.randn(64, 128)
        self.calibrator.quantize_weights(weights, "layer_0")
        self.calibrator.quantize_weights(weights, "layer_1")
        
        report = self.calibrator.get_calibration_report()
        
        self.assertEqual(report["num_quantized_layers"], 2)
        self.assertIn("layer_0", report["layer_statistics"])
        self.assertIn("layer_1", report["layer_statistics"])

    def test_error_metrics_are_chunked_and_labeled(self):
        """Verify reconstruction metrics do not require a full reconstructed tensor."""
        config = QuantizationConfig(metric_chunk_elements=17)
        calibrator = QuantizationCalibrator(config)
        calibrator.quantize_weights(torch.randn(9, 11), "chunked_layer")

        stats = calibrator.get_calibration_report()["layer_statistics"]["chunked_layer"]

        self.assertEqual(stats["error_stats_method"], "exact_chunked_reconstruction")
        self.assertEqual(stats["error_stats_chunk_elements"], 17)
        self.assertIn("mean_abs_error", stats)
        self.assertIn("max_abs_error", stats)


class TestCactusConfig(unittest.TestCase):
    """Test configuration object"""
    
    def setUp(self):
        self.config = CactusConfig(
            model_name="test-model",
            model_version="1.0.0",
            vocab_size=256000,
            max_seq_length=8192,
            hidden_size=2304,
            num_layers=18,
            num_attention_heads=8,
            quantization_scheme="symmetric",
            quantization_bits=4,
            activation_dtype="float16",
            weight_dtype="int4",
            kv_cache_quantization="dynamic",
        )
    
    def test_config_to_dict(self):
        """Test config serialization"""
        config_dict = self.config.to_dict()
        
        self.assertEqual(config_dict["model_name"], "test-model")
        self.assertEqual(config_dict["vocab_size"], 256000)
        self.assertIn("quantization_date", config_dict)
    
    def test_config_json_valid(self):
        """Verify config can be JSON serialized"""
        config_dict = self.config.to_dict()
        json_str = json.dumps(config_dict)
        
        self.assertIsInstance(json_str, str)
        
        # Verify it can be deserialized
        loaded = json.loads(json_str)
        self.assertEqual(loaded["model_name"], "test-model")


class TestCactusModelExporter(unittest.TestCase):
    """Test main exporter class"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = create_checkpoint_dir(self.temp_dir)
        self.exporter = CactusModelExporter(output_dir=self.temp_dir)

    def test_model_loading(self):
        """Test model loading uses an explicit checkpoint path."""
        with patch("src.mobile.cactus_export.load_checkpoint", return_value=tiny_checkpoint_pair()) as loader:
            model, tokenizer = self.exporter.load_model(self.checkpoint_dir)
        
        self.assertIsNotNone(model)
        self.assertIsNotNone(tokenizer)
        self.assertEqual(model.config.vocab_size, 256000)
        loader.assert_called_once_with(str(self.checkpoint_dir.resolve()))

    def test_model_loading_requires_checkpoint(self):
        """Verify no hidden mock fallback is used when checkpoint is omitted."""
        with self.assertRaises(ValueError):
            self.exporter.load_model(None)

    def test_model_loading_missing_checkpoint_fails(self):
        """Verify missing checkpoints fail before export work starts."""
        with self.assertRaises(FileNotFoundError):
            self.exporter.load_model(Path(self.temp_dir) / "missing")

    def test_model_loading_rejects_checkpoint_file(self):
        """Verify unsupported checkpoint shapes fail with an explicit error."""
        checkpoint_file = Path(self.temp_dir) / "checkpoint.pt"
        checkpoint_file.write_bytes(b"not-a-save-pretrained-dir")

        with self.assertRaisesRegex(UnsupportedCheckpointError, "must be a directory"):
            self.exporter.load_model(checkpoint_file)

    def test_model_loading_supports_adapter_checkpoint_path(self):
        """Verify adapter checkpoints use the explicit adapter load path."""
        (self.checkpoint_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        adapter_model = TinyCheckpointModel()
        adapter_tokenizer = TinyTokenizer()
        base_loader = MagicMock(return_value=MagicMock())
        adapter_loader = MagicMock(return_value=adapter_model)
        tokenizer_loader = MagicMock(return_value=adapter_tokenizer)
        fake_peft = types.SimpleNamespace(PeftModel=types.SimpleNamespace(from_pretrained=adapter_loader))
        fake_transformers = types.SimpleNamespace(
            AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=base_loader),
            AutoTokenizer=types.SimpleNamespace(from_pretrained=tokenizer_loader),
        )

        with patch("src.mobile.cactus_export.load_checkpoint", side_effect=RuntimeError("not a full checkpoint")):
            with patch.dict(sys.modules, {"peft": fake_peft, "transformers": fake_transformers}):
                model, tokenizer = self.exporter.load_model(self.checkpoint_dir)

        self.assertIs(model, adapter_model)
        self.assertIs(tokenizer, adapter_tokenizer)
        base_loader.assert_called_once_with("google/gemma-4-E2B-it")
        adapter_loader.assert_called_once()
        tokenizer_loader.assert_called_once_with(str(self.checkpoint_dir.resolve()))

    def test_adapter_checkpoint_failure_is_actionable(self):
        """Verify adapter-only failures identify the unsupported production path."""
        (self.checkpoint_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        base_loader = MagicMock(side_effect=OSError("base unavailable"))
        fake_peft = types.SimpleNamespace(PeftModel=types.SimpleNamespace(from_pretrained=MagicMock()))
        fake_transformers = types.SimpleNamespace(
            AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=base_loader),
            AutoTokenizer=types.SimpleNamespace(from_pretrained=MagicMock()),
        )

        with patch("src.mobile.cactus_export.load_checkpoint", side_effect=RuntimeError("not a full checkpoint")):
            with patch.dict(sys.modules, {"peft": fake_peft, "transformers": fake_transformers}):
                with self.assertRaisesRegex(UnsupportedCheckpointError, "Adapter loader error"):
                    self.exporter.load_model(self.checkpoint_dir)
    
    def test_quantization_setup(self):
        """Test quantization initialization"""
        self.exporter.setup_quantization()
        
        self.assertIsNotNone(self.exporter.quantization_config)
        self.assertIsNotNone(self.exporter.calibrator)
        self.assertEqual(self.exporter.quantization_config.quantization_bits, 4)
    
    def test_quantize_model(self):
        """Test model quantization"""
        with patch("src.mobile.cactus_export.load_checkpoint", return_value=tiny_checkpoint_pair()):
            self.exporter.load_model(self.checkpoint_dir)
        self.exporter.setup_quantization()
        
        stats = self.exporter.quantize_model()
        
        self.assertIn("total_layers_quantized", stats)
        self.assertIn("total_parameters_original_mb", stats)
        self.assertIn("total_parameters_quantized_mb", stats)
        self.assertGreater(stats["total_layers_quantized"], 0)

    def test_quantize_model_includes_one_dimensional_float_tensors(self):
        """Verify required norm and bias vectors are exported."""
        with patch("src.mobile.cactus_export.load_checkpoint", return_value=tiny_checkpoint_pair()):
            self.exporter.load_model(self.checkpoint_dir)
        self.exporter.setup_quantization()

        self.exporter.quantize_model()

        self.assertIn("pose_projection.bias", self.exporter.quantized_tensors)
        self.assertIn("decoder.layers.0.input_layernorm.weight", self.exporter.quantized_tensors)
        self.assertNotIn("decoder.layers.0.step", self.exporter.quantized_tensors)
    
    def test_config_creation(self):
        """Test Cactus config generation"""
        with patch("src.mobile.cactus_export.load_checkpoint", return_value=tiny_checkpoint_pair()):
            self.exporter.load_model(self.checkpoint_dir)
        config = self.exporter.create_cactus_config()
        
        self.assertEqual(config.model_name, "gemma-4-2b-e2b-asl")
        self.assertEqual(config.vocab_size, 256000)
        self.assertEqual(config.quantization_bits, 4)
    
    def test_tokenizer_export(self):
        """Test tokenizer export"""
        with patch("src.mobile.cactus_export.load_checkpoint", return_value=tiny_checkpoint_pair()):
            self.exporter.load_model(self.checkpoint_dir)
        
        tokens_path = self.exporter.export_tokenizer()
        
        self.assertTrue(tokens_path.exists())
        
        with open(tokens_path, 'r') as f:
            tokens = json.load(f)
        
        self.assertIn("vocab_size", tokens)
        self.assertIn("max_length", tokens)
    
    def test_weights_export(self):
        """Test weight export"""
        with patch("src.mobile.cactus_export.load_checkpoint", return_value=tiny_checkpoint_pair()):
            self.exporter.load_model(self.checkpoint_dir)
        self.exporter.setup_quantization()
        self.exporter.quantize_model()
        
        weights_path = self.exporter.export_weights()
        
        self.assertTrue(weights_path.exists())
        self.assertGreater(weights_path.stat().st_size, 0)
        manifest, records = read_cactus_bundle(weights_path)
        self.assertEqual(len(records), len(manifest["tensors"]))
        self.assertTrue(all(scale_bytes for _, scale_bytes, _ in records))
        self.assertTrue(all(packed_bytes for _, _, packed_bytes in records))

    def test_weights_bundle_covers_required_float_tensors(self):
        """Verify the serialized bundle includes every non-scalar floating tensor."""
        with patch("src.mobile.cactus_export.load_checkpoint", return_value=tiny_checkpoint_pair()):
            self.exporter.load_model(self.checkpoint_dir)
        self.exporter.setup_quantization()
        self.exporter.quantize_model()

        weights_path = self.exporter.export_weights()
        manifest, records = read_cactus_bundle(weights_path)
        manifest_by_name = {entry["name"]: entry for entry in manifest["tensors"]}
        record_names = {name for name, _, _ in records}
        required_names = {
            name
            for name, tensor in tiny_checkpoint_pair()[0].state_dict().items()
            if torch.is_tensor(tensor) and tensor.ndim >= 1 and tensor.is_floating_point()
        }

        self.assertEqual(set(manifest_by_name), required_names)
        self.assertEqual(record_names, required_names)
        self.assertIn("pose_projection.bias", manifest_by_name)
        self.assertIn("decoder.layers.0.input_layernorm.weight", manifest_by_name)
        self.assertNotIn("decoder.layers.0.step", manifest_by_name)
        self.assertEqual(manifest_by_name["pose_projection.bias"]["shape"], [64])
        self.assertEqual(manifest_by_name["pose_projection.bias"]["per_channel_axis"], None)
    
    def test_benchmark_inference(self):
        """Test inference benchmarking"""
        with patch("src.mobile.cactus_export.load_checkpoint", return_value=tiny_checkpoint_pair()):
            self.exporter.load_model(self.checkpoint_dir)
        self.exporter.setup_quantization()
        self.exporter.create_cactus_config()
        stats = self.exporter.benchmark_inference()
        
        self.assertIn("mean_latency_ms", stats)
        self.assertIn("median_latency_ms", stats)
        self.assertIn("p99_latency_ms", stats)
        self.assertIn("memory_peak_mb", stats)
        
        # Verify latency under 200ms target
        self.assertLess(stats["mean_latency_ms"], 200)
    
    def test_export_manifest_creation(self):
        """Test manifest creation"""
        with patch("src.mobile.cactus_export.load_checkpoint", return_value=tiny_checkpoint_pair()):
            self.exporter.load_model(self.checkpoint_dir)
        self.exporter.setup_quantization()
        quant_stats = self.exporter.quantize_model()
        self.exporter.create_cactus_config()
        self.exporter.export_weights()
        bench_stats = self.exporter.benchmark_inference()
        
        manifest_path = self.exporter.create_export_manifest(quant_stats, bench_stats)
        
        self.assertTrue(manifest_path.exists())
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        self.assertIn("export_metadata", manifest)
        self.assertIn("quantization_info", manifest)
        self.assertIn("performance", manifest)
        self.assertIn("success_criteria", manifest)
        self.assertIn("manifest", manifest["files"])
    
    def test_quantization_report_generation(self):
        """Test quantization report"""
        with patch("src.mobile.cactus_export.load_checkpoint", return_value=tiny_checkpoint_pair()):
            self.exporter.load_model(self.checkpoint_dir)
        self.exporter.setup_quantization()
        quant_stats = self.exporter.quantize_model()
        
        report_path = self.exporter.generate_quantization_report(quant_stats)
        
        self.assertTrue(report_path.exists())
        
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        self.assertIn("quantization_summary", report)
        self.assertIn("layer_statistics", report)
        first_layer_stats = next(iter(report["layer_statistics"].values()))
        self.assertEqual(first_layer_stats["error_stats_method"], "exact_chunked_reconstruction")
    
    def test_full_export_pipeline(self):
        """Test complete export workflow"""
        with patch("src.mobile.cactus_export.load_checkpoint", return_value=tiny_checkpoint_pair()):
            result = self.exporter.export(self.checkpoint_dir)
        
        self.assertTrue(result["success"])
        self.assertIn("output_dir", result)
        self.assertIn("model_size_mb", result)
        self.assertIn("quantization_stats", result)
        self.assertIn("benchmark_stats", result)
        
        output_dir = Path(result["output_dir"])
        
        # Verify all expected files exist
        expected_files = [
            "model_int4.bin",
            "config.json",
            "special_tokens.json",
            "export_manifest.json",
            "quantization_report.json",
        ]
        
        for file in expected_files:
            self.assertTrue((output_dir / file).exists(), f"Missing {file}")

        manifest = json.loads((output_dir / "export_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(
            set(manifest["files"]),
            {"weights", "special_tokens", "config", "quantization_report", "manifest"},
        )
        self.assertEqual(manifest["files"]["manifest"], str(output_dir / "export_manifest.json"))
        self.assertEqual(manifest["files"]["quantization_report"], str(output_dir / "quantization_report.json"))

    def test_export_failure_propagates_without_fake_success_artifacts(self):
        """Verify failed model loading does not create success artifacts."""
        bad_checkpoint = Path(self.temp_dir) / "bad-checkpoint"
        bad_checkpoint.mkdir()
        (bad_checkpoint / "config.json").write_text("{}", encoding="utf-8")

        with patch("src.mobile.cactus_export.load_checkpoint", side_effect=RuntimeError("loader failed")):
            with self.assertRaises(RuntimeError):
                self.exporter.export(bad_checkpoint)

        for file in ("model_int4.bin", "config.json", "special_tokens.json", "export_manifest.json", "quantization_report.json"):
            self.assertFalse((Path(self.temp_dir) / file).exists(), f"Unexpected artifact {file}")
    
    def test_model_size_constraint(self):
        """Verify model size is under 500MB"""
        with patch("src.mobile.cactus_export.load_checkpoint", return_value=tiny_checkpoint_pair()):
            result = self.exporter.export(self.checkpoint_dir)
        
        self.assertLess(result["model_size_mb"], 500)
    
    def test_inference_latency_constraint(self):
        """Verify inference latency under 200ms"""
        with patch("src.mobile.cactus_export.load_checkpoint", return_value=tiny_checkpoint_pair()):
            result = self.exporter.export(self.checkpoint_dir)
        
        mean_latency = result["benchmark_stats"]["mean_latency_ms"]
        self.assertLess(mean_latency, 200)
    
    def test_memory_constraint(self):
        """Verify memory usage under 500MB"""
        with patch("src.mobile.cactus_export.load_checkpoint", return_value=tiny_checkpoint_pair()):
            result = self.exporter.export(self.checkpoint_dir)
        
        peak_memory = result["benchmark_stats"]["memory_peak_mb"]
        self.assertLess(peak_memory, 500)
    
    def test_config_json_schema(self):
        """Verify config.json matches Cactus schema"""
        with patch("src.mobile.cactus_export.load_checkpoint", return_value=tiny_checkpoint_pair()):
            self.exporter.load_model(self.checkpoint_dir)
        self.exporter.create_cactus_config()
        
        config_path = Path(self.temp_dir) / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.exporter.config.to_dict(), f)
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Verify required fields
        required_fields = [
            "model_name",
            "vocab_size",
            "max_seq_length",
            "quantization_scheme",
            "quantization_bits",
        ]
        
        for field in required_fields:
            self.assertIn(field, config)


class TestOutputDirectory(unittest.TestCase):
    """Test output directory structure"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = create_checkpoint_dir(self.temp_dir)
        self.exporter = CactusModelExporter(output_dir=self.temp_dir)
    
    def test_output_directory_creation(self):
        """Verify output directory is created"""
        self.assertTrue(Path(self.temp_dir).exists())
        self.assertTrue(Path(self.temp_dir).is_dir())
    
    def test_output_structure_after_export(self):
        """Verify complete output structure"""
        with patch("src.mobile.cactus_export.load_checkpoint", return_value=tiny_checkpoint_pair()):
            self.exporter.export(self.checkpoint_dir)
        
        output_dir = Path(self.temp_dir)
        
        # Check directory exists and is not empty
        self.assertTrue(output_dir.exists())
        files = list(output_dir.glob("*"))
        self.assertGreater(len(files), 0)
    
    def test_manifest_integrity(self):
        """Verify manifest contains integrity hashes"""
        with patch("src.mobile.cactus_export.load_checkpoint", return_value=tiny_checkpoint_pair()):
            self.exporter.export(self.checkpoint_dir)
        
        manifest_path = Path(self.temp_dir) / "export_manifest.json"
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        self.assertIn("export_metadata", manifest)
        self.assertIn("model_hash_sha256", manifest["export_metadata"])
        self.assertIn("quantization_report", manifest["files"])
        self.assertIn("manifest", manifest["files"])

    def test_cli_quantization_bits_argument_is_effective(self):
        """Verify CLI arguments are wired into export config validation."""
        output_dir = Path(self.temp_dir) / "cli-output"
        with patch(
            "sys.argv",
            [
                "cactus_export.py",
                "--checkpoint",
                str(self.checkpoint_dir),
                "--output-dir",
                str(output_dir),
                "--quantization-bits",
                "8",
            ],
        ):
            self.assertEqual(main(), 1)

        self.assertFalse((output_dir / "export_manifest.json").exists())


class TestDockerContextHygiene(unittest.TestCase):
    """Regression tests for Docker build context safety."""

    def test_dockerignore_excludes_heavy_and_private_artifacts(self):
        dockerignore = Path(".dockerignore").read_text(encoding="utf-8").splitlines()
        patterns = {line.strip() for line in dockerignore if line.strip() and not line.startswith("#")}

        required_patterns = {
            ".git",
            ".env",
            ".env.*",
            "data",
            "checkpoints",
            "cactus_export",
            "outputs",
            "runs",
            "wandb",
            "mlruns",
            "*.pt",
            "*.pth",
            "*.ckpt",
            "*.safetensors",
            "*.bin",
            "*.parquet",
            "*.npy",
            "*.npz",
        }

        self.assertTrue(required_patterns.issubset(patterns))

    def test_dockerfile_uses_narrow_copy_strategy(self):
        dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

        self.assertIn("COPY requirements.txt ./", dockerfile)
        self.assertIn("COPY scripts ./scripts", dockerfile)
        self.assertIn("COPY src ./src", dockerfile)
        self.assertNotIn("COPY . .", dockerfile)


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = create_checkpoint_dir(self.temp_dir)
    
    def test_end_to_end_export(self):
        """Test complete export pipeline end-to-end"""
        exporter = CactusModelExporter(output_dir=self.temp_dir)
        
        # Run full pipeline
        with patch("src.mobile.cactus_export.load_checkpoint", return_value=tiny_checkpoint_pair()):
            result = exporter.export(self.checkpoint_dir)
        
        # Verify success
        self.assertTrue(result["success"])
        
        # Verify constraints
        self.assertLess(result["model_size_mb"], 500)
        self.assertLess(result["benchmark_stats"]["mean_latency_ms"], 200)
        self.assertLess(result["benchmark_stats"]["memory_peak_mb"], 500)
        
        # Verify output files
        output_dir = Path(self.temp_dir)
        self.assertTrue((output_dir / "model_int4.bin").exists())
        self.assertTrue((output_dir / "config.json").exists())
        self.assertTrue((output_dir / "special_tokens.json").exists())
        self.assertTrue((output_dir / "export_manifest.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
