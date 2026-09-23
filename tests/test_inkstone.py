"""Automated verification suite for the Intern InkStone SDK and live APIs."""

import unittest
import sys
from pathlib import Path

# Add package root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inkstone import (
    InkStoneClient,
    InkStoneConfig,
    AuthManager,
    TokenPlanManager,
    InferenceClient,
    ComputeManager,
    ModelDeployer,
)


class TestInkStoneSDK(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = InkStoneClient()

    def test_01_config_and_auth(self):
        """Test config loading and JWT introspection."""
        config = self.client.config
        self.assertTrue(config.uaa_token, "uaa_token must be present")
        self.assertTrue(config.api_key, "api_key must be present")

        auth_status = self.client.auth.get_token_status()
        self.assertTrue(auth_status["valid"], "JWT must be valid and unexpired")
        self.assertEqual(auth_status["issuer"], "OpenXLab")
        self.assertGreater(auth_status["remaining_seconds"], 0)

    def test_02_tokenplan_balance(self):
        """Test live credit balance query from TokenPlan API."""
        balance = self.client.tokenplan.get_balance()
        self.assertIn("available_credits", balance)
        self.assertGreaterEqual(balance["available_credits"], 0.0)
        self.assertGreater(balance["approx_tokens"], 0)
        self.assertIn("usage_windows", balance)
        self.assertIn("5h", balance["usage_windows"])
        self.assertIn("7d", balance["usage_windows"])

    def test_03_tokenplan_models(self):
        """Test live model catalog retrieval."""
        models = self.client.tokenplan.list_models()
        self.assertGreaterEqual(len(models), 10, "Should have at least 10 models in catalog")
        model_ids = [m["id"] for m in models]
        self.assertIn("qwen3.8-27b", model_ids)
        self.assertIn("deepseek-v4-pro-0813", model_ids)
        self.assertIn("glm-5.3", model_ids)
        self.assertIn("kimi-k2.6", model_ids)

    def test_04_tokenplan_keys(self):
        """Test API key listing."""
        keys = self.client.tokenplan.list_keys()
        self.assertIsInstance(keys, list)

    def test_05_inference_live(self):
        """Test real chat completion call to hosted model."""
        res = self.client.chat("Ping test. Reply with one word 'PONG'.", model="qwen3.8-27b", max_tokens=60)
        self.assertIn("content", res)
        # Verify that either answer content or reasoning output was generated
        has_output = bool(res["content"].strip() or res.get("reasoning_content", "").strip())
        self.assertTrue(has_output, "Model must return non-empty output or reasoning")
        self.assertIn("usage", res)
        self.assertGreater(res["usage"].get("total_tokens", 0), 0)

    def test_06_compute_user_resource(self):
        """Test compute points query."""
        compute_info = self.client.compute.get_user_resource()
        self.assertGreaterEqual(compute_info["remaining_points"], 0)
        self.assertGreaterEqual(compute_info["total_points"], 5000)
        self.assertGreaterEqual(compute_info["a100_hours_remaining"], 80.0)

    def test_07_compute_flavors_and_mirrors(self):
        """Test compute hardware flavors and container mirrors query."""
        flavors = self.client.compute.list_flavors()
        self.assertTrue(len(flavors) >= 3, "Expected at least A100, 910B, and CPU flavors")

        mirrors = self.client.compute.list_mirrors()
        self.assertTrue(len(mirrors) >= 3, "Expected official container image mirrors")

    def test_08_deploy_recipe_generation(self):
        """Test script generation for orcarouter/Qwen3.8-27B-Uncensored."""
        script = self.client.deployer.generate_setup_script("qwen3.8-uncensored")
        self.assertIn("orcarouter/Qwen3.8-27B-Uncensored", script)
        self.assertIn("https://hf-mirror.com", script)
        self.assertIn("vllm.entrypoints.openai.api_server", script)
        self.assertIn("/data/models/Qwen3.8-27B-Uncensored", script)


if __name__ == "__main__":
    unittest.main()
