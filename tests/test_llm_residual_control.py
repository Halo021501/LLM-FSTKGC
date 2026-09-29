"""Matched residual intervention checks, including saturation and gradients."""
import unittest
import torch
from src.model import LLMCandidateSidecar
from src.train import build_arg_parser


class ResidualControlTests(unittest.TestCase):
    def evaluate(self, module, mode="score"):
        base = torch.zeros(1, 4)
        ids = torch.tensor([[1, 2]])
        mask = torch.tensor([[True, False]])
        values = torch.ones(1, 2)
        return module(base, ids, mask, values, values, values, values, values, mode)[1]

    def test_none_default_removes_saturation_and_preserves_mask(self):
        module = LLMCandidateSidecar()
        module.scale_logit.data.fill_(2.0)
        bonus = self.evaluate(module)
        self.assertGreater(bonus[0, 1].item(), 0.35)
        self.assertEqual(torch.count_nonzero(bonus).item(), 1)
        bonus.sum().backward()
        self.assertGreater(module.scale_logit.grad.item(), 0)

    def test_hard_reproduces_saturated_value_and_gradient(self):
        module = LLMCandidateSidecar(residual_control="hard")
        module.scale_logit.data.fill_(2.0)
        bonus = self.evaluate(module)
        self.assertAlmostEqual(bonus[0, 1].item(), 0.35, places=6)
        bonus.sum().backward()
        self.assertEqual(module.scale_logit.grad.item(), 0)

    def test_untrained_scale_has_identical_results(self):
        hard = LLMCandidateSidecar(residual_control="hard")
        none = LLMCandidateSidecar(residual_control="none")
        none.load_state_dict(hard.state_dict(), strict=True)
        self.assertTrue(torch.equal(self.evaluate(hard), self.evaluate(none)))

    def test_candidate_remains_zero_even_with_large_scale(self):
        module = LLMCandidateSidecar(score_scale=20)
        self.assertEqual(self.evaluate(module, "candidate").abs().sum().item(), 0)

    def test_parser_defaults_to_requested_intervention(self):
        self.assertEqual(build_arg_parser().parse_args([]).llm_residual_control, "none")
        self.assertEqual(build_arg_parser().parse_args(["--llm-residual-control", "hard"]).llm_residual_control, "hard")

    def test_invalid_control_fails(self):
        with self.assertRaises(ValueError):
            LLMCandidateSidecar(residual_control="invalid")


if __name__ == "__main__":
    unittest.main()
