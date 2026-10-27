"""Regression tests: python -m unittest discover -s tests -v."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from src.data.scaler import ApplianceScaler
from src.data.pecanstreet_dataset import summarize_on_support
from src.checkpoint_selection import on_safety_status, monitor_improved
from src.training.losses import PISALoss, PISALossConfig
from src.training.trainer import PISATrainer, TrainerConfig


class ScriptedTrainer(PISATrainer):
    def train_one_epoch(self, loader, epoch):
        self.test_epoch = epoch
        return {"loss_total": 0.1, "lr": 0.001}

    def validate(self, loader, **kwargs):
        mae, on, coverage = self.sequence[self.test_epoch - 1]
        return {"loss_total": mae, "regression/macro_avg/MAE": mae,
                "domain/worst_home_appliance_normalized_on_power_MAE": on,
                "domain/on_pair_coverage": coverage}


def scripted(path, sequence):
    model = torch.nn.Linear(1, 1)
    trainer = ScriptedTrainer(model, torch.nn.L1Loss(),
        torch.optim.Adam(model.parameters()),
        TrainerConfig(device="cpu", use_amp=False, checkpoint_dir=str(path),
            max_epochs=len(sequence), monitor="val/regression/macro_avg/MAE",
            source_on_safety_scope="home_appliance", source_on_safety_ceiling=0.9))
    trainer.sequence = sequence
    return trainer


class TrainingRepairTests(unittest.TestCase):
    def test_shared_selection_rejects_missing_and_preserves_min_delta(self):
        self.assertFalse(on_safety_status({}, .9, "home_appliance")[0])
        metrics = {"val/domain/worst_home_appliance_normalized_on_power_MAE": None,
                   "val/domain/on_pair_coverage": 1.}
        self.assertFalse(on_safety_status(metrics, .9, "home_appliance")[0])
        self.assertFalse(monitor_improved(.0999999, .1, "min", 1e-6))
        self.assertTrue(monitor_improved(.09, .1, "min", 1e-6))

    def test_support_distinguishes_repeated_windows_from_unique_on_steps(self):
        ds = SimpleNamespace(indices=[0, 1, 2], input_window=1, horizon=3,
            target_mode="future", state=np.array([[0], [0], [1], [0], [0], [0]]),
            target_available=np.ones((6, 1)), bundle=SimpleNamespace(appliance_cols=["a"]), split="val")
        support = summarize_on_support(ds, 2)
        self.assertEqual(support["appliances"]["a"]["on_windows"], 2)
        self.assertEqual(support["appliances"]["a"]["on_unique_steps"], 1)
        self.assertTrue(support["all_supported"])
        ds.target_available[2, 0] = 0
        self.assertFalse(summarize_on_support(ds, 1)["all_supported"])

    def test_on_scaler_excludes_standby_unavailable_and_serializes(self):
        frame = pd.DataFrame({"a": [0.001] * 1000 + [1., 2., 100.],
                              "a_available": [1] * 1002 + [0],
                              "b": [0.] * 1003})
        scaler = ApplianceScaler("p99_on").fit(frame, ["a", "b"], {"a": .1, "b": .2})
        np.testing.assert_allclose(scaler.scale_, [1.99, .2])
        self.assertEqual(scaler.fit_audit_["a"]["on_train_samples"], 2)
        self.assertEqual(scaler.fit_audit_["b"]["fallback"], "training_state_threshold")
        restored = ApplianceScaler.from_state_dict(scaler.state_dict())
        np.testing.assert_equal(restored.scale_, scaler.scale_)
        self.assertEqual(restored.fit_audit_, scaler.fit_audit_)
        with self.assertRaises(ValueError):
            ApplianceScaler("p99_on").fit(frame, ["a"])

    def test_balanced_loss_state_counts_mask_and_backward(self):
        loss_fn = PISALoss([1., 1.], PISALossConfig(balanced_on_fraction=.35))
        pred = torch.tensor([[[2., 4., 4.], [6., 8., 8.]]], requires_grad=True)
        target = torch.zeros_like(pred)
        state = torch.tensor([[[1., 0., 0.], [1., 0., 0.]]])
        def value(p, t, s, mask=None):
            return loss_fn._power_loss(p, t, mask, "l1", s, balanced_on_off=True)
        result = value(pred, target, state)
        self.assertAlmostEqual(result.item(), (.35*2 + .65*4 + .35*6 + .65*8)/2, places=6)
        result.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertTrue((pred.grad.abs().sum((0, 2)) > 0).all())
        self.assertAlmostEqual(value(pred[:, :, :2], target[:, :, :2], state[:, :, :2]).item(), result.item(), places=6)
        pred.grad = None
        zero = value(pred, target, state, torch.zeros_like(pred))
        zero.backward()
        self.assertEqual(zero.item(), 0.)
        self.assertEqual(pred.grad.abs().sum().item(), 0.)

    def test_safety_selection_history_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = scripted(folder, [(0.05, 1.1, 1.), (.2, .8, 1.), (.15, .85, 1.), (.1, .7, .9)])
            trainer.fit([], [])
            best = torch.load(Path(folder)/"best.pt", map_location="cpu")
            last = torch.load(Path(folder)/"last.pt", map_location="cpu")
            self.assertEqual(best["epoch"], 3)
            self.assertEqual(len(best["history"]), 3)
            self.assertEqual(last["epoch"], 4)
            self.assertEqual(len(last["history"]), 4)
            self.assertEqual(last["history"][-1]["checkpoint/source_on_safety_eligible"], 0.)
            self.assertEqual(json.loads((Path(folder)/"selection_status.json").read_text())["status"], "selected")
            with self.assertRaises(FileExistsError):
                trainer.fit([], [])

    def test_no_safe_checkpoint_is_explicit_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = scripted(folder, [(.1, 1.1, 1.), (.05, .8, .9)])
            with self.assertRaisesRegex(RuntimeError, "No eligible checkpoint"):
                trainer.fit([], [])
            self.assertFalse((Path(folder)/"best.pt").exists())
            self.assertTrue((Path(folder)/"last.pt").exists())
            self.assertEqual(json.loads((Path(folder)/"selection_status.json").read_text())["status"], "no_eligible_checkpoint")

    def test_home_average_cannot_hide_failed_appliance(self):
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.zeros(1))
            def forward(self, batch):
                return {"y_power": batch["prediction"] + self.p * 0}
        class FakeLoss(torch.nn.Module):
            def forward(self, out, batch):
                loss = (out["y_power"] - batch["y_power"]).abs().mean()
                return loss, {"loss_total": loss.detach()}
        with tempfile.TemporaryDirectory() as folder:
            model = FakeModel()
            trainer = PISATrainer(model, FakeLoss(), torch.optim.Adam(model.parameters()),
                TrainerConfig(device="cpu", use_amp=False, checkpoint_dir=folder, source_on_min_windows=1),
                appliance_names=["large", "small"])
            batch = {"x_hist": torch.zeros(2, 3, 1), "home_index": torch.tensor([0, 1]),
                     "y_power": torch.tensor([[[10.], [1.]], [[10.], [1.]]]),
                     "prediction": torch.tensor([[[10.], [0.]], [[10.], [0.]]]),
                     "y_state": torch.ones(2, 2, 1)}
            stats = trainer.validate([batch], compute_metrics=False)
            self.assertLess(stats["domain/worst_home_normalized_on_power_MAE"], .1)
            self.assertEqual(stats["domain/worst_home_appliance_normalized_on_power_MAE"], 1.)
            self.assertEqual(stats["domain/on_pair_coverage"], 1.)
            off_batch = dict(batch)
            off_batch["y_state"] = torch.zeros_like(batch["y_state"])
            training = trainer.train_one_epoch([batch, off_batch], 1)
            self.assertEqual(training["supervision/small/on_batches"], 1.)
            self.assertEqual(training["supervision/small/no_on_batches"], 1.)
            self.assertEqual(training["supervision/small/off_batches"], 1.)


if __name__ == "__main__":
    unittest.main()
