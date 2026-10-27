"""Checks for the portable source-training configurations."""

from __future__ import annotations

import json
import sys
import unittest
from unittest.mock import patch

from scripts import train_home7951
from scripts.train_pcsa_from_config import ROOT, build_command


class SourceConfigRunnerTests(unittest.TestCase):
    def read_config(self, name: str) -> dict:
        path = ROOT / "configs" / "observed" / name
        return json.loads(path.read_text(encoding="utf-8"))

    def test_base_run_starts_without_checkpoint(self) -> None:
        config = self.read_config("pcsa_source_base_seed42_config.json")
        command = build_command(config, python="python")
        self.assertNotIn("--init_checkpoint", command)
        self.assertEqual(command[command.index("--train_mode") + 1] if "--train_mode" in command else "full", "full")
        self.assertEqual(command[command.index("--batch_size") + 1], "256")
        self.assertEqual(command[-1], "7951_base_auxlight_s42")

    def test_residual_run_uses_saved_base_history(self) -> None:
        config = self.read_config("pcsa_source_seed7_config.json")
        command = build_command(config, python="python")
        self.assertEqual(command[command.index("--train_mode") + 1], "future_residual_tcn")
        self.assertEqual(command[command.index("--residual_tcn_history_source") + 1], "base")
        self.assertEqual(command[command.index("--seed") + 1], "7")
        self.assertEqual(
            command[command.index("--init_checkpoint") + 1],
            "outputs/runs/7951_history_auxlight_fixed_s42/checkpoints/best.pt",
        )
        self.assertTrue(all("/home/dell/" not in value for value in command))

    def test_inconsistent_output_paths_are_rejected(self) -> None:
        config = self.read_config("pcsa_source_seed7_config.json")
        config["checkpoint_dir"] = "outputs/unrelated/checkpoints"
        with self.assertRaisesRegex(ValueError, "checkpoint_dir"):
            build_command(config, python="python")

    def test_all_recorded_stages_parse_with_the_training_cli(self) -> None:
        for name in (
            "pcsa_source_base_seed42_config.json",
            "pcsa_source_history_seed42_config.json",
            "pcsa_source_seed7_config.json",
        ):
            with self.subTest(config=name):
                command = build_command(self.read_config(name), python="python")
                with patch.object(sys, "argv", command[1:]):
                    parsed = train_home7951.parse_args()
                self.assertEqual(parsed.csv_path, "data/austin_2018_sep_4homes/home_7951_2018_sep_1min.csv")


if __name__ == "__main__":
    unittest.main()
