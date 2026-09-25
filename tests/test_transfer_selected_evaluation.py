import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts import evaluate_transfer_selected as ev


class SelectedEvaluationTests(unittest.TestCase):
    def test_checkpoint_selection(self):
        run = Path('/run')
        source = {'source_checkpoint': 'source.pt'}
        for label in ('source_zero_shot', 'support_calibrated_source'):
            self.assertEqual(ev.selected_path(run, 'Seq2Seq-NILM -> TCN',
                             {'selected_model': label}, source, {}, '0.010'), Path('source.pt').resolve())
        self.assertEqual(ev.selected_path(run, 'Aggregate-to-appliance TCN',
                         {'selected_model': 'few_shot_checkpoint'}, source, {}, '0.100'),
                         run / 'checkpoints/aggregate_tcn/fewshot_0.100/best.pt')
        self.assertEqual(ev.selected_path(run, 'PISA',
                         {'selected_model': 'two_stage_selected_checkpoint', 'selected_checkpoint': 'selected.pt'},
                         {}, {}, '0.100'), Path('selected.pt').resolve())
        with self.assertRaises(ValueError):
            ev.selected_path(run, 'PISA', {'selected_model': 'unknown'}, {}, {}, '0.100')

    def test_validation_guard(self):
        ev.validate_mae(.1, .100001, 5e-5)
        for actual, expected in ((.2, .1), (float('nan'), .1), (.1, None)):
            with self.assertRaises(ValueError):
                ev.validate_mae(actual, expected, 5e-5)

    def test_caps_and_frozen_weights(self):
        for selection, expected_calls in [('source_zero_shot', 0), ('support_calibrated_source', 1)]:
            model = torch.nn.Linear(1, 1)
            job = dict(source_config={}, config={}, method='Seq2Seq-NILM -> TCN',
                       checkpoint=Path('source.pt'), item={'selected_model': selection,
                       'power_cap_calibration': {'calibrated_caps_kw': [1, 2, 3, 4]}})
            with patch.object(ev, 'build_models', return_value=(torch.nn.Linear(1, 1), model)), \
                 patch.object(ev, 'load_checkpoint_strict'), patch.object(ev, 'apply_caps') as caps:
                result = ev.make_model(job)
                self.assertEqual(caps.call_count, expected_calls)
                self.assertFalse(any(p.requires_grad for p in result.parameters()))

    def test_strict_checkpoint_and_real_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            model = torch.nn.Linear(2, 1)
            torch.save({'model_state_dict': model.state_dict()}, path)
            other = torch.nn.Linear(2, 1)
            ev.load_checkpoint_strict(other, path)
            torch.testing.assert_close(model.weight, other.weight)
            with self.assertRaises(RuntimeError):
                ev.load_checkpoint_strict(torch.nn.Linear(3, 1), path)

    def test_main_exports_and_aborts_before_test_on_mismatch(self):
        for mismatch in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                checkpoint = root / 'selected.pt'
                checkpoint.write_bytes(b'fixture checkpoint')
                data = root / 'data.csv'
                data.write_text('fixture data')
                item = dict(selected_model='two_stage_selected_checkpoint', selected_validation_mae=.1)
                job = dict(group='PISA', method='PISA', key='0.100', run=str(root), item=item,
                           config={}, source_config={}, checkpoint=checkpoint)
                plan = [('3039', (data, data, 120, 30, 1, ['mains'], ['air1']), {'air1': .05}, [job])]
                calls = []
                def fake_evaluate(model, loader, *args, **kwargs):
                    calls.append(loader)
                    return {'nested': {}, 'flat': {'regression/macro_avg/MAE': .2 if mismatch else .1}}
                output = root / 'result'
                with patch.object(sys, 'argv', ['evaluate', '--output_dir', str(output), '--device', 'cpu']), \
                     patch.object(ev, 'prepare', return_value=plan), \
                     patch.object(ev, 'build_single_home_datasets', return_value=({'val': [1], 'test': [2]}, None)), \
                     patch.object(ev, 'build_loaders', return_value={'val': 'val', 'test': 'test'}), \
                     patch.object(ev, 'make_model', return_value=torch.nn.Linear(1, 1)), \
                     patch.object(ev, 'evaluate', side_effect=fake_evaluate):
                    if mismatch:
                        with self.assertRaises(ValueError):
                            ev.main()
                        self.assertEqual(calls, ['val'])
                        self.assertFalse((output / 'COMPLETE.json').exists())
                    else:
                        ev.main()
                        self.assertEqual(calls, ['val', 'test'])
                        self.assertTrue((output / 'comparison.csv').is_file())
                        self.assertTrue((output / 'metrics_long.csv').is_file())
                        self.assertEqual(json.loads((output / 'COMPLETE.json').read_text())['models'], 1)
                self.assertEqual(checkpoint.read_bytes(), b'fixture checkpoint')


if __name__ == '__main__':
    unittest.main()
