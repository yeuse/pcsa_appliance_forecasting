import sys
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from scripts.transfer_event_audit import event_report, predicted_previous, fusion_values, fusion_report, evaluate_audited
from scripts.run_pisa_cross_home_transfer import evaluate as legacy_evaluate


class EventAuditTests(unittest.TestCase):
    def test_real_evaluator_preserves_power_and_validation_only_trace(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.8))
            def forward(self, batch):
                power = torch.ones_like(batch['y_power']) * self.weight
                return {'y_power': power, 'p_on': power, 'amplitude_power': power,
                        'base_y_power': power * .5, 'past_power': power}
        model = Model()
        state = torch.tensor([[[0., 1., 1., 0., 0., 1.]]])
        batch = {'y_power': state, 'y_state': state,
                 'y_event': torch.tensor([[[0., 1., 0., 1., 0., 1.]]]),
                 'target_mask': torch.ones_like(state), 'y_prev_state': torch.zeros(1, 1)}
        args = (model, [batch], torch.device('cpu'), {'a': .5}, ['a'], False)
        old = legacy_evaluate(*args)
        new = evaluate_audited(*args, trace_fusion=True)
        self.assertEqual(set(old['flat']), set(new['flat']))
        for key, value in old['flat'].items():
            if math.isnan(value):
                self.assertTrue(math.isnan(new['flat'][key]))
            else:
                self.assertEqual(value, new['flat'][key], key)
        self.assertIn('fusion_audit', new)
        self.assertIn('event_audit', new)
        test_result = evaluate_audited(*args, trace_fusion=False)
        self.assertNotIn('fusion_audit', test_result)
        self.assertIsNone(model.weight.grad)
        self.assertAlmostEqual(model.weight.item(), .8)

    def report(self, pred, truth, true_prev=0, pred_prev=0, mask=None):
        p = torch.tensor([[pred]], dtype=torch.float32)
        t = torch.tensor([[truth]], dtype=torch.float32)
        m = torch.ones_like(t) if mask is None else torch.tensor([[mask]])
        return event_report(p, t, m, torch.tensor([[true_prev]]),
                            None if pred_prev is None else torch.tensor([[pred_prev]]), ['a'], 'fixture')

    def test_boundary_cannot_match_internal_event_with_tolerance(self):
        r = self.report([1, 1, 1], [0, 1, 1])
        scores = r['protocols']['internal/raw/start']
        self.assertEqual(scores['tolerance_2min']['a']['EventTP'], 0)
        self.assertEqual(scores['tolerance_2min']['a']['EventFN'], 1)

    def test_direction_and_adjacent_mask(self):
        r = self.report([0, 1, 0, 1], [0, 1, 0, 1], mask=[1, 1, 0, 1])
        self.assertEqual(r['protocols']['internal/raw/start']['exact']['a']['EventTP'], 1)
        self.assertEqual(r['protocols']['internal/raw/stop']['exact']['a']['EventTP'], 0)
        self.assertEqual(r['protocols']['internal/raw/start']['valid_transition_positions']['a'], 1)

    def test_true_history_is_separate_and_internal_invariant(self):
        first = self.report([0, 0, 0], [0, 0, 0], true_prev=1, pred_prev=0)
        second = self.report([0, 0, 0], [0, 0, 0], true_prev=1, pred_prev=1)
        self.assertEqual(first['protocols']['internal/raw/stop'], second['protocols']['internal/raw/stop'])
        self.assertEqual(first['protocols']['full_predicted_history/raw/stop']['exact']['a']['EventTP'], 0)
        self.assertEqual(first['protocols']['full_true_history_DIAGNOSTIC_ONLY/raw/stop']['exact']['a']['EventTP'], 1)

    def test_no_model_history_no_full_model_metric(self):
        r = self.report([0, 1], [0, 1], pred_prev=None)
        self.assertFalse(r['full_predicted_history_available'])
        self.assertFalse(any(k.startswith('full_predicted_history') for k in r['protocols']))
        value, source = predicted_previous({}, torch.tensor([.5]))
        self.assertIsNone(value)
        self.assertIn('unavailable', source)

    def test_actual_tcn_history_precedes_stale_event_field(self):
        state, source = predicted_previous({'future_tcn_input_power': torch.tensor([[[0., 1.]]]),
                                            'event_prev_power': torch.zeros(1, 1, 1)}, torch.tensor([.5]))
        self.assertEqual(state.item(), 1)
        self.assertEqual(source, 'future_tcn_input_power')

    def test_fusion_units_masks_and_empty_on(self):
        out = {'y_power': torch.tensor([[[2., 4.]]]), 'base_y_power': torch.ones(1, 1, 2),
               'future_tcn_power_gate': torch.tensor([[[.5]]]),
               'future_tcn_power_residual_raw': torch.tensor([[[2., 6.]]])}
        values = fusion_values(out)
        torch.testing.assert_close(values['gated_power_residual_raw'], torch.tensor([[[1., 3.]]]))
        report = fusion_report(values, torch.zeros(1, 1, 2), torch.zeros(1, 1, 2),
                               torch.tensor([[[1., 0.]]]), ['a'], SimpleNamespace())
        self.assertEqual(report['devices']['a']['all']['fields']['final_minus_base_power_kW']['mean'], 1)
        self.assertIsNone(report['devices']['a']['ON']['fields']['y_power'])


if __name__ == '__main__':
    unittest.main()
