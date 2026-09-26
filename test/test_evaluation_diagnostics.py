"""Evaluation summaries retain reasons and use per-episode action ratios."""

import pytest

from turtleboot3_autonomous_nav import dqn_trainer


def test_evaluation_summary_preserves_each_episode_and_its_turn_ratio():
    summary = dqn_trainer.summarize_evaluations([
        dqn_trainer.EpisodeDiagnostics(
            'target_coverage', 0.98, 0.97, 1, 10, 2, 15, True),
        dqn_trainer.EpisodeDiagnostics(
            'time_limit', 0.80, 0.78, 3, 20, 10, 16, True),
    ])
    assert summary['mean_coverage'] == pytest.approx(0.89)
    assert summary['mean_grid_coverage'] == pytest.approx(0.875)
    assert summary['mean_interventions'] == pytest.approx(2.0)
    assert summary['mean_turn_ratio'] == pytest.approx(0.35)
    assert summary['episodes'] == [
        {'episode': 15, 'is_evaluation': True,
         'reason': 'target_coverage', 'reachable_coverage': 0.98,
         'grid_coverage': 0.97, 'interventions': 1, 'policy_actions': 10,
         'turn_actions': 2, 'turn_ratio': 0.2},
        {'episode': 16, 'is_evaluation': True,
         'reason': 'time_limit', 'reachable_coverage': 0.80,
         'grid_coverage': 0.78, 'interventions': 3, 'policy_actions': 20,
         'turn_actions': 10, 'turn_ratio': 0.5},
    ]


def test_episode_without_actions_has_zero_turn_ratio():
    diagnostic = dqn_trainer.EpisodeDiagnostics('time_limit', 0.0, 0.0, 0, 0, 0)
    assert diagnostic.as_dict()['turn_ratio'] == 0.0


def test_empty_evaluation_batch_cannot_select_a_checkpoint():
    with pytest.raises(ValueError, match='at least one'):
        dqn_trainer.summarize_evaluations([])
