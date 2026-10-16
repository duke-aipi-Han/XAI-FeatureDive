"""Numeric checks for baseline-only hints, distinct from full ablation outcomes."""
import pytest
import torch
from backend.engine import InterpretabilityEngine
from backend.schemas import ExperimentRequest
from experiments.hint_examples import baseline_hints
from ml.interpretability import margins


@pytest.fixture(scope='module')
def engine():
    return InterpretabilityEngine()


def test_channel_hint_matches_small_gate_derivative(engine):
    request=ExperimentRequest(model_id='balanced',sample_id='fish:20708',background='blue',target_class='fish')
    hints=baseline_hints(engine,request)
    model=engine.models['balanced']; epsilon=.001; calls=0
    def gate_before_pool(module,args):
        nonlocal calls
        calls+=1
        if calls==1:
            value=args[0].clone();value[:,7]*=(1-epsilon)
            return (value,)
    hook=model.pool.register_forward_pre_hook(gate_before_pool)
    try:
        with torch.no_grad(): modified=float(margins(model(hints['rgb']),torch.tensor([1]))[0])
    finally:
        hook.remove()
    finite_difference=(modified-hints['margin'])/epsilon
    assert finite_difference==pytest.approx(float(hints['channels']['conv1']['estimated_margin_change'][7]),abs=.02)
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_erase_hint_matches_small_step_toward_existing_background(engine):
    request=ExperimentRequest(model_id='balanced',sample_id='airplane:72939',background='yellow',target_class='airplane')
    hints=baseline_hints(engine,request); epsilon=.001
    modified=hints['rgb'].clone()
    modified[:,:,14:18,4:8]+=epsilon*(hints['color'][None,:,None,None]-modified[:,:,14:18,4:8])
    with torch.no_grad(): margin=float(margins(engine.models['balanced'](modified),torch.tensor([2]))[0])
    finite_difference=(margin-hints['margin'])/epsilon
    assert finite_difference==pytest.approx(float(hints['erasure_delta_estimate'][14:18,4:8].sum()),abs=.02)
