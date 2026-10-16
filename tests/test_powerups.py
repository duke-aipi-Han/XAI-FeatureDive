"""Real-model hint API: original-input gradients, ranking, and request isolation."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from backend.engine import InterpretabilityEngine
from backend.main import create_app
from backend.schemas import ExperimentRequest, HintRequest
from experiments.hint_examples import baseline_hints
from ml.train import weights_sha256


@pytest.fixture(scope='module')
def engine():
    return InterpretabilityEngine()


def request(sample='fish:20708',target='fish',color='blue'):
    return HintRequest(model_id='balanced',sample_id=sample,target_class=target,background=color)


def test_channel_hints_match_independent_gradient_oracle_and_preserve_weights(engine):
    before={name:weights_sha256(model.state_dict()) for name,model in engine.models.items()}
    body=request();hints=engine.hints(body);oracle=baseline_hints(engine,body)
    assert hints.reference=='original_baseline' and hints.units=='logit_margin'
    assert hints.experiment==engine.infer(body)
    assert len(hints.channels)==24
    assert (hints.channels[0].layer,hints.channels[0].channel)==('conv1',7)
    for row in hints.channels:
        expected=oracle['channels'][row.layer]
        assert row.estimated_support==pytest.approx(-float(expected['estimated_margin_change'][row.channel]))
        assert row.mean_activation==pytest.approx(float(expected['mean_activity'][row.channel]))
        assert row.estimated_margin_change==-row.estimated_support
        assert row.rank==1+sum(other.estimated_support>row.estimated_support for other in hints.channels)
        if row.mean_activation==0: assert row.estimated_support==0
    assert before=={name:weights_sha256(model.state_dict()) for name,model in engine.models.items()}
    assert all(parameter.grad is None for model in engine.models.values() for parameter in model.parameters())


def test_erase_hint_uses_actual_stroke_direction_and_suggested_patch_matches_map(engine):
    body=request('airplane:72939','airplane','yellow');hints=engine.hints(body)
    oracle=baseline_hints(engine,body)
    np.testing.assert_allclose(hints.erase_support,-oracle['erasure_delta_estimate'],atol=1e-7)
    alpha=np.asarray(hints.experiment.input['source_alpha'])
    assert np.count_nonzero(np.asarray(hints.erase_support)[alpha==0])==0
    patch=hints.suggested_patch;box=patch.rectangle
    assert (box.x,box.y,box.width,box.height)==(4,14,4,4)
    assert patch.estimated_support==pytest.approx(np.asarray(hints.erase_support)[14:18,4:8].sum())
    actual=engine.infer(ExperimentRequest(**{**body.model_dump(),'interventions':[{'type':'drawing_erasure','region':{'rectangle':box.model_dump()}}]}))
    assert actual.comparison.classification_flip
    assert patch.estimated_margin_change!=actual.comparison.target_logit_margin_change


def test_hint_endpoint_rejects_modified_inputs_and_serves_real_estimates(engine):
    with TestClient(create_app(engine_factory=lambda config:engine)) as client:
        body=request().model_dump()
        response=client.post('/api/hints',json=body)
        assert response.status_code==200
        assert response.json()==engine.hints(request()).model_dump(mode='json')
        body['interventions']=[{'type':'channel_ablation','layer':'conv1','channel':7}]
        assert client.post('/api/hints',json=body).status_code==422
        with pytest.raises(ValueError,match='unmodified'):
            engine.hints(ExperimentRequest(**body))
        assert client.post('/api/hints',json={**request().model_dump(),'sample_id':'fish:9999999'}).status_code==404


def test_hints_are_target_specific_repeatable_and_safe_across_concurrent_requests(engine):
    fish=request();plane=request('airplane:72939','airplane','yellow')
    other=request(target='cat')
    expected={body.sample_id+body.target_class:engine.hints(body) for body in (fish,plane,other)}
    assert expected[fish.sample_id+'fish'].erase_support!=expected[other.sample_id+'cat'].erase_support
    bodies=[fish,plane,other,fish]
    with ThreadPoolExecutor(max_workers=2) as executor:
        results=list(executor.map(engine.hints,bodies))
    for body,result in zip(bodies,results):
        assert result==expected[body.sample_id+body.target_class]
