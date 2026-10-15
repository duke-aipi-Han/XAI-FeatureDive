"""Actual-checkpoint engine and API validation, replay, and isolation checks."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.catalog import DEFAULT_CONFIG
from backend.engine import InterpretabilityEngine, region_mask
from backend.main import create_app
from backend.schemas import ActivationRequest, ExperimentRequest, Region
from ml.backgrounds import RGB
from ml.common import ROOT, read_json, write_json
from ml.train import weights_sha256


@pytest.fixture(scope='module')
def engine():
    return InterpretabilityEngine()


@pytest.fixture
def base(engine):
    sample = next(iter(engine.samples.values()))
    return dict(model_id='balanced', sample_id=sample['id'], background='green', target_class=sample['class'])


@pytest.fixture
def client(engine):
    with TestClient(create_app(engine_factory=lambda config: engine)) as client:
        yield client


def test_models_and_demo_catalog_are_verified_and_test_sources_are_excluded(engine):
    assert set(engine.models) == {'balanced', 'shortcut'}
    assert len(engine.samples) == 24
    assert all(row['split'] == 'train' for row in engine.samples.values())
    for model in engine.models.values():
        assert not model.training
        assert all(parameter.device.type == 'cpu' and not parameter.requires_grad for parameter in model.parameters())


def test_background_swap_preserves_alpha_and_has_independent_expected_rgb(engine, base):
    original = engine.render(ExperimentRequest(**base))
    changed = engine.render(ExperimentRequest(**base, interventions=[{'type':'background_swap','background':'yellow'}]))
    assert torch.equal(original.source, changed.source)
    assert torch.equal(original.alpha, changed.alpha)
    color_delta = (torch.tensor(RGB[2]) - torch.tensor(RGB[0])).float()[None,:,None,None] / 255
    torch.testing.assert_close(changed.rgb - original.rgb, (1-original.alpha[None,None]) * color_delta, atol=1e-7, rtol=1e-5)


def test_drawing_erasure_only_removes_selected_foreground(engine, base):
    # Erase all foreground: background should be exactly the original color.
    state = ExperimentRequest(**base, interventions=[{'type':'drawing_erasure','region':{'rectangle':{'x':0,'y':0,'width':28,'height':28}}}])
    result = engine.infer(state)
    original = engine.render(ExperimentRequest(**base))
    erased = engine.render(state)
    assert torch.count_nonzero(erased.alpha) == 0
    background = torch.tensor(RGB[0]).float()[None,:,None,None].expand(1,3,28,28) / 255
    torch.testing.assert_close(erased.rgb, background, atol=0, rtol=0)
    assert result.intervention_log[0]['affected_pixel_count'] == int((original.alpha > 0).sum())
    torch.testing.assert_close(erased.rgb[:,:,original.alpha==0], original.rgb[:,:,original.alpha==0], atol=0, rtol=0)


def test_occlusion_changes_exactly_the_region_and_retains_fixed_policy_after_swap(engine, base):
    region={'rectangle':{'x':12,'y':12,'width':4,'height':4}}
    state=ExperimentRequest(**base,interventions=[{'type':'controlled_occlusion','region':region}, {'type':'background_swap','background':'blue'}])
    changed=engine.render(state)
    comparison=engine.render(ExperimentRequest(**{**base,'background':'blue'}))
    mask=region_mask(Region.model_validate(region))
    expected=torch.full((1,3,16),128/255)
    torch.testing.assert_close(changed.rgb[:,:,mask],expected,atol=0,rtol=0)
    torch.testing.assert_close(changed.rgb[:,:,~mask],comparison.rgb[:,:,~mask],atol=0,rtol=0)
    assert changed.log[0]['affected_pixel_count']==16
    assert changed.log[0]['mask_policy']=='fixed_rgb_full_region'


@pytest.mark.parametrize('layer',['conv1','conv2'])
def test_api_engine_ablation_injection_and_independent_downstream_scores(engine,base,layer):
    baseline=engine.activations(ActivationRequest(**base,layer=layer))
    channel=int(np.argmax(baseline.baseline_mean_by_channel))
    body=ActivationRequest(**base,layer=layer,interventions=[{'type':'channel_ablation','layer':layer,'channel':channel}])
    result=engine.activations(body)
    original=torch.tensor(result.baseline_values)
    changed=torch.tensor(result.modified_values)
    assert torch.count_nonzero(original[channel])>0
    assert torch.count_nonzero(changed[channel])==0
    others=[index for index in range(len(original)) if index!=channel]
    torch.testing.assert_close(changed[others],original[others],atol=0,rtol=0)
    model=engine.models[base['model_id']]
    value=original.unsqueeze(0).clone()
    value[:,channel]=0
    value=model.pool(value)
    if layer=='conv1':
        value=model.pool(torch.relu(model.conv2(value)))
    expected=model.output(torch.relu(model.dense(value.flatten(1))))
    torch.testing.assert_close(torch.tensor(result.experiment.modified.logits),expected[0],atol=0,rtol=0)
    assert result.experiment.modified.logits!=result.experiment.baseline.logits
    assert result.experiment.intervention_log[0]['injection_point']=='post_relu_before_pool'


def test_saliency_matches_class_logit_gradients_including_current_ablation(engine,base):
    request=ExperimentRequest(**base,interventions=[{'type':'channel_ablation','layer':'conv2','channel':7}])
    result=engine.saliency(request)
    rendered=engine.render(request)
    value=rendered.rgb.clone().requires_grad_(True)
    logits=engine.models['balanced'](value,ablation=rendered.ablation)
    target=['cat','fish','airplane'].index(base['target_class'])
    gradient,=torch.autograd.grad(logits[0,target],value)
    expected=gradient.abs().sum(1)[0]
    torch.testing.assert_close(torch.tensor(result.modified_scores),expected,atol=0,rtol=0)
    assert expected.shape==(28,28) and bool(torch.isfinite(expected).all())
    assert float(expected.sum())>0
    assert all(parameter.grad is None for model in engine.models.values() for parameter in model.parameters())


def test_replay_reset_undo_baseline_and_probability_point_units(engine,base):
    baseline=engine.infer(ExperimentRequest(**base))
    first={'type':'background_swap','background':'blue'}
    second={'type':'controlled_occlusion','region':{'rectangle':{'x':8,'y':8,'width':8,'height':8}}}
    intermediate=engine.infer(ExperimentRequest(**base,interventions=[first]))
    current=engine.infer(ExperimentRequest(**base,interventions=[first,second]))
    assert current.baseline==intermediate.baseline==baseline.baseline
    assert engine.infer(current.request)==current
    assert engine.infer(ExperimentRequest(**base,interventions=[first]))==intermediate
    assert engine.infer(ExperimentRequest(**base))==baseline
    target=['cat','fish','airplane'].index(base['target_class'])
    assert current.comparison.target_probability_point_change==pytest.approx(100*(current.modified.probabilities[target]-current.baseline.probabilities[target]))
    assert current.comparison.target_logit_margin_change==pytest.approx(current.modified.target_logit_margin-current.baseline.target_logit_margin)
    assert current.experiment_id!=baseline.experiment_id


def test_concurrent_calls_do_not_contaminate_models_or_source(engine,base):
    before={name:weights_sha256(model.state_dict()) for name,model in engine.models.items()}
    clean=ExperimentRequest(**base)
    modified=ExperimentRequest(**base,interventions=[{'type':'channel_ablation','layer':'conv1','channel':2}])
    expected=engine.infer(clean)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results=list(executor.map(lambda index: engine.saliency(modified) if index%2 else engine.infer(clean),range(12)))
    assert all(result==expected for result in results[::2])
    assert engine.infer(clean)==expected
    assert before=={name:weights_sha256(model.state_dict()) for name,model in engine.models.items()}


def test_all_endpoints_use_real_engine(client,base,engine):
    assert client.get('/api/health').json()=={'status':'ok'}
    assert client.get('/api/ready').json()['inference_device']=='cpu'
    assert len(client.get('/api/samples').json()['samples'])==24
    assert len(client.get('/api/models').json()['models'])==2
    for route in ('infer','intervene'):
        response=client.post(f'/api/{route}',json=base)
        assert response.status_code==200
        assert response.json()==engine.infer(ExperimentRequest(**base)).model_dump(mode='json')
    assert client.post('/api/activations',json={**base,'layer':'dense'}).status_code==200
    assert client.post('/api/saliency',json=base).status_code==200
    assert '/api/intervene' in client.get('/openapi.json').json()['paths']


@pytest.mark.parametrize('change',[
    {'model_id':'unregistered'}, {'sample_id':'../secret'}, {'target_class':'dog'}, {'background':'red'},
    {'inputs':[[[0]]]}, {'checkpoint_url':'https://example.com/model.pt'}, {'code':'print(1)'},
    {'interventions':[{'type':'channel_ablation','layer':'conv1','channel':8}]},
    {'interventions':[{'type':'channel_ablation','layer':'conv2','channel':-1}]},
    {'interventions':[{'type':'channel_ablation','layer':'dense','channel':0}]},
    {'interventions':[{'type':'channel_ablation','layer':'conv1','channel':True}]},
    {'interventions':[{'type':'drawing_erasure','region':{'pixels':[{'x':28,'y':0}]}}]},
    {'interventions':[{'type':'drawing_erasure','region':{'pixels':[{'x':1.5,'y':0}]}}]},
    {'interventions':[{'type':'drawing_erasure','region':{'pixels':[{'x':1,'y':0},{'x':1,'y':0}]}}]},
    {'interventions':[{'type':'drawing_erasure','region':{}}]},
    {'interventions':[{'type':'controlled_occlusion','region':{'rectangle':{'x':27,'y':0,'width':2,'height':1}}}]},
    {'interventions':[{'type':'controlled_occlusion','region':{'pixels':[{'x':0,'y':0}]},'replacement_rgb':[256,0,0]}]},
    {'interventions':[{'type':'channel_ablation','layer':'conv1','channel':0}]*2},
    {'interventions':[{'type':'background_swap','background':'gray'}]*17},
])
def test_invalid_requests_are_rejected(client,base,change):
    assert client.post('/api/intervene',json={**base,**change}).status_code==422


def test_unknown_but_valid_sample_id_is_404(client,base):
    assert client.post('/api/infer',json={**base,'sample_id':'cat:9999999'}).status_code==404


def test_failed_startup_keeps_liveness_but_not_readiness():
    def broken(config):
        raise ValueError('internal detail must not be exposed')
    with TestClient(create_app(engine_factory=broken)) as client:
        assert client.get('/api/health').status_code==200
        response=client.get('/api/ready')
        assert response.status_code==503 and 'internal detail' not in response.text


def test_payload_and_frequency_limits(tmp_path,engine,base):
    config=read_json(DEFAULT_CONFIG)
    config.update(max_body_bytes=400,requests_per_minute=2)
    path=tmp_path/'limits.json'
    write_json(path,config)
    with TestClient(create_app(path,engine_factory=lambda config:engine)) as client:
        assert client.post('/api/infer',content=' '*401).status_code==413
        assert client.post('/api/infer',json=base).status_code==200
        response=client.post('/api/infer',json=base)
        assert response.status_code==429 and int(response.headers['retry-after'])>0
        assert client.get('/api/ready').status_code==200


def test_catalog_or_registry_tampering_blocks_startup(tmp_path):
    config=read_json(DEFAULT_CONFIG)
    registry=read_json(ROOT/config['registry'])
    registry['background_rgb'][0][0]+=1
    registry_path=tmp_path/'registry.json'
    write_json(registry_path,registry)
    config['registry']=str(registry_path)
    path=tmp_path/'config.json'
    write_json(path,config)
    with pytest.raises(ValueError,match='input contract'):
        InterpretabilityEngine(path)


def test_inactive_channel_ablation_can_have_a_genuine_null_effect(engine,base):
    baseline=engine.activations(ActivationRequest(**base,layer='conv1'))
    inactive=next(index for index,mean in enumerate(baseline.baseline_mean_by_channel) if mean==0)
    response=engine.infer(ExperimentRequest(**base,interventions=[{'type':'channel_ablation','layer':'conv1','channel':inactive}]))
    assert response.modified==response.baseline
    assert response.comparison.target_logit_margin_change==0


def test_erasure_after_occlusion_and_overlapping_occlusions_are_replayable(engine,base):
    source=engine.render(ExperimentRequest(**base))
    y,x=(source.alpha>0).nonzero()[0].tolist()
    region={'pixels':[{'x':x,'y':y}]}
    operations=[{'type':'controlled_occlusion','region':region,'replacement_rgb':[255,0,0]},
                {'type':'controlled_occlusion','region':region,'replacement_rgb':[0,0,255]}]
    rendered=engine.render(ExperimentRequest(**base,interventions=operations))
    torch.testing.assert_close(rendered.rgb[0,:,y,x],torch.tensor([0.,0.,1.]),atol=0,rtol=0)
    operations += [{'type':'drawing_erasure','region':region},{'type':'background_swap','background':'yellow'}]
    rendered=engine.render(ExperimentRequest(**base,interventions=operations))
    assert rendered.alpha[y,x]==0
    torch.testing.assert_close(rendered.rgb[0,:,y,x],torch.tensor(RGB[2]).float()/255,atol=0,rtol=0)


def test_streamed_oversize_body_is_rejected_without_content_length():
    import asyncio
    from backend.limits import RequestLimits
    sent=[]
    messages=iter([{'type':'http.request','body':b'a'*200,'more_body':True},
                   {'type':'http.request','body':b'b'*200,'more_body':False}])
    async def receive():
        return next(messages)
    async def send(message):
        sent.append(message)
    async def should_not_run(scope,receive,send):
        raise AssertionError('Oversize body reached the application')
    middleware=RequestLimits(should_not_run,max_body_bytes=300)
    asyncio.run(middleware({'type':'http','path':'/api/infer','headers':[],'client':('test',1)},receive,send))
    assert sent[0]['status']==413
