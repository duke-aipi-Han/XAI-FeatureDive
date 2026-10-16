"""Local CPU API: uvicorn backend.main:app --host 127.0.0.1 --port 8000."""
from contextlib import asynccontextmanager
import logging
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request

from backend.catalog import DEFAULT_CONFIG
from backend.engine import InterpretabilityEngine
from backend.limits import RequestLimits
from backend.schemas import (
    ActivationRequest, ActivationResponse, ExperimentRequest,
    ExperimentResponse, SaliencyResponse, HintRequest, HintResponse,
)
from ml.common import read_json


def create_app(config_path: Path = DEFAULT_CONFIG, engine_factory=InterpretabilityEngine) -> FastAPI:
    config = read_json(config_path)

    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = None
        try:
            app.state.engine = engine_factory(config_path)
        except (OSError, ValueError, KeyError, RuntimeError, TypeError):
            logging.getLogger(__name__).exception('FeatureDive model/catalog initialization failed')
        yield
        app.state.engine = None

    app = FastAPI(title='FeatureDive', version='0.2.0', lifespan=lifespan)
    app.state.engine = None
    app.add_middleware(RequestLimits, max_body_bytes=config['max_body_bytes'],
                       requests_per_minute=config['requests_per_minute'], max_rate_clients=config['max_rate_clients'])

    def engine(request: Request) -> InterpretabilityEngine:
        if request.app.state.engine is None:
            raise HTTPException(503, 'Models or sample catalog unavailable; check local setup')
        return request.app.state.engine

    def invoke(call, body):
        try:
            return call(body)
        except KeyError as error:
            raise HTTPException(404, 'Unknown sample ID; use GET /api/samples') from error

    @app.get('/api/health')
    def health() -> dict:
        return {'status': 'ok'}

    @app.get('/api/ready')
    def ready(service: InterpretabilityEngine = Depends(engine)) -> dict:
        return {'status': 'ready', 'models': sorted(service.models),
                'samples': len(service.samples), 'inference_device': 'cpu'}

    @app.get('/api/models')
    def models(service: InterpretabilityEngine = Depends(engine)) -> dict:
        return service.model_catalog()

    @app.get('/api/samples')
    def samples(service: InterpretabilityEngine = Depends(engine)) -> dict:
        return {'samples': service.sample_catalog(), 'role': 'illustrative_training_examples'}

    @app.post('/api/infer', response_model=ExperimentResponse)
    def infer(body: ExperimentRequest, service: InterpretabilityEngine = Depends(engine)):
        return invoke(service.infer, body)

    @app.post('/api/intervene', response_model=ExperimentResponse)
    def intervene(body: ExperimentRequest, service: InterpretabilityEngine = Depends(engine)):
        return invoke(service.infer, body)

    @app.post('/api/activations', response_model=ActivationResponse)
    def activations(body: ActivationRequest, service: InterpretabilityEngine = Depends(engine)):
        return invoke(service.activations, body)

    @app.post('/api/saliency', response_model=SaliencyResponse)
    def saliency_route(body: ExperimentRequest, service: InterpretabilityEngine = Depends(engine)):
        return invoke(service.saliency, body)

    @app.post('/api/hints', response_model=HintResponse)
    def hints(body: HintRequest, service: InterpretabilityEngine = Depends(engine)):
        return invoke(service.hints, body)

    return app


app = create_app()
