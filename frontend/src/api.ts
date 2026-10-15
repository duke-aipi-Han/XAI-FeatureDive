import type { ActivationResult, ExperimentRequest, ExperimentResult, Layer, ModelCatalog, SaliencyResult, Sample } from './types';

async function json<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`/api/${path}`, {
    method: body === undefined ? 'GET' : 'POST',
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body), signal,
  });
  if (!response.ok) {
    let message = `Request failed (${response.status}).`;
    try {
      const problem = await response.json();
      if (typeof problem.detail === 'string') message = problem.detail;
    } catch { /* A proxy failure may not have a JSON body. */ }
    if (response.status === 429) message = `Too many requests. Try again in ${response.headers.get('Retry-After') ?? '60'} seconds.`;
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}
export const api = {
  samples: (signal?: AbortSignal) => json<{ samples: Sample[] }>('samples', undefined, signal),
  models: (signal?: AbortSignal) => json<ModelCatalog>('models', undefined, signal),
  ready: (signal?: AbortSignal) => json<{ status: string }>('ready', undefined, signal),
  infer: (request: ExperimentRequest) => json<ExperimentResult>(request.interventions.length ? 'intervene' : 'infer', request),
  activations: (request: ExperimentRequest, layer: Layer, signal?: AbortSignal) => json<ActivationResult>('activations', { ...request, layer }, signal),
  saliency: (request: ExperimentRequest, signal?: AbortSignal) => json<SaliencyResult>('saliency', request, signal),
};
