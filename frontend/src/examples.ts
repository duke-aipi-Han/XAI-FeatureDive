import type { ExperimentRequest } from './types';
export const FLIP_EXAMPLES: { name: string; layer: 'conv1'|'conv2'; channel: number; request: ExperimentRequest }[] = [
  { name:'cat',layer:'conv1',channel:3,request:{model_id:'balanced',sample_id:'cat:89133',background:'green',target_class:'cat',interventions:[]} },
  { name:'fish',layer:'conv1',channel:7,request:{model_id:'balanced',sample_id:'fish:20708',background:'blue',target_class:'fish',interventions:[]} },
  { name:'airplane',layer:'conv2',channel:3,request:{model_id:'balanced',sample_id:'airplane:4513',background:'yellow',target_class:'airplane',interventions:[]} },
];

export const HINT_EXAMPLES = {
  feature: {model_id:'balanced',sample_id:'fish:20708',background:'blue',target_class:'fish',interventions:[]} as ExperimentRequest,
  erase: {model_id:'balanced',sample_id:'airplane:72939',background:'yellow',target_class:'airplane',interventions:[]} as ExperimentRequest,
};
