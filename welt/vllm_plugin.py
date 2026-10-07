"""
vLLM plugin (entry point `vllm.general_plugins`): bidirectional attention for the latent transformer's shift blocks.

vLLM supports bidirectional attention ranges for "prefix-LM" models (`is_mm_prefix_lm` in the model config),
taking the ranges from each request's multimodal placeholders. WeLT passes word embeddings as `prompt_embeds`
instead, so the shift block ranges are sent in `PoolingParams.extra_kwargs["bidirectional_ranges"]` and
exposed to vLLM's model runner as placeholders here.
"""
from types import SimpleNamespace

RANGES_KEY = "bidirectional_ranges"


def register():
    from vllm.multimodal.inputs import PlaceholderRange
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_welt_patched", False):
        return
    update_states = GPUModelRunner._update_states

    def _update_states(self, scheduler_output):
        result = update_states(self, scheduler_output)
        for new_request in scheduler_output.scheduled_new_reqs:
            state = self.requests.get(new_request.req_id)
            params = state.pooling_params if state is not None else None
            ranges = (params.extra_kwargs or {}).get(RANGES_KEY) if params is not None else None
            if ranges:
                # ponytail: relies on vLLM's runner reading `mm_features[i].mm_position` for prefix-LM ranges
                state.mm_features = [
                    SimpleNamespace(modality="welt_shift_block",
                                    mm_position=PlaceholderRange(offset=start, length=end - start + 1))
                    for start, end in ranges]
        return result

    GPUModelRunner._update_states = _update_states
    GPUModelRunner._welt_patched = True
