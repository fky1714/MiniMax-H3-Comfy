"""AttentionBackendOverride: pick ComfyUI's attention kernel per model patch.

ComfyUI's global switches (--use-sage-attention, --use-ck-attention) are
process-wide, so flipping them means a redeploy. Since 0.3x, every
@wrap_attn attention function (comfy/ldm/modules/attention.py) first checks
transformer_options["optimized_attention_override"]; setting it on a model
clone swaps the kernel for that one sampling run only. That is what this
node does, so one deployment can A/B backends per request.

Backends are looked up in ComfyUI's own registry (register_attention_function):
"sage" exists only when the `sageattention` package imports, and
"comfy_kitchen_int8" only when comfy_kitchen reports its INT8 attention
kernel as available on this GPU. A missing backend fails the job loudly
rather than silently running the default kernel.

Checked against ComfyUI 0.36.0, where MiniMax-H3's DiT attention
(comfy/ldm/minimax/model.py Attention.forward) goes through
optimized_attention with transformer_options, so the override reaches it.
"""

import comfy.ldm.modules.attention as attention
from comfy_api.latest import ComfyExtension, io

BACKENDS = ["sage", "comfy_kitchen_int8"]


class AttentionBackendOverride(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AttentionBackendOverride",
            display_name="Attention Backend Override",
            description="Use a specific registered attention kernel for this model.",
            category="model/patch",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("backend", options=BACKENDS),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, backend) -> io.NodeOutput:
        func = attention.get_attention_function(backend, default=None)
        if func is None:
            raise ValueError(
                f"Attention backend {backend!r} is not available in this ComfyUI "
                f"(registered: {sorted(attention.REGISTERED_ATTENTION_FUNCTIONS)})"
            )

        def override(_default_func, *args, **kwargs):
            # kwargs carries _inside_attn_wrapper=True, so `func`'s own
            # @wrap_attn passes straight through instead of recursing here.
            return func(*args, **kwargs)

        m = model.clone()
        to = m.model_options["transformer_options"] = m.model_options.get("transformer_options", {}).copy()
        to["optimized_attention_override"] = override
        return io.NodeOutput(m)


class _Extension(ComfyExtension):
    async def get_node_list(self):
        return [AttentionBackendOverride]


async def comfy_entrypoint() -> _Extension:
    return _Extension()
