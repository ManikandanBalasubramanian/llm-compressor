"""
Layerwise calibration pipeline for memory-efficient quantization.

This pipeline enables quantization of models that exceed available memory by
loading weights from safetensors files per-subgraph during calibration, rather
than loading the entire model at once.

The pipeline follows the same structure as SequentialPipeline:
1. Model is traced into subgraphs (using meta-device model structure)
2. For each subgraph:
   a. Weights are loaded from safetensors onto GPU
   b. Calibration pass triggers modifier hooks (AWQ, GPTQ, etc.)
   c. Quantization/smoothing is applied
   d. Propagation pass captures compressed outputs
   e. Weights are offloaded back to meta device
"""

import contextlib
from typing import TYPE_CHECKING, Iterator

import torch
from compressed_tensors.offload import disable_offloading
from loguru import logger
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from llmcompressor.core import LifecycleCallbacks, active_session
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.pipelines.layerwise.helpers import (
    build_weight_map,
    get_subgraph_weight_names,
    load_subgraph_weights,
    offload_subgraph_weights,
)
from llmcompressor.pipelines.registry import CalibrationPipeline
from llmcompressor.pipelines.sequential.helpers import (
    handle_sequential_oom,
    trace_subgraphs,
)
from llmcompressor.utils.dev import get_main_device
from llmcompressor.utils.helpers import DisableQuantization, calibration_forward_context
from llmcompressor.utils.pytorch.module import infer_sequential_targets

if TYPE_CHECKING:
    from llmcompressor.args.dataset_arguments import DatasetArguments

__all__ = ["LayerwisePipeline"]


def _get_batches(
    activations: IntermediatesCache,
    num_batches: int,
    input_names: list[str],
    desc: str,
    sequential_prefetch: bool = False,
) -> Iterator[tuple[int, dict]]:
    """
    Yield (batch_idx, inputs) with optional prefetching.
    """
    batch_source = (
        activations.iter_prefetch(input_names)
        if sequential_prefetch
        else activations.iter(input_names)
    )
    for batch_idx, inputs in tqdm(
        enumerate(batch_source), total=num_batches, desc=desc
    ):
        yield batch_idx, inputs


@CalibrationPipeline.register("layerwise")
class LayerwisePipeline(CalibrationPipeline):
    @staticmethod
    @handle_sequential_oom
    def __call__(
        model: torch.nn.Module,
        dataloader: DataLoader,
        dataset_args: "DatasetArguments",
    ):
        """
        Run a layerwise calibration pipeline for memory-efficient quantization.

        This pipeline is identical to SequentialPipeline except that model weights
        are loaded from safetensors files per-subgraph rather than being resident
        in memory. This enables quantization of models that are too large to fit
        in GPU or CPU memory.

        Requirements:
        - The model must be on meta device (loaded with layerwise=True)
        - The original model weights must be in safetensors format

        :param model: model on meta device being calibrated
        :param dataloader: loads data for calibration
        :param dataset_args: dataset arguments relevant to pipelines
        """
        session = active_session()

        # Determine model source path for weight loading
        model_path = getattr(model.config, "_name_or_path", None) or getattr(
            model, "name_or_path", None
        )
        if model_path is None:
            raise ValueError(
                "Cannot determine model source path for layerwise weight loading. "
                "Ensure the model was loaded with layerwise=True."
            )

        # Build weight map from safetensors index
        weight_map = build_weight_map(model_path)
        logger.info(
            f"Built weight map with {len(weight_map)} parameters "
            f"from {model_path}"
        )

        # prepare model for sequential onloading
        onload_device = get_main_device()
        offload_device = torch.device(dataset_args.sequential_offload_device)

        # prepare to trace subgraphs
        sequential_targets = infer_sequential_targets(
            model, dataset_args.sequential_targets
        )
        ignore = dataset_args.tracing_ignore

        # For meta-device models, we need a sample input on meta device for tracing
        # torch.fx tracing is symbolic and doesn't need real values
        sample_input = next(iter(dataloader))

        # trace subgraphs (symbolic, works on meta device)
        subgraphs = trace_subgraphs(
            model,
            sample_input,
            sequential_targets,
            ignore,
            dataset_args.sequential_targets_per_subgraph,
        )
        num_subgraphs = len(subgraphs)
        logger.info(f"Traced {num_subgraphs} subgraphs for layerwise calibration")

        LifecycleCallbacks.calibration_epoch_start()

        with contextlib.ExitStack() as stack:
            stack.enter_context(calibration_forward_context(model))
            stack.enter_context(DisableQuantization(model))

            # prepare intermediates cache
            activations = IntermediatesCache.from_dataloader(
                dataloader, onload_device, offload_device
            )

            # Populate loss_masks for AWQ masking support
            use_loss_mask = getattr(dataset_args, "use_loss_mask", False)
            if use_loss_mask:
                session.state.loss_masks = [
                    activations.fetch(batch_idx, ["loss_mask"]).get("loss_mask")
                    for batch_idx in range(len(dataloader))
                ]
            else:
                session.state.loss_masks = None

            sequential_prefetch = getattr(dataset_args, "sequential_prefetch", False)
            session.state.sequential_prefetch = sequential_prefetch

            for subgraph_index, subgraph in enumerate(subgraphs):
                # Determine which weights this subgraph needs
                weight_names = get_subgraph_weight_names(
                    model, weight_map, sequential_targets,
                    subgraph_index, num_subgraphs,
                )

                # Load weights for this subgraph from safetensors
                logger.info(
                    f"Loading weights for subgraph "
                    f"{subgraph_index + 1}/{num_subgraphs} "
                    f"({len(weight_names)} parameters)"
                )
                load_subgraph_weights(
                    model, weight_names, weight_map, onload_device
                )

                # prepare tqdm description texts
                calib_desc = f"({subgraph_index + 1}/{num_subgraphs}): Calibrating"
                prop_desc = f"({subgraph_index + 1}/{num_subgraphs}): Propagating"

                num_batches = len(dataloader)
                with disable_offloading():
                    # Calibration pass: triggers modifier hooks
                    for batch_idx, inputs in _get_batches(
                        activations,
                        num_batches,
                        subgraph.input_names,
                        calib_desc,
                        sequential_prefetch,
                    ):
                        session.state.current_batch_idx = batch_idx
                        outputs = subgraph.forward(model, **inputs)

                        if not dataset_args.propagate_error:
                            if subgraph_index < num_subgraphs - 1:
                                activations.update(batch_idx, outputs)
                                activations.delete(batch_idx, subgraph.consumed_names)

                    modules = list(subgraph.submodules(model))
                    LifecycleCallbacks.sequential_epoch_end(modules)

                    if dataset_args.propagate_error:
                        # Propagation pass: capture outputs of compressed modules
                        with HooksMixin.disable_hooks():
                            for batch_idx, inputs in _get_batches(
                                activations,
                                num_batches,
                                subgraph.input_names,
                                prop_desc,
                                sequential_prefetch,
                            ):
                                output = subgraph.forward(model, **inputs)
                                if subgraph_index < num_subgraphs - 1:
                                    activations.update(batch_idx, output)
                                    activations.delete(
                                        batch_idx, subgraph.consumed_names
                                    )

                # Offload processed weights to CPU to free GPU memory.
                # Weights remain on CPU so save_pretrained can compress them later.
                offload_subgraph_weights(model, weight_names, device="cpu")
                logger.info(
                    f"Completed subgraph {subgraph_index + 1}/{num_subgraphs}"
                )

            # Move any remaining GPU tensors (quantization scales, zero_points,
            # observer buffers) to CPU for the compression/save phase
            for name, param in model.named_parameters():
                if param.device.type == "cuda":
                    parts = name.rsplit(".", 1)
                    parent = model
                    for p in parts[0].split("."):
                        parent = getattr(parent, p)
                    parent._parameters[parts[-1]] = torch.nn.Parameter(
                        param.data.cpu(), requires_grad=param.requires_grad
                    )
            for name, buf in model.named_buffers():
                if buf.device.type == "cuda":
                    parts = name.rsplit(".", 1)
                    parent = model
                    for p in parts[0].split("."):
                        parent = getattr(parent, p)
                    parent._buffers[parts[-1]] = buf.cpu()

            # Finish any remaining compression
            LifecycleCallbacks.calibration_epoch_end()
