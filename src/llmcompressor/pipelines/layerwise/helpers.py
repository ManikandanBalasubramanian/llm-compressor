"""
Helpers for layerwise weight loading and offloading from safetensors files.
"""

import json
import os
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from safetensors import safe_open
from safetensors.torch import save_file
from torch.nn import Module


__all__ = [
    "build_weight_map",
    "get_subgraph_weight_names",
    "load_subgraph_weights",
    "offload_subgraph_weights",
]


def build_weight_map(model_path: str | os.PathLike) -> dict[str, str]:
    """
    Build a mapping from weight name -> safetensors file path.

    Reads the model.safetensors.index.json if present (sharded model),
    otherwise assumes a single model.safetensors file. Supports both
    local paths and Hugging Face Hub model IDs.

    :param model_path: path to the model directory or HF hub model ID
    :return: dict mapping weight name to absolute safetensors file path
    """
    model_path_str = str(model_path)
    model_path = Path(model_path_str)

    # If model_path is not a local directory, resolve from HF cache
    if not model_path.is_dir():
        from huggingface_hub import snapshot_download

        model_path = Path(snapshot_download(model_path_str))

    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        weight_map = {}
        for weight_name, shard_file in index["weight_map"].items():
            weight_map[weight_name] = str(model_path / shard_file)
        return weight_map

    # Single file model
    single_file = model_path / "model.safetensors"
    if single_file.exists():
        weight_map = {}
        with safe_open(str(single_file), framework="pt") as f:
            for key in f.keys():
                weight_map[key] = str(single_file)
        return weight_map

    raise FileNotFoundError(
        f"No safetensors files found in {model_path}. "
        "Layerwise quantization requires safetensors format."
    )


def _get_module_weight_names(
    model: Module, subgraph_modules: set[Module]
) -> dict[str, str]:
    """
    Get the full parameter names for all parameters in the given modules.

    :param model: the full model (for resolving parameter names)
    :param subgraph_modules: set of modules whose weights to load
    :return: dict mapping parameter name -> module name
    """
    # Build a mapping from module to its fully qualified name
    module_to_name = {module: name for name, module in model.named_modules()}

    param_names = {}
    for module in subgraph_modules:
        module_name = module_to_name.get(module)
        if module_name is None:
            continue
        for param_name, _ in module.named_parameters(recurse=True):
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            param_names[full_name] = module_name

    return param_names


def get_subgraph_weight_names(
    model: Module,
    weight_map: dict[str, str],
    sequential_targets: list[str],
    subgraph_index: int,
    num_subgraphs: int,
) -> list[str]:
    """
    Determine which weight names belong to a given subgraph based on the
    sequential target partitioning.

    The subgraphs are laid out as:
    - subgraph 0: head (embedding, pre-layer norms, etc.) — everything before
      the first sequential target
    - subgraph 1..N: one per sequential target (decoder layer)
    - Note: the last subgraph may include post-layer modules (final norm, lm_head)

    :param model: the model (on meta device)
    :param weight_map: mapping from weight name to safetensors file path
    :param sequential_targets: list of target module name patterns
    :param subgraph_index: which subgraph (0-indexed)
    :param num_subgraphs: total number of subgraphs
    :return: list of weight names to load for this subgraph
    """
    from compressed_tensors.utils.match import match_named_modules

    # Find actual target module names (preserving model order)
    target_modules = list(match_named_modules(model, sequential_targets))
    target_names = [name for name, _ in target_modules]

    all_weight_names = list(weight_map.keys())
    target_prefixes = [f"{name}." for name in target_names]

    if subgraph_index == 0:
        # First subgraph: non-target weights that come BEFORE the first target
        # in the model's module definition order (e.g., embed_tokens)
        module_order = {
            name: idx for idx, (name, _) in enumerate(model.named_modules())
        }
        first_target_order = min(
            module_order.get(name, float("inf")) for name in target_names
        )
        return [
            w for w in all_weight_names
            if not any(w.startswith(p) for p in target_prefixes)
            and module_order.get(w.rsplit(".", 1)[0] if "." in w else "", float("inf"))
            < first_target_order
        ]
    elif subgraph_index == num_subgraphs - 1:
        # Last subgraph: last target's weights + post-target non-target weights
        # (e.g., model.norm, lm_head that come after all decoder layers)
        target_idx = subgraph_index - 1
        result = []
        if target_idx < len(target_names):
            prefix = f"{target_names[target_idx]}."
            result = [w for w in all_weight_names if w.startswith(prefix)]

        module_order = {
            name: idx for idx, (name, _) in enumerate(model.named_modules())
        }
        last_target_order = max(
            module_order.get(name, 0) for name in target_names
        )
        for w in all_weight_names:
            if any(w.startswith(p) for p in target_prefixes):
                continue
            module_name = w.rsplit(".", 1)[0] if "." in w else ""
            if module_order.get(module_name, -1) > last_target_order:
                result.append(w)
        return result
    else:
        # Middle subgraph: just the target's weights
        target_idx = subgraph_index - 1
        if target_idx < len(target_names):
            prefix = f"{target_names[target_idx]}."
            return [w for w in all_weight_names if w.startswith(prefix)]
        return []


def load_subgraph_weights(
    model: Module,
    weight_names: list[str],
    weight_map: dict[str, str],
    device: torch.device,
) -> None:
    """
    Load weights from safetensors files for the given weight names,
    replacing meta-device parameters with real tensors on the target device.

    :param model: the full model (meta device)
    :param weight_names: list of parameter names to load
    :param weight_map: mapping from weight name to safetensors file path
    :param device: device to load weights onto
    """
    if not weight_names:
        return

    # Group parameters by safetensors file for efficient loading
    file_to_params: dict[str, list[str]] = {}
    missing_params = []
    for param_name in weight_names:
        if param_name in weight_map:
            file_path = weight_map[param_name]
            file_to_params.setdefault(file_path, []).append(param_name)
        else:
            missing_params.append(param_name)

    if missing_params:
        logger.warning(
            f"Could not find {len(missing_params)} parameters in weight map: "
            f"{missing_params[:5]}{'...' if len(missing_params) > 5 else ''}"
        )

    # Load weights from each safetensors file
    loaded_count = 0
    for file_path, params in file_to_params.items():
        with safe_open(file_path, framework="pt", device=str(device)) as f:
            for param_name in params:
                tensor = f.get_tensor(param_name)
                _set_parameter(model, param_name, tensor)
                loaded_count += 1

    logger.debug(f"Loaded {loaded_count} parameters for subgraph onto {device}")

    # Also move any quantization buffers (observers, scales, zero_points)
    # that were initialized on meta device to the target device
    _move_quantization_buffers(model, weight_names, device)


def offload_subgraph_weights(
    model: Module,
    weight_names: list[str],
    device: str = "meta",
) -> None:
    """
    Offload subgraph weights to the specified device to free GPU memory.

    :param model: the full model
    :param weight_names: list of parameter names to offload
    :param device: target device ("meta" to free all memory, "cpu" to keep on CPU)
    """
    target_device = torch.device(device)
    freed_count = 0

    for param_name in weight_names:
        parts = param_name.split(".")
        module = model
        try:
            for part in parts[:-1]:
                module = getattr(module, part)
            attr = parts[-1]
            param = getattr(module, attr, None)
        except AttributeError:
            continue

        if isinstance(param, torch.nn.Parameter) and param.device != target_device:
            if target_device.type == "meta":
                new_data = torch.empty_like(param, device="meta")
            else:
                new_data = param.data.to(target_device)

            new_param = torch.nn.Parameter(
                new_data,
                requires_grad=param.requires_grad,
            )
            module._parameters[attr] = new_param
            freed_count += 1

    if freed_count > 0:
        logger.debug(f"Offloaded {freed_count} parameters to {device}")
        if device != "cpu":
            torch.cuda.empty_cache()


def save_subgraph_weights(
    model: Module,
    subgraph_modules: set[Module],
    output_dir: str | os.PathLike,
    shard_index: int,
    weight_map_output: dict[str, str],
) -> int:
    """
    Save the current weights of subgraph modules to a safetensors shard.

    :param model: the full model
    :param subgraph_modules: set of modules whose weights to save
    :param output_dir: directory to save safetensors shards
    :param shard_index: index for naming the shard file
    :param weight_map_output: dict to update with weight_name -> shard_file mappings
    :return: total size in bytes of saved tensors
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    param_names = _get_module_weight_names(model, subgraph_modules)
    if not param_names:
        return 0

    # Collect tensors to save
    tensors = {}
    for param_name in param_names:
        parts = param_name.split(".")
        obj = model
        for part in parts:
            obj = getattr(obj, part)
        if isinstance(obj, torch.Tensor) and obj.device.type != "meta":
            tensors[param_name] = obj.contiguous().cpu()

    if not tensors:
        return 0

    shard_name = f"model-{shard_index:05d}-of-99999.safetensors"
    shard_path = output_dir / shard_name
    save_file(tensors, str(shard_path))

    total_size = sum(t.nbytes for t in tensors.values())
    for name in tensors:
        weight_map_output[name] = shard_name

    logger.info(
        f"Saved {len(tensors)} tensors ({total_size / 1e9:.2f} GB) to {shard_name}"
    )
    return total_size


def _set_parameter(model: Module, param_name: str, tensor: torch.Tensor) -> None:
    """
    Set a parameter on the model by its fully qualified name,
    replacing a meta-device parameter with a real tensor.

    :param model: the model to set the parameter on
    :param param_name: fully qualified parameter name (e.g., "model.layers.0.weight")
    :param tensor: the real tensor to set
    """
    parts = param_name.split(".")
    module = model
    for part in parts[:-1]:
        module = getattr(module, part)

    param_attr = parts[-1]
    old_param = getattr(module, param_attr, None)

    if isinstance(old_param, torch.nn.Parameter):
        new_param = torch.nn.Parameter(tensor, requires_grad=old_param.requires_grad)
        module._parameters[param_attr] = new_param
    else:
        setattr(module, param_attr, tensor)


def _move_quantization_buffers(
    model: Module, weight_names: list[str], device: torch.device
) -> None:
    """
    Move quantization-related buffers and parameters (observers, scales, zero_points)
    from meta device to the target device. These are created by QuantizationModifier
    during initialization and may be on meta device for layerwise models.

    :param model: the model
    :param weight_names: list of weight names that were just loaded (used to
        identify which modules to process)
    :param device: device to move buffers to
    """
    # Extract unique module prefixes from weight names
    module_prefixes = set()
    for name in weight_names:
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            module_prefixes.add(parts[0])

    moved_count = 0
    for prefix in module_prefixes:
        try:
            parts = prefix.split(".")
            module = model
            for part in parts:
                module = getattr(module, part)
        except AttributeError:
            continue

        # Move all meta-device parameters and buffers on this module
        for attr_name in list(module._parameters.keys()):
            param = module._parameters[attr_name]
            if param is not None and param.device.type == "meta":
                new_param = torch.nn.Parameter(
                    torch.zeros(param.shape, dtype=param.dtype, device=device),
                    requires_grad=param.requires_grad,
                )
                module._parameters[attr_name] = new_param
                moved_count += 1

        for attr_name in list(module._buffers.keys()):
            buf = module._buffers[attr_name]
            if buf is not None and buf.device.type == "meta":
                module._buffers[attr_name] = torch.zeros(
                    buf.shape, dtype=buf.dtype, device=device
                )
                moved_count += 1

        # Move observer modules if they exist
        for attr_name in dir(module):
            if attr_name.endswith("_observer"):
                observer = getattr(module, attr_name, None)
                if observer is not None and isinstance(observer, Module):
                    for pname, param in observer.named_parameters():
                        if param.device.type == "meta":
                            parts_p = pname.split(".")
                            target = observer
                            for p in parts_p[:-1]:
                                target = getattr(target, p)
                            target._parameters[parts_p[-1]] = torch.nn.Parameter(
                                torch.zeros(
                                    param.shape, dtype=param.dtype, device=device
                                ),
                                requires_grad=param.requires_grad,
                            )
                            moved_count += 1
                    for bname, buf in observer.named_buffers():
                        if buf.device.type == "meta":
                            parts_b = bname.split(".")
                            target = observer
                            for p in parts_b[:-1]:
                                target = getattr(target, p)
                            target._buffers[parts_b[-1]] = torch.zeros(
                                buf.shape, dtype=buf.dtype, device=device
                            )
                            moved_count += 1

    if moved_count > 0:
        logger.debug(
            f"Moved {moved_count} quantization buffers/params to {device}"
        )
