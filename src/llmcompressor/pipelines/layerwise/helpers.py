"""
Helpers for layerwise weight loading and offloading from safetensors files.

Optimizations:
- On-demand shard downloads: only fetches the safetensors shards needed for
  the current subgraph instead of downloading the full model upfront.
- Compress-as-you-go: compresses and saves each subgraph immediately after
  calibration, keeping only 1 subgraph of base weights in memory at a time.
- Background prefetch: downloads the next subgraph's shards while the current
  subgraph is being calibrated on GPU.
"""

import json
import os
import threading
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
    "compress_and_save_subgraph",
    "write_safetensors_index",
    "ShardPrefetcher",
]


def build_weight_map(model_path: str | os.PathLike) -> dict[str, str]:
    """
    Build a mapping from weight name -> safetensors file path.

    For HF Hub models, downloads only the index file (not full model weights).
    Shard files are downloaded on-demand during load_subgraph_weights().

    :param model_path: path to the model directory or HF hub model ID
    :return: dict mapping weight name to absolute safetensors file path
    """
    model_path_str = str(model_path)
    model_path = Path(model_path_str)

    # If model_path is a local directory, use it directly
    if model_path.is_dir():
        return _build_weight_map_from_dir(model_path)

    # For HF Hub model IDs, download only the index/config files first
    from huggingface_hub import hf_hub_download

    # Try to download just the index file (sharded model)
    try:
        index_path = hf_hub_download(
            model_path_str, "model.safetensors.index.json"
        )
        with open(index_path) as f:
            index = json.load(f)

        # The index file is in the snapshot dir — derive the model dir from it
        snapshot_dir = Path(index_path).parent
        weight_map = {}
        for weight_name, shard_file in index["weight_map"].items():
            weight_map[weight_name] = str(snapshot_dir / shard_file)
        return weight_map
    except Exception:
        pass

    # Try single-file model
    try:
        single_path = hf_hub_download(model_path_str, "model.safetensors")
        weight_map = {}
        with safe_open(single_path, framework="pt") as f:
            for key in f.keys():
                weight_map[key] = single_path
        return weight_map
    except Exception:
        pass

    # Fall back to full snapshot_download
    from huggingface_hub import snapshot_download

    model_path = Path(snapshot_download(model_path_str))
    return _build_weight_map_from_dir(model_path)


def _build_weight_map_from_dir(model_path: Path) -> dict[str, str]:
    """Build weight map from a local directory."""
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        weight_map = {}
        for weight_name, shard_file in index["weight_map"].items():
            weight_map[weight_name] = str(model_path / shard_file)
        return weight_map

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
    model_path: str | None = None,
) -> None:
    """
    Load weights from safetensors files for the given weight names,
    replacing meta-device parameters with real tensors on the target device.

    If a shard file does not exist locally, it will be downloaded on-demand
    from the HF Hub using model_path as the repo ID.

    :param model: the full model (meta device)
    :param weight_names: list of parameter names to load
    :param weight_map: mapping from weight name to safetensors file path
    :param device: device to load weights onto
    :param model_path: HF Hub model ID for on-demand shard downloads
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

    # Load weights from each safetensors file, downloading on demand if needed
    loaded_count = 0
    for file_path, params in file_to_params.items():
        resolved_path = _ensure_shard_available(file_path, model_path)
        with safe_open(resolved_path, framework="pt", device=str(device)) as f:
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


def _ensure_shard_available(
    file_path: str, model_path: str | None = None
) -> str:
    """
    Ensure a safetensors shard file exists locally. If the file is missing
    (because we only downloaded the index, not all shards), download it
    on-demand from the HF Hub.

    :param file_path: expected local path to the shard file
    :param model_path: HF Hub model ID for downloading
    :return: resolved local path to the shard file
    """
    if os.path.isfile(file_path):
        return file_path

    if model_path is None:
        raise FileNotFoundError(
            f"Shard file not found: {file_path}. "
            "Provide model_path for on-demand downloading."
        )

    # Extract shard filename from the path
    shard_filename = os.path.basename(file_path)
    logger.info(f"Downloading shard on-demand: {shard_filename}")

    from huggingface_hub import hf_hub_download

    downloaded_path = hf_hub_download(model_path, shard_filename)
    return downloaded_path


def compress_and_save_subgraph(
    model: Module,
    weight_names: list[str],
    output_dir: str | os.PathLike,
    shard_index: int,
    shard_weight_map: dict[str, str],
) -> int:
    """
    Compress quantized modules for the given weight names in-place, then save
    all parameters (compressed weights + quantization params) to a safetensors
    shard. After saving, offloads weights to meta device to free memory.

    This enables "compress-as-you-go": each subgraph is compressed and saved
    immediately after calibration, so only 1 subgraph's base weights are ever
    in CPU/GPU memory at a time.

    :param model: the full model
    :param weight_names: list of parameter names belonging to this subgraph
    :param output_dir: directory to save safetensors shards
    :param shard_index: index for naming the shard file
    :param shard_weight_map: dict to update with weight_name -> shard_file mappings
    :return: total size in bytes of saved tensors
    """
    from compressed_tensors.compressors.base import compress_module
    from compressed_tensors.quantization.utils.helpers import is_module_quantized

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Identify modules that contain the subgraph's weights
    module_prefixes = set()
    for name in weight_names:
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            module_prefixes.add(parts[0])

    # Compress quantized modules in this subgraph
    compressed_count = 0
    for prefix in module_prefixes:
        try:
            parts = prefix.split(".")
            module = model
            for part in parts:
                module = getattr(module, part)
        except AttributeError:
            continue

        if is_module_quantized(module):
            compress_module(module)
            compressed_count += 1

    if compressed_count > 0:
        logger.debug(f"Compressed {compressed_count} quantized modules in subgraph")

    # Collect all non-meta tensors (parameters + buffers) from these modules
    tensors = {}
    for prefix in module_prefixes:
        try:
            parts = prefix.split(".")
            module = model
            for part in parts:
                module = getattr(module, part)
        except AttributeError:
            continue

        for pname, param in module.named_parameters(recurse=False):
            full_name = f"{prefix}.{pname}"
            if param.device.type != "meta":
                tensors[full_name] = param.data.contiguous().cpu()

        for bname, buf in module.named_buffers(recurse=False):
            full_name = f"{prefix}.{bname}"
            if buf.device.type != "meta":
                tensors[full_name] = buf.contiguous().cpu()

    if not tensors:
        return 0

    # Save to shard file
    shard_name = f"model-{shard_index + 1:05d}-of-99999.safetensors"
    shard_path = output_dir / shard_name
    save_file(tensors, str(shard_path))

    total_size = sum(t.nbytes for t in tensors.values())
    for name in tensors:
        shard_weight_map[name] = shard_name

    logger.info(
        f"Saved subgraph {shard_index + 1}: "
        f"{len(tensors)} tensors ({total_size / 1e9:.2f} GB) -> {shard_name}"
    )

    # Offload to meta to free memory
    offload_subgraph_weights(model, weight_names, device="meta")

    return total_size


def write_safetensors_index(
    output_dir: str | os.PathLike,
    shard_weight_map: dict[str, str],
    total_size: int,
) -> None:
    """
    Write the model.safetensors.index.json file and rename shard files to
    reflect the actual total number of shards.

    :param output_dir: directory containing the shard files
    :param shard_weight_map: mapping of weight name -> shard filename
    :param total_size: total size in bytes of all saved tensors
    """
    output_dir = Path(output_dir)

    # Determine actual shard files used
    shard_files = sorted(set(shard_weight_map.values()))
    num_shards = len(shard_files)

    # Rename shards to have correct total count
    rename_map = {}
    for i, old_name in enumerate(shard_files):
        new_name = f"model-{i + 1:05d}-of-{num_shards:05d}.safetensors"
        if old_name != new_name:
            old_path = output_dir / old_name
            new_path = output_dir / new_name
            if old_path.exists():
                old_path.rename(new_path)
            rename_map[old_name] = new_name

    # Update weight map with renamed shard files
    final_weight_map = {}
    for weight_name, shard_file in shard_weight_map.items():
        final_weight_map[weight_name] = rename_map.get(shard_file, shard_file)

    # Write index file
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": final_weight_map,
    }
    index_path = output_dir / "model.safetensors.index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    logger.info(
        f"Written safetensors index: {num_shards} shards, "
        f"{total_size / 1e9:.2f} GB total"
    )


class ShardPrefetcher:
    """
    Background prefetcher that downloads the next subgraph's shard files
    while the current subgraph is being calibrated on GPU.

    Usage:
        prefetcher = ShardPrefetcher(model_path)
        prefetcher.prefetch(next_weight_names, weight_map)
        # ... calibrate current subgraph ...
        prefetcher.wait()  # ensure next shards are ready
    """

    def __init__(self, model_path: str | None = None):
        self._model_path = model_path
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None

    def prefetch(
        self,
        weight_names: list[str],
        weight_map: dict[str, str],
    ) -> None:
        """Start background download of shard files for the given weight names."""
        self.wait()  # ensure previous prefetch is complete

        # Determine unique shard files needed
        shard_files = set()
        for name in weight_names:
            if name in weight_map:
                shard_files.add(weight_map[name])

        # Filter to only files that need downloading
        files_to_download = [f for f in shard_files if not os.path.isfile(f)]
        if not files_to_download:
            return

        self._error = None
        self._thread = threading.Thread(
            target=self._download_shards,
            args=(files_to_download,),
            daemon=True,
        )
        self._thread.start()

    def _download_shards(self, file_paths: list[str]) -> None:
        """Download shard files in background thread."""
        try:
            for file_path in file_paths:
                _ensure_shard_available(file_path, self._model_path)
        except Exception as e:
            self._error = e

    def wait(self) -> None:
        """Wait for background prefetch to complete. Raises if download failed."""
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._error is not None:
            error = self._error
            self._error = None
            raise error
