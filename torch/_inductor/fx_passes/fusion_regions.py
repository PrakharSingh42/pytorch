"""Predict possible fusion regions for overlap scheduling."""

from math import prod

import torch
import torch.fx as fx
from torch._inductor import config
from torch._inductor.fx_passes.control_dependencies import is_ordering_alias
from torch._inductor.fx_passes.overlap_scheduling import (
    estimate_roofline_runtime_ms,
    is_compute_node,
)
from torch._inductor.fx_utils import get_mutated_input_nodes, get_storage
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.experimental.symbolic_shapes import optimization_hint
from torch.utils._ordered_set import OrderedSet
from torch.utils._pytree import tree_flatten
from torch.utils._runtime_estimation import get_transfer_time


def is_uninitialized_allocation(n: fx.Node) -> bool:
    if n.op != "call_function" or n.target not in (
        torch.ops.aten.empty.memory_format,
        torch.ops.aten.empty_strided.default,
        torch.ops.aten.empty_like.default,
        torch.ops.aten.new_empty.default,
        torch.ops.aten.new_empty_strided.default,
    ):
        return False
    value = n.meta.get("val")
    return (
        isinstance(value, torch.Tensor)
        and type(value) in (torch.Tensor, FakeTensor)
        and value.layout == torch.strided
        and not value.is_nested
        and value.ndim > 0
        and not (
            torch.are_deterministic_algorithms_enabled()
            and torch._C._get_deterministic_fill_uninitialized_memory()
        )
    )


def is_view_node(n: fx.Node) -> bool:
    """Check if a node is a view operation (zero cost, no memory allocation)."""

    return isinstance(n.target, torch._ops.OpOverload) and (
        n.target.is_view and n.target.namespace in ("aten", "prims")
    )


def is_fusible_node(n: fx.Node) -> bool:
    """Whether a lowering can participate in a candidate fusion region.

    Storage ordering is enforced by the graph's dependencies. Backend selection
    and final fusion decisions remain with the lowered Inductor scheduler.
    """
    if n.op == "get_attr":
        return bool(n.users) and all(is_ordering_alias(user) for user in n.users)
    if n.op != "call_function":
        return False
    if is_ordering_alias(n):
        return True

    target = n.target
    if not isinstance(target, torch._ops.OpOverload):
        return False
    if torch.Tag.nondeterministic_seeded in target.tags:
        return False
    custom = n.meta.get("custom")
    if n.meta.get("should_fallback") or (
        isinstance(custom, dict) and custom.get("fallback_to_eager")
    ):
        return False

    # Exclude collectives and waits (they have their own scheduling)
    if target.namespace == "_c10d_functional":
        return False

    from torch._inductor.lowering import fallbacks, lowerings

    # Must have a lowering
    if target not in lowerings:
        return False

    # Exclude fallbacks (ops that fall back to eager execution)
    if target in fallbacks:
        return False

    # GEMM templates can absorb producers and epilogues while retaining compute cost.
    overload_packet = target.overloadpacket
    if is_compute_node(n):
        return overload_packet in (
            torch.ops.aten.mm,
            torch.ops.aten.bmm,
            torch.ops.aten.addmm,
            torch.ops.aten.baddbmm,
            torch.ops.aten._scaled_mm,
        ) and (
            config.prologue_fusion
            or config.epilogue_fusion
            or config.triton.native_matmul
        )

    # Special case: cat is only fusible if it has few enough inputs
    if target == torch.ops.aten.cat.default:
        inputs = n.args[0] if n.args else []
        if isinstance(inputs, (list, tuple)):
            if len(inputs) > config.max_pointwise_cat_inputs:
                return False

    return True


def _fusion_inputs(n: fx.Node) -> list[fx.Node]:
    if is_ordering_alias(n):
        return [arg for arg in n.args[1:] if isinstance(arg, fx.Node)]
    return n.all_input_nodes


def _has_fusion_cost(n: fx.Node) -> bool:
    return (
        n.op == "call_function"
        and not is_view_node(n)
        and not is_ordering_alias(n)
        and not is_uninitialized_allocation(n)
    )


def _horizontal_domain(n: fx.Node) -> tuple[object, ...] | None:
    if not _has_fusion_cost(n) or not isinstance(n.target, torch._ops.OpOverload):
        return None
    val = n.meta.get("val")
    if not isinstance(val, torch.Tensor):
        return None
    shape = tuple(optimization_hint(size, fallback=-1) for size in val.shape)
    if -1 in shape:
        return None
    if is_compute_node(n):
        if not config.triton.native_matmul:
            return None
        tensors = [inp.meta.get("val") for inp in n.all_input_nodes]
        # addmm/baddbmm include a bias input before the matrix operands.
        offset = int(
            n.target.overloadpacket in (torch.ops.aten.addmm, torch.ops.aten.baddbmm)
        )
        matrix = tensors[offset] if len(tensors) > offset else None
        if not isinstance(matrix, torch.Tensor) or matrix.ndim < 2:
            return None
        k = optimization_hint(matrix.shape[-1], fallback=-1)
        return (val.device, shape, "matmul", k) if k >= 0 else None
    if torch.Tag.reduction in n.target.tags:
        tensors = [inp.meta.get("val") for inp in n.all_input_nodes]
        value = tensors[0] if tensors else None
        if not isinstance(value, torch.Tensor):
            return None
        input_shape = tuple(
            optimization_hint(size, fallback=-1) for size in value.shape
        )
        if -1 in input_shape:
            return None
        return val.device, prod(shape), "reduction", prod(input_shape)
    return val.device, prod(shape), "pointwise"


def _can_attach_matmul(producer: fx.Node, consumer: fx.Node) -> bool:
    a, b = producer.meta.get("val"), consumer.meta.get("val")
    if (
        isinstance(a, torch.Tensor)
        and isinstance(b, torch.Tensor)
        and a.device != b.device
    ):
        return False
    if is_compute_node(producer) and is_compute_node(consumer):
        return False
    if is_compute_node(consumer):
        if not isinstance(producer.target, torch._ops.OpOverload):
            return True
        return (
            (config.triton.native_matmul or config.prologue_fusion)
            and not producer.target._schema.is_mutable
            and torch.Tag.reduction not in producer.target.tags
        )
    if is_compute_node(producer):
        if is_view_node(consumer) or is_ordering_alias(consumer):
            return True
        if not isinstance(consumer.target, torch._ops.OpOverload):
            return False
        return (config.triton.native_matmul or config.epilogue_fusion) and (
            torch.Tag.reduction not in consumer.target.tags
            or config.triton.native_matmul
        )
    return True


def _get_contiguous_fusible_spans(gm: fx.GraphModule) -> list[list[fx.Node]]:
    """Get contiguous spans of fusible nodes from the graph.

    Walks the graph in topological order and groups consecutive fusible
    nodes into spans. Non-fusible nodes act as span boundaries.
    """
    spans: list[list[fx.Node]] = []
    current_span: list[fx.Node] = []

    for node in gm.graph.nodes:
        if is_fusible_node(node):
            current_span.append(node)
        else:
            # Non-fusible node ends the current span
            if current_span:
                spans.append(current_span)
                current_span = []

    if current_span:
        spans.append(current_span)

    return spans


def _find_connected_components(span: list[fx.Node]) -> list[list[fx.Node]]:
    """Join data neighbors and compatible horizontal candidates within a span."""
    if not span:
        return []

    from torch.fx.experimental.optimization import UnionFind

    span_set = OrderedSet(span)
    node_to_idx = {n: i for i, n in enumerate(span)}

    uf = UnionFind(len(span))
    for i in range(len(span)):
        uf.make_set(i)

    sizes = [int(_has_fusion_cost(n)) for n in span]
    matmuls = [int(is_compute_node(n)) for n in span]
    matmul_domains = [
        _horizontal_domain(n) if is_compute_node(n) else None for n in span
    ]
    domains: dict[tuple[object, ...], int] = {}

    def join(
        left: int, right: int, *, producer_edge: bool = False, horizontal: bool = False
    ) -> None:
        a, b = uf.find(left), uf.find(right)
        if a == b or sizes[a] + sizes[b] > config.max_fusion_size:
            return
        if (
            horizontal
            and (matmuls[a] or matmuls[b])
            and not config.triton.native_matmul
        ):
            return
        if matmuls[a] and matmuls[b]:
            if (
                producer_edge
                or not config.triton.native_matmul
                or matmul_domains[a] is None
                or matmul_domains[a] != matmul_domains[b]
            ):
                return
        count, compute = sizes[a] + sizes[b], matmuls[a] + matmuls[b]
        domain = matmul_domains[a] if matmuls[a] else matmul_domains[b]
        uf.join(a, b)
        root = uf.find(a)
        sizes[root], matmuls[root], matmul_domains[root] = count, compute, domain

    for node in span:
        node_idx = node_to_idx[node]
        for inp in _fusion_inputs(node):
            if inp in span_set and _can_attach_matmul(inp, node):
                join(node_to_idx[inp], node_idx, producer_edge=is_compute_node(node))
        domain = _horizontal_domain(node)
        if domain is not None:
            if domain in domains:
                join(domains[domain], node_idx, horizontal=True)
            domains[domain] = node_idx

    # Group by root
    root_to_nodes: dict[int, list[fx.Node]] = {}
    for node in span:
        root = uf.find(node_to_idx[node])
        if root not in root_to_nodes:
            root_to_nodes[root] = []
        root_to_nodes[root].append(node)

    return list(root_to_nodes.values())


def build_fusion_regions(
    gm: fx.GraphModule,
) -> dict[fx.Node, OrderedSet[fx.Node]]:
    """Build candidate regions without predicting the selected kernel implementation.

    1. Identify contiguous spans of fusible nodes (separated by non-fusible nodes)
    2. Connect data neighbors and compatible horizontal operations within each span
    3. Return regions that have 2+ operations with compute or I/O cost

    This ensures fusion regions are strictly local - no reordering across
    non-fusible node boundaries.

    Returns a dict mapping each node to its fusion group (OrderedSet of nodes).
    """
    # Build node -> topo index map for sorting
    node_to_idx: dict[fx.Node, int] = {n: i for i, n in enumerate(gm.graph.nodes)}

    # Step 1: Get contiguous spans of fusible nodes
    spans = _get_contiguous_fusible_spans(gm)

    # Step 2: Find connected components within each span
    region_of: dict[fx.Node, OrderedSet[fx.Node]] = {}

    for span in spans:
        if len(span) < 2:
            continue

        components = _find_connected_components(span)

        for component in components:
            if sum(_has_fusion_cost(n) for n in component) < 2:
                continue

            # Sort nodes in topological order to preserve original ordering
            sorted_component = sorted(component, key=lambda n: node_to_idx[n])
            node_set = OrderedSet(sorted_component)

            for node in sorted_component:
                region_of[node] = node_set

    return region_of


def contiguous_fusion_regions(gm: fx.GraphModule) -> list[OrderedSet[fx.Node]]:
    region_of = build_fusion_regions(gm)
    regions = []
    current: OrderedSet[fx.Node] = OrderedSet()
    previous = None
    for node in gm.graph.nodes:
        region = region_of.get(node)
        if region is not previous:
            if len(current) > 1:
                regions.append(current)
            current = OrderedSet()
        if region is not None:
            current.add(node)
        previous = region
    if len(current) > 1:
        regions.append(current)
    return [
        region for region in regions if sum(_has_fusion_cost(n) for n in region) > 1
    ]


def estimate_fused_node_costs(
    region_of: dict[fx.Node, OrderedSet[fx.Node]],
) -> dict[fx.Node, float]:
    """Estimate per-node costs accounting for fusion (no graph collapse).

    For each node in a fusion region, only count I/O that crosses the
    region boundary (external reads/writes), assuming candidate intermediates
    are eliminated. GEMM runtime and destination writes remain represented.

    Nodes with only internal I/O get cost 0. Nodes with external I/O
    get the bandwidth cost of just those external tensors.
    """
    costs: dict[fx.Node, float] = {}
    seen: OrderedSet[int] = OrderedSet()

    for node_set in region_of.values():
        rid = id(node_set)
        if rid in seen:
            continue
        seen.add(rid)

        for n in node_set:
            if is_compute_node(n):
                costs[n] = estimate_roofline_runtime_ms(n)
                continue
            if n.op == "get_attr" or is_uninitialized_allocation(n):
                costs[n] = 0.0
                continue
            # External inputs: values from nodes outside this region
            ext_inputs = []
            for inp in _fusion_inputs(n):
                if inp not in node_set:
                    val = inp.meta.get("val")
                    if val is not None:
                        ext_inputs.append(val)

            # External outputs: this node's value is used outside the region
            written = OrderedSet(get_mutated_input_nodes(n))
            ext_outputs = [inp.meta["val"] for inp in written if "val" in inp.meta]
            flat_written, _ = tree_flatten(ext_outputs)
            written_storages = OrderedSet(
                get_storage(t)
                for t in flat_written
                if isinstance(t, torch.Tensor) and torch._C._has_storage(t)
            )
            has_external_user = any(u not in node_set for u in n.users)
            if has_external_user:
                val = n.meta.get("val")
                if val is not None:
                    outputs, _ = tree_flatten(val)
                    ext_outputs.extend(
                        t
                        for t in outputs
                        if not isinstance(t, torch.Tensor)
                        or not torch._C._has_storage(t)
                        or get_storage(t) not in written_storages
                    )

            if not ext_inputs and not ext_outputs:
                costs[n] = 0.0
            else:
                flat_in, _ = tree_flatten(ext_inputs)
                flat_out, _ = tree_flatten(ext_outputs)
                costs[n] = get_transfer_time(flat_in, flat_out) / 1e6

    return costs
