"""Validation, planning, and deterministic local execution for Workflows."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any


class WorkflowError(ValueError):
    """A user-correctable Workflow definition error."""


@dataclass(frozen=True)
class Plan:
    order: list[str]
    levels: list[list[str]]


SUPPORTED_TYPES = {"source", "map", "filter", "aggregate", "fail"}


def validate(workflow: dict[str, Any]) -> Plan:
    """Validate a Workflow and return its stable topological Plan."""
    nodes = workflow.get("nodes")
    edges = workflow.get("edges")
    if not isinstance(nodes, list) or not nodes:
        raise WorkflowError("A workflow needs at least one node.")
    if not isinstance(edges, list):
        raise WorkflowError("Edges must be a list.")

    ids = [node.get("id") for node in nodes if isinstance(node, dict)]
    if len(ids) != len(nodes) or any(not isinstance(node_id, str) or not node_id.strip() for node_id in ids):
        raise WorkflowError("Every node needs a non-empty string id.")
    if len(set(ids)) != len(ids):
        raise WorkflowError("Node ids must be unique.")
    for node in nodes:
        if node.get("type") not in SUPPORTED_TYPES:
            raise WorkflowError(f"Node '{node['id']}' has an unsupported type.")
        if not isinstance(node.get("config", {}), dict):
            raise WorkflowError(f"Node '{node['id']}' config must be an object.")

    positions = {node_id: index for index, node_id in enumerate(ids)}
    outgoing = {node_id: [] for node_id in ids}
    indegree = {node_id: 0 for node_id in ids}
    seen_edges: set[tuple[str, str]] = set()
    for edge in edges:
        if not isinstance(edge, dict) or edge.get("source") not in positions or edge.get("target") not in positions:
            raise WorkflowError("Every edge must connect two existing nodes.")
        pair = (edge["source"], edge["target"])
        if pair[0] == pair[1]:
            raise WorkflowError("A node cannot depend on itself.")
        if pair in seen_edges:
            raise WorkflowError("Duplicate edges are not allowed.")
        seen_edges.add(pair)
        outgoing[pair[0]].append(pair[1])
        indegree[pair[1]] += 1

    frontier = sorted((node_id for node_id in ids if indegree[node_id] == 0), key=positions.get)
    order: list[str] = []
    levels: list[list[str]] = []
    while frontier:
        level = frontier
        levels.append(level)
        order.extend(level)
        next_frontier: list[str] = []
        for source in level:
            for target in outgoing[source]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    next_frontier.append(target)
        frontier = sorted(next_frontier, key=positions.get)
    if len(order) != len(ids):
        raise WorkflowError("Workflow contains a cycle.")
    return Plan(order, levels)


def run(workflow: dict[str, Any]) -> dict[str, Any]:
    """Execute a valid Workflow and return its observable Run result."""
    plan = validate(workflow)
    nodes = {node["id"]: node for node in workflow["nodes"]}
    parents = {node_id: [] for node_id in nodes}
    for edge in workflow["edges"]:
        parents[edge["target"]].append(edge["source"])

    outcomes: list[dict[str, Any]] = []
    values: dict[str, Any] = {}
    statuses: dict[str, str] = {}
    for node_id in plan.order:
        started = perf_counter()
        blocked_by = [parent for parent in parents[node_id] if statuses[parent] != "succeeded"]
        if blocked_by:
            outcome = {"node_id": node_id, "status": "skipped", "error": f"Blocked by: {', '.join(blocked_by)}"}
        else:
            try:
                inputs = [values[parent] for parent in parents[node_id]]
                value = _execute_node(nodes[node_id], inputs)
                values[node_id] = value
                outcome = {"node_id": node_id, "status": "succeeded", "output": value}
            except (TypeError, ValueError) as exc:
                outcome = {"node_id": node_id, "status": "failed", "error": str(exc)}
        outcome["duration_ms"] = round((perf_counter() - started) * 1000, 3)
        statuses[node_id] = outcome["status"]
        outcomes.append(outcome)
    status = "failed" if any(item["status"] == "failed" for item in outcomes) else "succeeded"
    return {"status": status, "plan": plan.order, "outcomes": outcomes}


def _execute_node(node: dict[str, Any], inputs: list[Any]) -> Any:
    node_type, config = node["type"], node.get("config", {})
    if node_type == "source":
        value = config.get("value", [])
        if not isinstance(value, list):
            raise ValueError("Source value must be a list.")
        return value
    if node_type == "fail":
        raise ValueError(str(config.get("message", "Node failed as configured.")))
    data = _merged_input(inputs)
    field = config.get("field")
    if node_type == "filter":
        operator, expected = config.get("operator", "gte"), config.get("value")
        checks = {
            "eq": lambda actual: actual == expected,
            "gte": lambda actual: actual >= expected,
            "lte": lambda actual: actual <= expected,
        }
        if operator not in checks or not field:
            raise ValueError("Filter needs a field and eq, gte, or lte operator.")
        return [item for item in data if checks[operator](item.get(field))]
    if node_type == "map":
        if not field or "factor" not in config:
            raise ValueError("Map needs a field and numeric factor.")
        return [{**item, field: item.get(field, 0) * config["factor"]} for item in data]
    if node_type == "aggregate":
        if not field:
            raise ValueError("Aggregate needs a field.")
        values = [item.get(field) for item in data]
        operation = config.get("operation", "sum")
        if operation == "sum":
            return {"operation": operation, "field": field, "value": sum(values), "count": len(values)}
        if operation == "avg":
            return {"operation": operation, "field": field, "value": sum(values) / len(values) if values else 0, "count": len(values)}
        raise ValueError("Aggregate operation must be sum or avg.")
    raise ValueError("Unsupported node type.")


def _merged_input(inputs: list[Any]) -> list[dict[str, Any]]:
    if not inputs:
        raise ValueError("Transform nodes need at least one dependency.")
    merged: list[dict[str, Any]] = []
    for value in inputs:
        if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
            raise ValueError("Transform input must be a list of objects.")
        merged.extend(value)
    return merged

