import pytest

from flowforge.engine import WorkflowError, run, validate


def workflow(nodes, edges):
    return {"name": "test", "nodes": nodes, "edges": edges}


def test_plan_is_stable_and_groups_parallel_nodes():
    graph = workflow(
        [
            {"id": "source", "type": "source", "config": {"value": []}},
            {"id": "left", "type": "filter", "config": {"field": "x", "value": 0}},
            {"id": "right", "type": "map", "config": {"field": "x", "factor": 2}},
        ],
        [{"source": "source", "target": "left"}, {"source": "source", "target": "right"}],
    )
    plan = validate(graph)
    assert plan.order == ["source", "left", "right"]
    assert plan.levels == [["source"], ["left", "right"]]


def test_cycle_is_rejected_with_domain_message():
    graph = workflow(
        [{"id": "a", "type": "source"}, {"id": "b", "type": "source"}],
        [{"source": "a", "target": "b"}, {"source": "b", "target": "a"}],
    )
    with pytest.raises(WorkflowError, match="cycle"):
        validate(graph)


def test_run_transforms_data_and_records_worked_result():
    graph = workflow(
        [
            {"id": "source", "type": "source", "config": {"value": [{"x": 2}, {"x": 4}]}},
            {"id": "double", "type": "map", "config": {"field": "x", "factor": 2}},
            {"id": "total", "type": "aggregate", "config": {"field": "x", "operation": "sum"}},
        ],
        [{"source": "source", "target": "double"}, {"source": "double", "target": "total"}],
    )
    result = run(graph)
    assert result["status"] == "succeeded"
    assert result["outcomes"][-1]["output"] == {"operation": "sum", "field": "x", "value": 12, "count": 2}


def test_failed_node_skips_its_descendant():
    graph = workflow(
        [
            {"id": "boom", "type": "fail", "config": {"message": "bad input"}},
            {"id": "after", "type": "aggregate", "config": {"field": "x"}},
        ],
        [{"source": "boom", "target": "after"}],
    )
    result = run(graph)
    assert [item["status"] for item in result["outcomes"]] == ["failed", "skipped"]
    assert result["outcomes"][0]["error"] == "bad input"

