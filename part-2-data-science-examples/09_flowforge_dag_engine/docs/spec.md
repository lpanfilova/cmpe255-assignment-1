# FlowForge DAG Engine specification

## Problem Statement

Students need a compact full-stack example that makes DAG validation, dependency ordering, persistence, and execution observable without cloud infrastructure.

## Solution

Provide a local workflow studio where a user edits a graph, validates it, saves it, runs it, and inspects ordered Node outcomes and Run history.

## User Stories

1. As a workflow author, I want to add and remove Nodes and Edges so that I can model dependencies.
2. As a workflow author, I want cycles, dangling Edges, duplicate IDs, and invalid configuration rejected with useful messages.
3. As a workflow author, I want a stable Plan preview so that execution order is understandable.
4. As a workflow author, I want to save and reload Workflows so that designs survive server restarts.
5. As a workflow author, I want to run built-in transform Nodes so that the graph produces inspectable data.
6. As a workflow author, I want downstream Nodes skipped after a failure so that dependency semantics are visible.
7. As a workflow author, I want Run history and per-Node timing so that behavior can be audited.
8. As a learner, I want a seeded example so that the application is useful immediately.

## Implementation Decisions

- The DAG module exposes validation, planning, and synchronous local execution behind a small interface.
- Supported Node types are source, map, filter, aggregate, and fail; their constrained configurations avoid arbitrary code execution.
- A SQLite adapter persists Workflow and Run JSON through one repository interface.
- Flask serves both the JSON interface and a dependency-free browser application.
- Plans are deterministic: Nodes with equal readiness are ordered by their original Workflow order.

## Testing Decisions

- Test the DAG module through its public functions with worked examples for order, transforms, failures, and invalid graphs.
- Test persistence and user flows through the HTTP JSON interface using Flask's test client.
- Test browser graph manipulation through dependency-free JavaScript functions using Node's test runner.
- Tests assert externally observable behavior and do not inspect private helpers or database tables.

## Out of Scope

Authentication, arbitrary user code, distributed workers, retries, cron scheduling, collaboration, cloud deployment, and production observability.

## Further Notes

The project demonstrates Matt Pocock's domain-modeling, codebase-design, specification, TDD, implementation, and review disciplines while remaining laptop-friendly.

