# FlowForge

FlowForge is a local workflow studio for authoring and executing deterministic directed acyclic graphs.

## Language

**Workflow**:
A named, versioned graph that can be validated, saved, and run.
_Avoid_: Pipeline, job

**Node**:
A single executable step in a Workflow with a type and configuration.
_Avoid_: Task, block

**Edge**:
A dependency from one Node to another; its target cannot run before its source succeeds.
_Avoid_: Link, connection

**Run**:
An immutable record of one Workflow execution, including ordered Node outcomes.
_Avoid_: Execution, attempt

**Plan**:
The stable topological order produced for a valid Workflow before a Run begins.
_Avoid_: Schedule, queue

