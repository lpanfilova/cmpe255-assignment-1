const test = require("node:test");
const assert = require("node:assert/strict");
const { addNode, addEdge, removeNode } = require("../static/graph-utils.js");

test("browser graph authoring adds nodes and unique edges", () => {
  let workflow = { nodes: [], edges: [] };
  workflow = addNode(workflow, "source");
  workflow = addNode(workflow, "filter");
  workflow = addEdge(workflow, workflow.nodes[0].id, workflow.nodes[1].id);
  workflow = addEdge(workflow, workflow.nodes[0].id, workflow.nodes[1].id);
  assert.equal(workflow.nodes.length, 2);
  assert.deepEqual(workflow.edges, [{ source: "source-1", target: "filter-2" }]);
});

test("removing a node also removes incident edges", () => {
  const workflow = { nodes: [{ id: "a" }, { id: "b" }], edges: [{ source: "a", target: "b" }] };
  assert.deepEqual(removeNode(workflow, "a"), { nodes: [{ id: "b" }], edges: [] });
});

