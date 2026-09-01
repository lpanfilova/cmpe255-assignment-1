(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.FlowGraph = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  function addNode(workflow, type) {
    const prefix = type || "source";
    let suffix = workflow.nodes.length + 1;
    while (workflow.nodes.some((node) => node.id === `${prefix}-${suffix}`)) suffix += 1;
    const defaults = {
      source: { value: [{ region: "west", revenue: 120 }, { region: "east", revenue: 80 }] },
      filter: { field: "revenue", operator: "gte", value: 100 },
      map: { field: "revenue", factor: 1.1 },
      aggregate: { field: "revenue", operation: "sum" },
      fail: { message: "Intentional demonstration failure" },
    };
    return { ...workflow, nodes: [...workflow.nodes, { id: `${prefix}-${suffix}`, type: prefix, config: defaults[prefix] }] };
  }

  function addEdge(workflow, source, target) {
    if (!source || !target || source === target) return workflow;
    if (workflow.edges.some((edge) => edge.source === source && edge.target === target)) return workflow;
    return { ...workflow, edges: [...workflow.edges, { source, target }] };
  }

  function removeNode(workflow, nodeId) {
    return {
      ...workflow,
      nodes: workflow.nodes.filter((node) => node.id !== nodeId),
      edges: workflow.edges.filter((edge) => edge.source !== nodeId && edge.target !== nodeId),
    };
  }

  return { addNode, addEdge, removeNode };
});

