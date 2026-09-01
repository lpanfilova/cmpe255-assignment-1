const sample = {
  name: "Revenue quality workflow",
  nodes: [
    { id: "orders", type: "source", config: { value: [{ region: "west", revenue: 120 }, { region: "east", revenue: 80 }, { region: "north", revenue: 200 }] } },
    { id: "qualified", type: "filter", config: { field: "revenue", operator: "gte", value: 100 } },
    { id: "forecast", type: "map", config: { field: "revenue", factor: 1.1 } },
    { id: "total", type: "aggregate", config: { field: "revenue", operation: "sum" } },
  ],
  edges: [
    { source: "orders", target: "qualified" },
    { source: "qualified", target: "forecast" },
    { source: "forecast", target: "total" },
  ],
};

let workflow = structuredClone(sample);
let selected = workflow.nodes[0].id;
const $ = (selector) => document.querySelector(selector);
const escapeHtml = (value) => String(value).replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]);

function render() {
  $("#workflow-name").value = workflow.name;
  $("#canvas").innerHTML = workflow.nodes.map((node, index) => `
    <button class="node ${node.id === selected ? "selected" : ""}" data-node="${escapeHtml(node.id)}" style="--i:${index}">
      <span class="node-type">${escapeHtml(node.type)}</span><strong>${escapeHtml(node.id)}</strong>
      <small>${workflow.edges.filter((edge) => edge.target === node.id).length} dependencies</small>
    </button>`).join("");
  $("#edge-list").innerHTML = workflow.edges.map((edge) => `<li>${escapeHtml(edge.source)} <span>→</span> ${escapeHtml(edge.target)}</li>`).join("") || "<li>No edges yet</li>";
  const options = workflow.nodes.map((node) => `<option value="${escapeHtml(node.id)}">${escapeHtml(node.id)}</option>`).join("");
  $("#edge-source").innerHTML = options;
  $("#edge-target").innerHTML = options;
  const node = workflow.nodes.find((item) => item.id === selected);
  $("#inspector-empty").hidden = Boolean(node);
  $("#inspector-form").hidden = !node;
  if (node) {
    $("#node-id").value = node.id;
    $("#node-type").textContent = node.type;
    $("#node-config").value = JSON.stringify(node.config, null, 2);
  }
  document.querySelectorAll("[data-node]").forEach((button) => button.addEventListener("click", () => { selected = button.dataset.node; render(); }));
}

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Request failed");
  return payload;
}

function notice(message, tone = "ok") {
  const element = $("#notice");
  element.textContent = message;
  element.className = `notice ${tone}`;
}

$("#add-node").addEventListener("click", () => {
  workflow = FlowGraph.addNode(workflow, $("#new-type").value);
  selected = workflow.nodes.at(-1).id;
  render();
});
$("#add-edge").addEventListener("click", () => { workflow = FlowGraph.addEdge(workflow, $("#edge-source").value, $("#edge-target").value); render(); });
$("#delete-node").addEventListener("click", () => { workflow = FlowGraph.removeNode(workflow, selected); selected = workflow.nodes[0]?.id; render(); });
$("#apply-config").addEventListener("click", () => {
  try {
    const config = JSON.parse($("#node-config").value);
    workflow = { ...workflow, nodes: workflow.nodes.map((node) => node.id === selected ? { ...node, config } : node) };
    notice("Node configuration applied."); render();
  } catch { notice("Configuration must be valid JSON.", "error"); }
});
$("#workflow-name").addEventListener("input", (event) => { workflow.name = event.target.value; });
$("#validate").addEventListener("click", async () => {
  try { const result = await api("/api/validate", { method: "POST", body: JSON.stringify(workflow) }); notice(`Valid plan: ${result.plan.join(" → ")}`); }
  catch (error) { notice(error.message, "error"); }
});
$("#save").addEventListener("click", async () => {
  try { workflow = await api("/api/workflows", { method: "POST", body: JSON.stringify(workflow) }); notice(`Saved version ${workflow.version}.`); render(); }
  catch (error) { notice(error.message, "error"); }
});
$("#run").addEventListener("click", async () => {
  try {
    if (!workflow.id) workflow = await api("/api/workflows", { method: "POST", body: JSON.stringify(workflow) });
    const result = await api(`/api/workflows/${workflow.id}/runs`, { method: "POST" });
    $("#results").innerHTML = result.outcomes.map((item) => `<article class="outcome ${item.status}"><div><strong>${escapeHtml(item.node_id)}</strong><span>${item.status} · ${item.duration_ms} ms</span></div><pre>${escapeHtml(JSON.stringify(item.output ?? item.error, null, 2))}</pre></article>`).join("");
    notice(`Run ${result.status}: ${result.id}.`, result.status === "succeeded" ? "ok" : "error");
  } catch (error) { notice(error.message, "error"); }
});
$("#reset").addEventListener("click", () => { workflow = structuredClone(sample); selected = workflow.nodes[0].id; $("#results").innerHTML = '<p class="muted">Run the workflow to inspect outcomes.</p>'; render(); notice("Sample restored."); });

render();

