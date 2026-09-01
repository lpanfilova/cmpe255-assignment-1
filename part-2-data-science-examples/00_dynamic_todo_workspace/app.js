(() => {
  "use strict";

  const STORAGE_KEY = "daymark.tasks.v1";
  const THEME_KEY = "daymark.theme";
  const priorityRank = { high: 0, medium: 1, low: 2 };
  const state = { tasks: loadTasks(), filter: "all", query: "", sort: "created" };

  const $ = (selector) => document.querySelector(selector);
  const elements = {
    form: $("#taskForm"), title: $("#taskTitle"), due: $("#taskDue"), priority: $("#taskPriority"), tag: $("#taskTag"),
    list: $("#taskList"), empty: $("#emptyState"), emptyTitle: $("#emptyTitle"), emptyCopy: $("#emptyCopy"),
    taskCount: $("#taskCount"), allCount: $("#allCount"), activeCount: $("#activeCount"), completedCount: $("#completedCount"),
    remaining: $("#remainingLabel"), clear: $("#clearCompleted"), search: $("#searchInput"), sort: $("#sortSelect"),
    progress: $("#progressRing"), progressValue: $("#progressValue"), progressTitle: $("#progressTitle"), progressSubtitle: $("#progressSubtitle"),
    dialog: $("#editDialog"), editForm: $("#editForm"), editId: $("#editId"), editTitle: $("#editTitle"), editDue: $("#editDue"), editPriority: $("#editPriority"), editTag: $("#editTag"),
    toast: $("#toast")
  };

  function loadTasks() {
    try {
      const value = JSON.parse(localStorage.getItem(STORAGE_KEY));
      return Array.isArray(value) ? value : [];
    } catch { return []; }
  }

  function saveTasks() { localStorage.setItem(STORAGE_KEY, JSON.stringify(state.tasks)); }
  function makeId() { return crypto.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`; }
  function escapeHtml(value) { return String(value).replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]); }
  function localDate(dateString) { return new Date(`${dateString}T00:00:00`); }
  function todayString() { const now = new Date(); return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")}`; }
  function formatDue(dateString) {
    if (!dateString) return "";
    if (dateString === todayString()) return "Today";
    const tomorrow = new Date(); tomorrow.setDate(tomorrow.getDate() + 1);
    const tomorrowString = `${tomorrow.getFullYear()}-${String(tomorrow.getMonth() + 1).padStart(2, "0")}-${String(tomorrow.getDate()).padStart(2, "0")}`;
    if (dateString === tomorrowString) return "Tomorrow";
    return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" }).format(localDate(dateString));
  }
  function isOverdue(task) { return Boolean(task.due && !task.completed && task.due < todayString()); }

  function visibleTasks() {
    const query = state.query.trim().toLowerCase();
    return state.tasks
      .filter((task) => state.filter === "all" || (state.filter === "active" ? !task.completed : task.completed))
      .filter((task) => !query || task.title.toLowerCase().includes(query) || task.tag.toLowerCase().includes(query))
      .sort((a, b) => {
        if (state.sort === "priority") return priorityRank[a.priority] - priorityRank[b.priority] || b.createdAt - a.createdAt;
        if (state.sort === "due") return (a.due || "9999-12-31").localeCompare(b.due || "9999-12-31") || b.createdAt - a.createdAt;
        if (state.sort === "alphabetical") return a.title.localeCompare(b.title);
        return b.createdAt - a.createdAt;
      });
  }

  function render() {
    const tasks = visibleTasks();
    const completed = state.tasks.filter((task) => task.completed).length;
    const active = state.tasks.length - completed;
    const percent = state.tasks.length ? Math.round((completed / state.tasks.length) * 100) : 0;

    elements.list.innerHTML = tasks.map((task) => `
      <article class="task-card ${task.completed ? "completed" : ""}" data-id="${task.id}">
        <button class="check-button" type="button" data-action="toggle" aria-label="${task.completed ? "Mark active" : "Mark complete"}" aria-pressed="${task.completed}">✓</button>
        <div>
          <p class="task-title">${escapeHtml(task.title)}</p>
          <div class="task-meta">
            <span class="badge priority-${task.priority}">${task.priority}</span>
            ${task.due ? `<span class="task-date ${isOverdue(task) ? "overdue" : ""}">${isOverdue(task) ? "Overdue · " : ""}${formatDue(task.due)}</span>` : ""}
            ${task.tag ? `<span>• ${escapeHtml(task.tag)}</span>` : ""}
          </div>
        </div>
        <div class="task-actions">
          <button class="task-action" type="button" data-action="edit" aria-label="Edit ${escapeHtml(task.title)}">Edit</button>
          <button class="task-action" type="button" data-action="delete" aria-label="Delete ${escapeHtml(task.title)}">Delete</button>
        </div>
      </article>`).join("");

    const noResults = state.tasks.length > 0 && tasks.length === 0;
    elements.empty.hidden = tasks.length > 0;
    elements.emptyTitle.textContent = noResults ? "Nothing found" : state.filter === "completed" ? "No completed tasks yet" : "Your slate is clear";
    elements.emptyCopy.textContent = noResults ? "Try another search or filter." : state.filter === "completed" ? "Finished tasks will collect here." : "Add your first task above and start building momentum.";
    elements.taskCount.textContent = state.tasks.length;
    elements.allCount.textContent = state.tasks.length;
    elements.activeCount.textContent = active;
    elements.completedCount.textContent = completed;
    elements.remaining.textContent = `${active} ${active === 1 ? "task" : "tasks"} remaining`;
    elements.clear.disabled = completed === 0;
    elements.progress.style.setProperty("--progress", `${percent * 3.6}deg`);
    elements.progressValue.textContent = `${percent}%`;
    elements.progressTitle.textContent = percent === 100 && state.tasks.length ? "Everything is done" : completed ? "Momentum is building" : "Ready when you are";
    elements.progressSubtitle.textContent = state.tasks.length ? `${completed} of ${state.tasks.length} completed` : "Add a task to begin";
  }

  function addTask(event) {
    event.preventDefault();
    const title = elements.title.value.trim();
    if (!title) return elements.title.focus();
    state.tasks.push({ id: makeId(), title, due: elements.due.value, priority: elements.priority.value, tag: elements.tag.value.trim(), completed: false, createdAt: Date.now() });
    saveTasks(); elements.form.reset(); elements.priority.value = "medium"; render(); elements.title.focus(); showToast("Task added");
  }

  function handleTaskAction(event) {
    const button = event.target.closest("[data-action]");
    const card = event.target.closest("[data-id]");
    if (!button || !card) return;
    const index = state.tasks.findIndex((task) => task.id === card.dataset.id);
    if (index < 0) return;
    if (button.dataset.action === "toggle") {
      state.tasks[index].completed = !state.tasks[index].completed;
      showToast(state.tasks[index].completed ? "Nice work — task completed" : "Task moved back to active");
    } else if (button.dataset.action === "delete") {
      state.tasks.splice(index, 1); showToast("Task deleted");
    } else if (button.dataset.action === "edit") {
      openEditor(state.tasks[index]); return;
    }
    saveTasks(); render();
  }

  function openEditor(task) {
    elements.editId.value = task.id; elements.editTitle.value = task.title; elements.editDue.value = task.due; elements.editPriority.value = task.priority; elements.editTag.value = task.tag;
    elements.dialog.showModal(); setTimeout(() => elements.editTitle.focus(), 0);
  }

  function saveEdit(event) {
    event.preventDefault();
    const task = state.tasks.find((item) => item.id === elements.editId.value);
    const title = elements.editTitle.value.trim();
    if (!task || !title) return;
    Object.assign(task, { title, due: elements.editDue.value, priority: elements.editPriority.value, tag: elements.editTag.value.trim() });
    saveTasks(); elements.dialog.close(); render(); showToast("Changes saved");
  }

  let toastTimer;
  function showToast(message) { clearTimeout(toastTimer); elements.toast.textContent = message; elements.toast.classList.add("show"); toastTimer = setTimeout(() => elements.toast.classList.remove("show"), 2200); }

  function initializeTheme() {
    const saved = localStorage.getItem(THEME_KEY);
    const theme = saved || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    document.documentElement.dataset.theme = theme;
  }

  elements.form.addEventListener("submit", addTask);
  elements.list.addEventListener("click", handleTaskAction);
  elements.search.addEventListener("input", (event) => { state.query = event.target.value; render(); });
  elements.sort.addEventListener("change", (event) => { state.sort = event.target.value; render(); });
  document.querySelector(".filters").addEventListener("click", (event) => {
    const button = event.target.closest("[data-filter]"); if (!button) return;
    state.filter = button.dataset.filter; document.querySelectorAll(".filter").forEach((item) => item.classList.toggle("active", item === button)); render();
  });
  elements.clear.addEventListener("click", () => { state.tasks = state.tasks.filter((task) => !task.completed); saveTasks(); render(); showToast("Completed tasks cleared"); });
  elements.editForm.addEventListener("submit", saveEdit);
  $("#closeDialog").addEventListener("click", () => elements.dialog.close());
  $("#cancelEdit").addEventListener("click", () => elements.dialog.close());
  $("#themeToggle").addEventListener("click", () => { const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark"; document.documentElement.dataset.theme = theme; localStorage.setItem(THEME_KEY, theme); });
  elements.dialog.addEventListener("click", (event) => { if (event.target === elements.dialog) elements.dialog.close(); });

  $("#todayLabel").textContent = new Intl.DateTimeFormat(undefined, { weekday: "long", month: "long", day: "numeric" }).format(new Date());
  elements.due.min = todayString();
  initializeTheme();
  render();
})();
