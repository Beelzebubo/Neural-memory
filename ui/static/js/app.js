(function () {
  'use strict';

  const API_BASE = '/api';
  const PAGE_SIZE = 50;
  let currentOffset = 0;
  let currentTotal = 0;
  let currentMemoryId = null;
  let searchMode = 'semantic';
  let searchQuery = '';

  // ── API Helpers ──
  async function api(method, path, body) {
    const opts = {
      method,
      headers: { 'Accept': 'application/json' },
    };
    if (body !== undefined) {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(`${API_BASE}${path}`, opts);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
  }

  function getApi(path) { return api('GET', path); }
  function postApi(path, body) { return api('POST', path, body); }
  function putApi(path, body) { return api('PUT', path, body); }
  function patchApi(path, body) { return api('PATCH', path, body); }
  function delApi(path) { return api('DELETE', path); }

  // ── State ──
  let state = {
    memories: [],
    total: 0,
    sidebarMemories: [],
    tags: {},
    stats: null,
    config: null,
    currentDetail: null,
  };

  // ── View Routing ──
  function showView(viewId) {
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    const view = document.getElementById(viewId);
    if (view) view.classList.add('active');
  }

  // ── Formatting ──
  function fmtTime(ts) {
    if (!ts) return '—';
    const d = new Date(ts * 1000);
    const now = new Date();
    const diff = now - d;
    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  }

  function fmtImportance(score) {
    const v = typeof score === 'number' ? score : 0.5;
    let cls = 'importance-low';
    if (v >= 0.7) cls = 'importance-high';
    else if (v >= 0.4) cls = 'importance-medium';
    return `<span class="importance-badge ${cls}">${v.toFixed(2)}</span>`;
  }

  function truncate(text, len = 80) {
    if (!text) return '';
    return text.length > len ? text.slice(0, len) + '…' : text;
  }

  function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
  }

  function fmtTags(tags) {
    if (!tags || !Array.isArray(tags) || tags.length === 0) return '';
    return tags.map(t => `<span class="tag-badge">${escapeHtml(t)}</span>`).join('');
  }

  // ── Error display ──
  function showError(msg) {
    const banner = document.getElementById('errorBanner');
    document.getElementById('errorMessage').textContent = msg;
    banner.classList.remove('hidden');
  }

  function hideError() {
    document.getElementById('errorBanner').classList.add('hidden');
  }

  // ── Loading ──
  function showLoading() {
    document.getElementById('loadingState').classList.remove('hidden');
    document.getElementById('tableContainer').classList.add('hidden');
    document.getElementById('emptyState').classList.add('hidden');
    document.getElementById('searchEmpty').classList.add('hidden');
  }

  function hideLoading() {
    document.getElementById('loadingState').classList.add('hidden');
  }

  // ── Warning Banner ──
  function showWarning(msg) {
    const container = document.getElementById('browserWarnings');
    container.innerHTML = `<div class="warning-banner"><i class="ph ph-warning-circle"></i> ${escapeHtml(msg)}</div>`;
  }

  function clearWarnings() {
    document.getElementById('browserWarnings').innerHTML = '';
  }

  // ── Sidebar ──
  async function loadSidebar() {
    try {
      const st = await getApi('/stats');
      state.stats = st;
      document.getElementById('statDim').textContent = st.dimension || '—';
      document.getElementById('statTotal').textContent = st.total;
      document.getElementById('statCapacity').textContent = st.capacity || '—';
      document.getElementById('sidebarCount').textContent = st.total;
      document.getElementById('headerCount').textContent = st.total;
      document.getElementById('statusDot').className = 'status-indicator' + (st.status === 'degraded' ? ' degraded' : '');
      document.getElementById('statusText').textContent = st.status === 'degraded' ? 'Degraded (no embedder)' : 'Online';
      document.getElementById('statusPath').textContent = st.store_path || '';
      if (st.capacity && st.total >= st.capacity * 0.9) {
        showWarning(`Store at capacity (${st.total}/${st.capacity}). Prune to free space.`);
      }
    } catch (e) {
      console.error('Failed to load stats', e);
    }

    try {
      const data = await getApi(`/memories?limit=200&offset=0`);
      state.sidebarMemories = data.memories || [];
      renderSidebarList();
      extractTags(data.memories || []);
    } catch (e) {
      console.error('Failed to load sidebar', e);
    }
  }

  function renderSidebarList() {
    const list = document.getElementById('sidebarList');
    const items = state.sidebarMemories;
    if (!items || items.length === 0) {
      list.innerHTML = '<div class="sidebar-item" style="color:var(--text-muted);font-size:12px;cursor:default">No memories</div>';
      return;
    }
    list.innerHTML = items.map(m => {
      const active = m.id === currentMemoryId ? ' active' : '';
      const imp = typeof m.metadata?.importance_score === 'number' ? m.metadata.importance_score : 0.5;
      return `<div class="sidebar-item${active}" data-id="${escapeHtml(m.id)}">
        <span class="sidebar-item-icon"><i class="ph ph-file-text"></i></span>
        <span class="sidebar-item-text">${escapeHtml(truncate(m.text || m.id, 30))}</span>
        <span class="sidebar-item-priority" data-id="${escapeHtml(m.id)}">${fmtImportance(imp)}</span>
        <span class="sidebar-item-time">${fmtTime(m.metadata?.timestamp)}</span>
      </div>`;
    }).join('');
    list.querySelectorAll('.sidebar-item').forEach(el => {
      el.addEventListener('click', () => {
        navigateToDetail(el.dataset.id);
      });
    });
  }

  function extractTags(memories) {
    const tagMap = {};
    memories.forEach(m => {
      const tags = m.metadata?.tags;
      if (Array.isArray(tags)) {
        tags.forEach(t => { tagMap[t] = (tagMap[t] || 0) + 1; });
      }
    });
    state.tags = tagMap;
    renderTags();
  }

  function renderTags() {
    const list = document.getElementById('tagsList');
    const entries = Object.entries(state.tags);
    if (entries.length === 0) {
      list.innerHTML = '<div class="tag-item" style="color:var(--text-muted);font-size:12px;cursor:default">No tags</div>';
      return;
    }
    list.innerHTML = entries.map(([tag, count]) =>
      `<div class="tag-item" data-tag="${escapeHtml(tag)}">
        <span>#${escapeHtml(tag)}</span>
        <span class="tag-count">${count}</span>
      </div>`
    ).join('');
    list.querySelectorAll('.tag-item').forEach(el => {
      el.addEventListener('click', () => {
        document.getElementById('searchInput').value = `#${el.dataset.tag}`;
        searchQuery = `#${el.dataset.tag}`;
        searchMode = 'keyword';
        document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
        document.querySelector('.toggle-btn[data-mode="keyword"]').classList.add('active');
        loadMemories();
      });
    });
  }

  // ── Tags collapsible ──
  document.getElementById('tagsHeader').addEventListener('click', function () {
    this.classList.toggle('collapsed');
    document.getElementById('tagsList').classList.toggle('hidden');
  });

  // ── Browser View ──
  async function loadMemories() {
    hideError();
    clearWarnings();
    showLoading();
    currentOffset = 0;

    try {
      let data;
      if (searchQuery) {
        if (searchMode === 'semantic') {
          data = await postApi('/search', { query: searchQuery, k: PAGE_SIZE, threshold: 0 });
          data = { memories: data.results.map(r => ({
            id: r.id,
            text: r.text,
            metadata: r.metadata,
            score: r.score,
          })), total: data.results.length };
        } else {
          const [allData, filterData] = await Promise.all([
            getApi(`/memories?limit=500&offset=0`),
            (searchQuery.startsWith('#')) ? postApi('/filter', { tags: [searchQuery.slice(1)] }) : Promise.resolve(null),
          ]);
          if (filterData && filterData.results) {
            data = { memories: filterData.results, total: filterData.results.length };
          } else {
            const q = searchQuery.toLowerCase();
            const filtered = (allData.memories || []).filter(m => {
              const text = (m.text || '').toLowerCase();
              const src = (m.metadata?.source || '').toLowerCase();
              return text.includes(q) || src.includes(q);
            });
            data = { memories: filtered, total: filtered.length };
          }
        }
      } else {
        data = await getApi(`/memories?limit=${PAGE_SIZE}&offset=0`);
      }

      state.memories = data.memories || [];
      state.total = data.total || state.memories.length;
      hideLoading();
      renderTable();
    } catch (e) {
      hideLoading();
      showError(e.message);
    }
  }

  function renderTable() {
    const body = document.getElementById('memoriesBody');
    const container = document.getElementById('tableContainer');
    const empty = document.getElementById('emptyState');
    const searchEmpty = document.getElementById('searchEmpty');

    if (state.memories.length === 0 && !searchQuery) {
      container.classList.add('hidden');
      empty.classList.remove('hidden');
      searchEmpty.classList.add('hidden');
      document.getElementById('pagination').classList.add('hidden');
      return;
    }

    if (state.memories.length === 0 && searchQuery) {
      container.classList.add('hidden');
      empty.classList.add('hidden');
      searchEmpty.classList.remove('hidden');
      document.getElementById('pagination').classList.add('hidden');
      return;
    }

    container.classList.remove('hidden');
    empty.classList.add('hidden');
    searchEmpty.classList.add('hidden');

    body.innerHTML = state.memories.map(m => {
      const meta = m.metadata || {};
      return `<tr data-id="${escapeHtml(m.id)}">
        <td class="col-check"><input type="checkbox" class="row-check" data-id="${escapeHtml(m.id)}"></td>
        <td class="col-text"><span class="text-preview" title="${escapeHtml(m.text || '')}">${escapeHtml(truncate(m.text || '(no text)', 80))}</span></td>
        <td class="col-source"><span class="source-badge">${escapeHtml(meta.source || 'unknown')}</span></td>
        <td class="col-importance">${fmtImportance(meta.importance_score)}</td>
        <td class="col-time">${fmtTime(meta.timestamp)}</td>
        <td class="col-tags">${fmtTags(meta.tags)}</td>
        <td class="col-actions">
          <button class="action-btn edit-btn" data-id="${escapeHtml(m.id)}" title="Edit"><i class="ph ph-pencil-simple"></i></button>
          <button class="action-btn danger delete-btn" data-id="${escapeHtml(m.id)}" title="Delete"><i class="ph ph-trash"></i></button>
        </td>
      </tr>`;
    }).join('');

    body.querySelectorAll('tr').forEach(tr => {
      tr.addEventListener('click', (e) => {
        if (e.target.closest('.action-btn') || e.target.closest('.row-check')) return;
        navigateToDetail(tr.dataset.id);
      });
    });
    body.querySelectorAll('.edit-btn').forEach(btn => {
      btn.addEventListener('click', (e) => { e.stopPropagation(); navigateToDetail(btn.dataset.id); });
    });
    body.querySelectorAll('.delete-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (confirm('Delete this memory?')) {
          deleteMemory(btn.dataset.id).then(() => loadMemories());
        }
      });
    });

    renderPagination();
  }

  function renderPagination() {
    const pag = document.getElementById('pagination');
    if (state.total <= PAGE_SIZE && !searchQuery) {
      pag.classList.add('hidden');
      return;
    }
    pag.classList.remove('hidden');
    const page = Math.floor(currentOffset / PAGE_SIZE) + 1;
    const totalPages = Math.ceil(state.total / PAGE_SIZE);
    document.getElementById('pageInfo').textContent = `Page ${page} of ${totalPages || 1}`;
    document.getElementById('prevPage').disabled = currentOffset <= 0;
    document.getElementById('nextPage').disabled = currentOffset + PAGE_SIZE >= state.total;
  }

  // ── Navigation ──
  function navigateToDetail(id) {
    currentMemoryId = id;
    showView('view-detail');
    loadDetail(id);
    renderSidebarList();
  }

  function navigateToBrowser() {
    currentMemoryId = null;
    showView('view-browser');
    loadMemories();
    loadSidebar();
  }

  // ── Detail View ──
  async function loadDetail(id) {
    const content = document.getElementById('detailContent');
    const loading = document.querySelector('.detail-loading');
    const confirm = document.getElementById('deleteConfirm');
    confirm.classList.add('hidden');
    content.classList.add('hidden');
    loading.classList.remove('hidden');

    try {
      const mem = await getApi(`/memories/${id}`);
      state.currentDetail = mem;
      loading.classList.add('hidden');
      content.classList.remove('hidden');
      renderDetail(mem);
    } catch (e) {
      loading.classList.add('hidden');
      showError(e.message);
    }
  }

  function renderDetail(mem) {
    const meta = mem.metadata || {};
    document.getElementById('detailId').textContent = mem.id;
    document.getElementById('detailText').value = mem.text || '';
    document.getElementById('detailSource').value = meta.source || '';
    const imp = typeof meta.importance_score === 'number' ? meta.importance_score : 0.5;
    document.getElementById('detailImportance').value = imp;
    document.getElementById('detailImportanceVal').textContent = imp.toFixed(2);
    document.getElementById('detailTags').value = Array.isArray(meta.tags) ? meta.tags.join(', ') : '';
    const emb = mem.embedding;
    if (emb && Array.isArray(emb)) {
      document.getElementById('detailEmbeddingDim').textContent = `[${emb.length}-dim]`;
      document.getElementById('detailEmbedding').textContent = `[${emb.slice(0, 8).map(v => v.toFixed(4)).join(', ')}, ...]`;
    } else {
      document.getElementById('detailEmbeddingDim').textContent = '';
      document.getElementById('detailEmbedding').textContent = '(embedding not loaded)';
    }
    document.getElementById('detailAccessCount').textContent = `Access count: ${meta.access_count || 0}`;
    document.getElementById('detailTimestamp').textContent = `Created: ${fmtTime(meta.timestamp)}`;

    // Load similar memories
    loadSimilar(mem);
  }

  async function loadSimilar(mem) {
    const container = document.getElementById('similarMemories');
    if (!mem.text) {
      container.innerHTML = '<div class="text-secondary">No text to search with</div>';
      return;
    }
    try {
      const data = await postApi('/search', { query: mem.text, k: 5, threshold: 0.3 });
      const results = (data.results || []).filter(r => r.id !== mem.id);
      if (results.length === 0) {
        container.innerHTML = '<div class="text-secondary">No similar memories found</div>';
        return;
      }
      container.innerHTML = results.map(r =>
        `<div class="similar-item" data-id="${escapeHtml(r.id)}">
          <div class="similar-item-text">${escapeHtml(truncate(r.text || '(no text)', 60))}</div>
          <div class="similar-item-score">Similarity: ${(r.score * 100).toFixed(1)}%</div>
        </div>`
      ).join('');
      container.querySelectorAll('.similar-item').forEach(el => {
        el.addEventListener('click', () => navigateToDetail(el.dataset.id));
      });
    } catch (e) {
      container.innerHTML = `<div class="text-secondary">Failed to load: ${escapeHtml(e.message)}</div>`;
    }
  }

  // ── Detail Edit ──
  async function saveDetail() {
    const mem = state.currentDetail;
    if (!mem) return;
    const tags = document.getElementById('detailTags').value.split(',').map(t => t.trim()).filter(Boolean);
    const body = {
      text: document.getElementById('detailText').value,
      source: document.getElementById('detailSource').value || 'manual',
      importance: parseFloat(document.getElementById('detailImportance').value) || 0.5,
      tags: tags.length > 0 ? tags : undefined,
    };
    try {
      await putApi(`/memories/${mem.id}`, body);
      await loadDetail(mem.id);
      loadSidebar();
    } catch (e) {
      showError(e.message);
    }
  }

  async function deleteMemory(id) {
    try {
      await delApi(`/memories/${id}`);
      navigateToBrowser();
      loadSidebar();
    } catch (e) {
      showError(e.message);
    }
  }

  // ── Config View ──
  async function loadConfig() {
    const content = document.getElementById('configContent');
    const loading = document.querySelector('.config-loading');
    content.classList.add('hidden');
    loading.classList.remove('hidden');

    try {
      const cfg = await getApi('/config');
      state.config = cfg;
      loading.classList.add('hidden');
      content.classList.remove('hidden');
      renderConfig(cfg);
    } catch (e) {
      loading.classList.add('hidden');
      showError(e.message);
    }
  }

  function renderConfig(cfg) {
    const emb = cfg.embedder || {};
    const store = cfg.store || {};
    const ret = cfg.retriever || {};
    const con = cfg.consolidator || {};

    document.getElementById('cfgModelName').value = emb.model_name || 'all-MiniLM-L6-v2';
    document.getElementById('cfgModelName').disabled = false;
    const dim = state.stats?.dimension || 384;
    document.getElementById('cfgDimension').value = dim;
    document.getElementById('cfgCapacity').value = con.max_memories || 10000;
    document.getElementById('cfgStrategy').value = con.prune_strategy || 'hybrid';
    const alpha = typeof ret.hybrid_alpha === 'number' ? ret.hybrid_alpha : 0.5;
    document.getElementById('cfgAlpha').value = alpha;
    document.getElementById('cfgAlphaVal').textContent = alpha.toFixed(2);
    document.getElementById('cfgMaxMemories').value = con.max_memories || 10000;
    document.getElementById('cfgMinImportance').value = con.similarity_threshold || 0.85;
  }

  // ── Search Input ──
  let searchTimer = null;
  document.getElementById('searchInput').addEventListener('input', function () {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      searchQuery = this.value.trim();
      if (searchMode === 'keyword') {
        loadMemories();
      } else if (searchQuery) {
        loadMemories();
      } else {
        loadMemories();
      }
    }, 300);
  });

  document.querySelectorAll('.toggle-btn').forEach(btn => {
    btn.addEventListener('click', function () {
      document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
      this.classList.add('active');
      searchMode = this.dataset.mode;
      if (searchQuery) loadMemories();
    });
  });

  // ── Pagination ──
  document.getElementById('prevPage').addEventListener('click', async () => {
    if (currentOffset <= 0) return;
    currentOffset = Math.max(0, currentOffset - PAGE_SIZE);
    try {
      hideError();
      showLoading();
      const data = await getApi(`/memories?limit=${PAGE_SIZE}&offset=${currentOffset}`);
      state.memories = data.memories || [];
      state.total = data.total || 0;
      hideLoading();
      renderTable();
    } catch (e) {
      hideLoading();
      showError(e.message);
    }
  });

  document.getElementById('nextPage').addEventListener('click', async () => {
    currentOffset += PAGE_SIZE;
    try {
      hideError();
      showLoading();
      const data = await getApi(`/memories?limit=${PAGE_SIZE}&offset=${currentOffset}`);
      state.memories = data.memories || [];
      state.total = data.total || 0;
      hideLoading();
      renderTable();
    } catch (e) {
      hideLoading();
      showError(e.message);
    }
  });

  // ── Select All ──
  document.getElementById('selectAll').addEventListener('change', function () {
    document.querySelectorAll('.row-check').forEach(cb => cb.checked = this.checked);
  });

  // ── Modal ──
  const modal = document.getElementById('addModal');
  function openModal() { modal.classList.remove('hidden'); }
  function closeModal() {
    modal.classList.add('hidden');
    document.getElementById('modalText').value = '';
    document.getElementById('modalSource').value = '';
    document.getElementById('modalImportance').value = 0.5;
    document.getElementById('modalImportanceVal').textContent = '0.50';
    document.getElementById('modalTags').value = '';
  }

  document.getElementById('btnAddMemory').addEventListener('click', openModal);
  document.getElementById('btnEmptyAdd').addEventListener('click', openModal);
  document.getElementById('modalClose').addEventListener('click', closeModal);
  document.getElementById('modalCancel').addEventListener('click', closeModal);
  modal.addEventListener('click', function (e) { if (e.target === modal) closeModal(); });

  function updateSliderFill(slider) {
    const pct = ((parseFloat(slider.value) - parseFloat(slider.min)) / (parseFloat(slider.max) - parseFloat(slider.min))) * 100;
    slider.style.setProperty('--slider-pct', pct + '%');
  }

  document.getElementById('modalImportance').addEventListener('input', function () {
    document.getElementById('modalImportanceVal').textContent = parseFloat(this.value).toFixed(2);
    updateSliderFill(this);
  });

  document.getElementById('detailImportance').addEventListener('input', function () {
    document.getElementById('detailImportanceVal').textContent = parseFloat(this.value).toFixed(2);
    updateSliderFill(this);
  });

  // Priority auto-save
  let prioritySaveTimer = null;
  let previousPriority = null;
  document.getElementById('detailImportance').addEventListener('change', function () {
    const slider = this;
    const newVal = parseFloat(slider.value);
    if (previousPriority === null) {
      previousPriority = parseFloat(document.getElementById('detailImportanceVal').textContent) || 0.5;
    }
    clearTimeout(prioritySaveTimer);
    prioritySaveTimer = setTimeout(() => savePriority(slider, newVal), 500);
  });

  async function savePriority(slider, newVal) {
    let indicator = document.querySelector('.priority-save-indicator');
    if (!indicator) {
      indicator = document.createElement('span');
      indicator.className = 'priority-save-indicator';
      slider.parentNode.appendChild(indicator);
    }
    indicator.textContent = 'saving…';
    try {
      await patchApi(`/memories/${state.currentDetail.id}/priority`, { priority: newVal });
      indicator.textContent = '✓ saved';
      previousPriority = newVal;
      setTimeout(() => { indicator.textContent = ''; }, 2000);

      // Update sidebar entry
      const sidebarBadge = document.querySelector(`.sidebar-item-priority[data-id="${state.currentDetail.id}"]`);
      if (sidebarBadge) sidebarBadge.innerHTML = fmtImportance(newVal);

      // Update table row if visible
      const tableRow = document.querySelector(`tr[data-id="${state.currentDetail.id}"] .col-importance`);
      if (tableRow) tableRow.innerHTML = fmtImportance(newVal);
    } catch (e) {
      indicator.textContent = '✗ failed';
      slider.value = previousPriority;
      document.getElementById('detailImportanceVal').textContent = previousPriority.toFixed(2);
      setTimeout(() => { indicator.textContent = ''; }, 3000);
    }
  }
  document.getElementById('cfgAlpha').addEventListener('input', function () {
    document.getElementById('cfgAlphaVal').textContent = parseFloat(this.value).toFixed(2);
  });

  document.getElementById('modalSave').addEventListener('click', async function () {
    const text = document.getElementById('modalText').value.trim();
    if (!text) return;
    const tags = document.getElementById('modalTags').value.split(',').map(t => t.trim()).filter(Boolean);
    const body = {
      text,
      source: document.getElementById('modalSource').value || 'manual',
      importance: parseFloat(document.getElementById('modalImportance').value) || 0.5,
      tags: tags.length > 0 ? tags : undefined,
    };
    try {
      await postApi('/memories', body);
      closeModal();
      navigateToBrowser();
      loadSidebar();
    } catch (e) {
      showError(e.message);
    }
  });

  // ── Detail Buttons ──
  document.getElementById('btnBackToList').addEventListener('click', navigateToBrowser);
  document.getElementById('btnSaveMemory').addEventListener('click', saveDetail);

  document.getElementById('btnDeleteMemory').addEventListener('click', function () {
    document.getElementById('deleteConfirm').classList.remove('hidden');
  });
  document.getElementById('cancelDelete').addEventListener('click', function () {
    document.getElementById('deleteConfirm').classList.add('hidden');
  });
  document.getElementById('confirmDelete').addEventListener('click', function () {
    if (state.currentDetail) {
      deleteMemory(state.currentDetail.id);
    }
  });

  document.getElementById('copyEmbedding').addEventListener('click', function () {
    const text = document.getElementById('detailEmbedding').textContent;
    navigator.clipboard.writeText(text).catch(() => {});
  });

  // ── Config Buttons ──
  document.getElementById('btnSaveConfig').addEventListener('click', async function () {
    const cfg = {
      embedder: {
        model_name: document.getElementById('cfgModelName').value || 'all-MiniLM-L6-v2',
      },
      consolidator: {
        max_memories: parseInt(document.getElementById('cfgMaxMemories').value) || 10000,
        prune_strategy: document.getElementById('cfgStrategy').value,
        similarity_threshold: parseFloat(document.getElementById('cfgMinImportance').value) || 0.85,
      },
      retriever: {
        hybrid_alpha: parseFloat(document.getElementById('cfgAlpha').value) || 0.5,
        default_k: 10,
        min_score: 0.0,
      },
      store: {
        capacity: parseInt(document.getElementById('cfgCapacity').value) || 10000,
      },
    };
    try {
      await putApi('/config', cfg);
      loadConfig();
      loadSidebar();
    } catch (e) {
      showError(e.message);
    }
  });

  document.getElementById('btnPrune').addEventListener('click', async function () {
    const strategy = document.getElementById('cfgStrategy').value;
    const maxItems = parseInt(document.getElementById('cfgMaxMemories').value) || 10000;
    try {
      const result = await postApi('/prune', { strategy, max_items: maxItems });
      alert(`Pruned ${result.removed} memories. ${result.remaining} remaining.`);
      loadSidebar();
    } catch (e) {
      showError(e.message);
    }
  });

  document.getElementById('btnExport').addEventListener('click', async function () {
    try {
      const data = await postApi('/backup');
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `neural-memory-backup-${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      showError(e.message);
    }
  });

  document.getElementById('btnImport').addEventListener('change', async function (e) {
    const file = e.target.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await fetch('/api/restore', { method: 'POST', body: formData });
      if (!res.ok) throw new Error((await res.json()).detail || 'Import failed');
      const data = await res.json();
      alert(`Imported ${data.imported} memories.`);
      navigateToBrowser();
      loadSidebar();
    } catch (e) {
      showError(e.message);
    }
    e.target.value = '';
  });

  // ── Topbar Buttons ──
  document.getElementById('btnSearch').addEventListener('click', function () {
    navigateToBrowser();
    document.getElementById('searchInput').focus();
  });

  document.getElementById('btnSettings').addEventListener('click', function () {
    showView('view-config');
    loadConfig();
  });

  document.getElementById('errorRetry').addEventListener('click', function () {
    const active = document.querySelector('.view.active');
    if (active) {
      const viewId = active.id;
      if (viewId === 'view-browser') navigateToBrowser();
      else if (viewId === 'view-detail' && currentMemoryId) navigateToDetail(currentMemoryId);
      else if (viewId === 'view-config') loadConfig();
    }
    hideError();
  });

  // ── Init ──
  async function init() {
    await loadSidebar();
    navigateToBrowser();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
