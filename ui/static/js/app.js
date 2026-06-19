// ── Neural Memory — Lino UI ──

const API = {
  memories: '/api/memories',
  graph: '/api/graph',
  search: '/api/search',
  stats: '/api/stats',
  profile: '/api/profile',
  brainstorm: '/api/brainstorm',
}

const PROFILE_UUID = '1b9d6bcd-bbfd-4b2d-9b5d-ab8dfbbd4bed'

const TAG_COLORS = {
  'type:profile': '#F59E0B',
  'type:preferences': '#38BDF8',
  'type:entity': '#FB7185',
  'session-summary': '#F59E0B',
  'decision': '#38BDF8',
  'fact': '#34D399',
  'compressed': '#F59E0B',
  'vault': '#EC4899',
  'raw-session': '#6366f1',
  'project:': '#F59E0B',
  'person:': '#FB7185',
  default: '#55557a',
}

const EDGE_COLORS = {
  'related': '#1a1a2e',
  'works_at': '#F59E0B',
  'employed_by': '#F59E0B',
  'part_of': '#38BDF8',
  'member_of': '#38BDF8',
  'founded': '#34D399',
  'created': '#34D399',
  'related_to': '#55557a',
  'connected_to': '#55557a',
  'mentions': '#6366f1',
  'self': '#FB7185',
  default: '#1a1a2e',
}

function edgeColor(type) {
  return EDGE_COLORS[type] || EDGE_COLORS.default
}

let graphData = { nodes: [], edges: [] }
let allNodes = []
let allEdges = []
let selectedNodeId = null
let localGraphMode = false
let localNodeIds = new Set()
let simulation = null
let svg, gLink, gNode, container
let currentView = 'graph'
let memories = []
let currentPage = 0
let sortMode = 'newest'
const PAGE_SIZE = 50
let statDimension = null

function tagColor(tags) {
  if (!tags || tags.length === 0) return TAG_COLORS.default
  for (const t of tags) {
    if (TAG_COLORS[t]) return TAG_COLORS[t]
  }
  for (const t of tags) {
    for (const [key, color] of Object.entries(TAG_COLORS)) {
      if (t.startsWith(key) && key.endsWith(':')) return color
    }
  }
  return TAG_COLORS.default
}

function importanceSize(d) {
  if (d.tags && d.tags.includes('type:profile')) return 26
  if (d.tags && d.tags.includes('type:preferences')) return 16
  if (d.tags && d.tags.includes('type:entity')) return 14
  return 5 + (d.importance || 0.5) * 14
}

function isProfileNode(d) { return d.tags && d.tags.includes('type:profile') }
function isEntityNode(d) { return d.tags && d.tags.includes('type:entity') }

async function apiGet(url) {
  const r = await fetch(url)
  if (!r.ok) throw new Error(`GET ${url} → ${r.status}`)
  return r.json()
}
async function apiPost(url, body) {
  const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
  if (!r.ok) throw new Error(`POST ${url} → ${r.status}`)
  return r.json()
}
async function apiPut(url, body) {
  const r = await fetch(url, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
  if (!r.ok) throw new Error(`PUT ${url} → ${r.status}`)
  return r.json()
}
async function apiDelete(url) {
  const r = await fetch(url, { method: 'DELETE' })
  if (!r.ok) throw new Error(`DELETE ${url} → ${r.status}`)
  return r.json()
}

function initGraph() {
  svg = d3.select('#graph-svg')
  svg.selectAll('*').remove()
  container = document.getElementById('graph-container')
  const width = container.clientWidth, height = container.clientHeight
  svg.attr('width', width).attr('height', height)

  const zoom = d3.zoom().scaleExtent([0.1, 8]).on('zoom', (event) => g.attr('transform', event.transform))
  const g = svg.append('g')
  svg.call(zoom)
  svg.on('dblclick.zoom', null)
  svg.on('click', (event) => { if (event.target === svg.node()) deselectNode() })
  gLink = g.append('g')
  gNode = g.append('g')

  window.addEventListener('resize', () => {
    const w = container.clientWidth, h = container.clientHeight
    svg.attr('width', w).attr('height', h)
  })
}

function buildGraph() {
  const limit = parseInt(document.getElementById('nodeLimitSlider').value)
  const minImp = parseFloat(document.getElementById('minImpSlider').value)
  let nodes = [...allNodes]
  if (minImp > 0) nodes = nodes.filter(n => n.importance >= minImp)
  const profileNode = nodes.find(n => isProfileNode(n))
  const alwaysKeep = new Set()
  if (profileNode) alwaysKeep.add(profileNode.id)
  nodes.sort((a, b) => b.importance - a.importance)
  nodes = nodes.slice(0, limit)
  alwaysKeep.forEach(id => { if (!nodes.find(n => n.id === id) && allNodes.find(n => n.id === id)) nodes.push(allNodes.find(n => n.id === id)) })
  const nodeIds = new Set(nodes.map(n => n.id))
  const edges = allEdges.filter(e => nodeIds.has(e.source) && nodeIds.has(e.target))
  const linkedNodeIds = new Set()
  edges.forEach(e => { linkedNodeIds.add(e.source); linkedNodeIds.add(e.target) })
  linkedNodeIds.forEach(id => nodeIds.add(id))
  graphData = { nodes, edges }
  if (selectedNodeId && !nodeIds.has(selectedNodeId)) selectedNodeId = null
  updateGraph()
  updateStats()
  renderLegend()
  renderSidebar()
}

function nodeShape(d) {
  if (isProfileNode(d)) return 'diamond'
  return 'circle'
}

function renderNode(selection) {
  const circles = selection.filter(d => nodeShape(d) === 'circle')
  circles.append('circle')
    .attr('r', d => importanceSize(d))
    .attr('fill', d => tagColor(d.tags))
    .attr('stroke', d => {
      if (selectedNodeId === d.id) return tagColor(d.tags)
      return 'transparent'
    })
    .attr('stroke-width', selectedNodeId ? 2 : 0)
    .attr('stroke-opacity', 0.5)

  const diamonds = selection.filter(d => nodeShape(d) === 'diamond')
  diamonds.append('rect')
    .attr('x', d => -importanceSize(d) * 0.7)
    .attr('y', d => -importanceSize(d) * 0.7)
    .attr('width', d => importanceSize(d) * 1.4)
    .attr('height', d => importanceSize(d) * 1.4)
    .attr('transform', d => 'rotate(45)')
    .attr('fill', 'url(#profile-grad)')
    .attr('stroke', '#F59E0B')
    .attr('stroke-width', 2)
    .attr('stroke-opacity', 0.6)
    .style('filter', 'drop-shadow(0 0 8px rgba(245, 158, 11, 0.4))')
}

function updateGraph() {
  if (!container) container = document.getElementById('graph-container')
  const width = container.clientWidth, height = container.clientHeight
  if (simulation) simulation.stop()

  const links = graphData.edges.map(e => ({ source: e.source, target: e.target, type: e.type }))
  const profileNode = graphData.nodes.find(n => isProfileNode(n))

  simulation = d3.forceSimulation(graphData.nodes)
    .force('link', d3.forceLink(links).id(d => d.id).distance(90).strength(0.25))
    .force('charge', d3.forceManyBody().strength(-150))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('collision', d3.forceCollide().radius(d => importanceSize(d) + 6))
    .alphaDecay(0.015)

  const defs = svg.select('defs')
  if (defs.empty()) svg.insert('defs', ':first-child')
  const grad = svg.select('defs').selectAll('#profile-grad').data([0])
  grad.join('enter').append('radialGradient').attr('id', 'profile-grad')
    .attr('cx', '30%').attr('cy', '30%').attr('r', '70%')
    .each(function() {
      d3.select(this).append('stop').attr('offset', '0%').attr('stop-color', '#FCD34D').attr('stop-opacity', 0.8)
      d3.select(this).append('stop').attr('offset', '100%').attr('stop-color', '#F59E0B').attr('stop-opacity', 0.3)
    })

  const linkGroup = gLink.selectAll('g.link').data(links).join('g').attr('class', 'link')
  const link = linkGroup.append('line')
    .attr('stroke', d => {
      const etype = d.type || 'related'
      if (profileNode) {
        const sid = d.source.id || d.source, tid = d.target.id || d.target
        if (sid === profileNode.id || tid === profileNode.id) return 'rgba(245, 158, 11, 0.25)'
      }
      if (etype !== 'related') return edgeColor(etype)
      return '#1a1a2e'
    })
    .attr('stroke-width', d => (d.type && d.type !== 'related') ? 1.5 : 1)
    .attr('stroke-opacity', d => (d.type && d.type !== 'related') ? 0.5 : 0.3)
    .attr('data-type', d => d.type || 'related')

  const linkLabel = linkGroup.append('text')
    .text(d => {
      const t = d.type || 'related'
      return t !== 'related' ? t.replace(/_/g, ' ') : ''
    })
    .attr('font-size', 9)
    .attr('fill', '#E8E8F0')
    .attr('font-family', "'JetBrains Mono', monospace")
    .attr('dy', -5)
    .attr('text-anchor', 'middle')
    .attr('opacity', 0.9)
    .attr('paint-order', 'stroke')
    .attr('stroke', '#08080E')
    .attr('stroke-width', 5)
    .attr('font-weight', 500)
    .style('pointer-events', 'none')

  const node = gNode.selectAll('g.node').data(graphData.nodes).join('g')
    .attr('class', 'node')
    .attr('transform', d => `translate(${d.x || width/2},${d.y || height/2})`)
    .style('cursor', 'pointer')

  node.each(function(d) {
    const el = d3.select(this)
    el.selectAll('*').remove()
    if (nodeShape(d) === 'diamond') {
      const s = importanceSize(d)
      el.append('rect')
        .attr('x', -s * 0.7).attr('y', -s * 0.7)
        .attr('width', s * 1.4).attr('height', s * 1.4)
        .attr('transform', 'rotate(45)')
        .attr('fill', 'url(#profile-grad)')
        .attr('stroke', '#F59E0B').attr('stroke-width', 2).attr('stroke-opacity', d.id === selectedNodeId ? 0.9 : 0.4)
        .style('filter', 'drop-shadow(0 0 8px rgba(245, 158, 11, 0.3))')
    } else {
      el.append('circle')
        .attr('r', importanceSize(d))
        .attr('fill', d.tags && d.tags.includes('type:preferences') ? '#38BDF8' : tagColor(d.tags))
        .attr('stroke', selectedNodeId === d.id ? tagColor(d.tags) : 'transparent')
        .attr('stroke-width', selectedNodeId ? 2 : 0)
    }
  })

  const label = gNode.selectAll('text').data(graphData.nodes).join('text')
    .text(d => d.text.substring(0, 18))
    .attr('font-size', d => isProfileNode(d) ? 9 : 8)
    .attr('dx', d => importanceSize(d) + 5)
    .attr('dy', 3)
    .attr('fill', '#55557a')
    .style('pointer-events', 'none')
    .style('font-family', "'JetBrains Mono', monospace")

  node.on('click', (event, d) => { event.stopPropagation(); selectNode(d.id) })
  node.on('mouseenter', (event, d) => {
    if (selectedNodeId === d.id) return
    showTooltip(event, d)
    highlightConnections(d.id)
  })
  node.on('mouseleave', () => { hideTooltip(); clearHighlight() })

  linkGroup.on('mouseenter', (event, d) => {
    d3.select(event.currentTarget).select('line')
      .attr('stroke', '#F59E0B').attr('stroke-width', 1.5).attr('stroke-opacity', 0.5)
    d3.select(event.currentTarget).select('text')
      .attr('fill', '#F59E0B').attr('opacity', 1)
  })
  linkGroup.on('mouseleave', (event, d) => {
    const etype = d.type || 'related'
    d3.select(event.currentTarget).select('line')
      .attr('stroke', edgeColor(etype)).attr('stroke-width', 1).attr('stroke-opacity', 0.3)
    d3.select(event.currentTarget).select('text')
      .attr('fill', edgeColor(etype)).attr('opacity', 0.6)
  })

  simulation.on('tick', () => {
    if (profileNode) { profileNode.fx = width / 2; profileNode.fy = height / 2 }
    linkGroup.selectAll('line')
      .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x).attr('y2', d => d.target.y)
    linkGroup.selectAll('text')
      .attr('x', d => (d.source.x + d.target.x) / 2)
      .attr('y', d => (d.source.y + d.target.y) / 2)
    node.attr('transform', d => `translate(${d.x},${d.y})`)
    label.attr('x', d => d.x).attr('y', d => d.y)
  })
}

function selectNode(id) {
  if (!id) return
  selectedNodeId = id
  const connected = new Set([id])
  graphData.edges.forEach(e => {
    if ((e.source.id || e.source) === id) connected.add(e.target.id || e.target)
    if ((e.target.id || e.target) === id) connected.add(e.source.id || e.source)
  })
  gNode.selectAll('g.node').transition().duration(300)
    .attr('opacity', d => connected.has(d.id) ? 1 : 0.12)
  gNode.selectAll('rect, circle').transition().duration(300)
    .attr('opacity', d => connected.has(d.id) ? 1 : 0.12)
  gLink.selectAll('g.link').each(function(d) {
    const el = d3.select(this)
    const sid = d.source.id || d.source, tid = d.target.id || d.target
    const isConnected = (sid === id || tid === id)
    el.select('line').transition().duration(300)
      .attr('stroke-opacity', isConnected ? 0.6 : 0.04)
      .attr('stroke-width', isConnected ? 2 : 1)
    el.select('text').transition().duration(300)
      .attr('opacity', isConnected ? 1 : 0.02)
  })
  showDetail(id)
}

function deselectNode() {
  selectedNodeId = null
  gNode.selectAll('g.node').transition().duration(300).attr('opacity', 1)
  gNode.selectAll('rect, circle').transition().duration(300).attr('opacity', 1)
  gLink.selectAll('g.link').each(function(d) {
    d3.select(this).select('line').transition().duration(300).attr('stroke-opacity', 0.3).attr('stroke-width', 1)
    d3.select(this).select('text').transition().duration(300).attr('opacity', 0.6)
  })
  hideDetail()
}

function highlightConnections(id) {
  const connected = new Set([id])
  graphData.edges.forEach(e => {
    if ((e.source.id || e.source) === id) connected.add(e.target.id || e.target)
    if ((e.target.id || e.target) === id) connected.add(e.source.id || e.source)
  })
  gNode.selectAll('g.node').attr('opacity', d => connected.has(d.id) ? 1 : 0.1)
  gLink.selectAll('g.link').each(function(d) {
    const el = d3.select(this)
    const sid = d.source.id || d.source, tid = d.target.id || d.target
    const isConnected = (sid === id || tid === id)
    el.select('line').attr('stroke-opacity', isConnected ? 0.5 : 0.03)
    el.select('text').attr('opacity', isConnected ? 1 : 0.02)
  })
}

function clearHighlight() {
  if (selectedNodeId) { selectNode(selectedNodeId); return }
  gNode.selectAll('g.node').attr('opacity', 1)
  gLink.selectAll('g.link').each(function() {
    d3.select(this).select('line').attr('stroke-opacity', 0.3)
    d3.select(this).select('text').attr('opacity', 0.6)
  })
}

function showTooltip(event, d) {
  const tt = document.getElementById('graph-tooltip')
  tt.classList.remove('hidden')
  let label = escHtml(d.text)
  if (isProfileNode(d)) label = '<span style="color:#F59E0B">★ Profile</span> — ' + label
  tt.innerHTML = `
    <div class="tt-title">${label}</div>
    <div class="tt-meta">${d.importance.toFixed(2)} strength</div>
    <div class="tt-tags">${(d.tags || []).map(t => `<span class="tag-pill" style="font-size:9px">${escHtml(t)}</span>`).join('')}</div>`
  const rect = container.getBoundingClientRect()
  const cx = event.clientX || event.sourceEvent?.clientX || 0
  const cy = event.clientY || event.sourceEvent?.clientY || 0
  tt.style.left = (cx - rect.left + 14) + 'px'
  tt.style.top = (cy - rect.top - 12) + 'px'
}

function hideTooltip() { document.getElementById('graph-tooltip').classList.add('hidden') }

function renderLegend() {
  const tagCounts = {}
  allNodes.forEach(n => (n.tags || []).forEach(t => { tagCounts[t] = (tagCounts[t] || 0) + 1 }))
  const top = Object.entries(tagCounts).sort((a, b) => b[1] - a[1]).slice(0, 4)
  const legend = document.getElementById('graphLegend')
  if (top.length === 0) { legend.innerHTML = ''; return }

  // Collect unique edge types present in the current graph
  const edgeTypes = new Set()
  allEdges.forEach(e => {
    const t = e.type || 'related'
    if (t !== 'related') edgeTypes.add(t)
  })

  let html = ''
  // Node tag legend
  top.forEach(([tag]) => {
    const color = tagColor([tag])
    html += `<div class="legend-item"><span class="legend-dot" style="background:${color}"></span>${escHtml(tag)}</div>`
  })
  // Edge type legend
  if (edgeTypes.size > 0) {
    html += '<div class="legend-sep" style="border-top:1px solid #2a2a3e;margin:4px 0;padding-top:4px"></div>'
    edgeTypes.forEach(t => {
      html += `<div class="legend-item"><span class="legend-dot" style="background:${edgeColor(t)};border-radius:2px;height:3px;width:12px"></span>${escHtml(t)}</div>`
    })
  }
  legend.innerHTML = html
}

async function showDetail(id) {
  const panel = document.getElementById('detailPanel')
  panel.classList.remove('hidden')
  document.getElementById('detailContent').classList.add('hidden')
  document.getElementById('detailPanelTitle').textContent = 'Loading...'
  document.getElementById('profileSection').classList.add('hidden')
  document.getElementById('goalsSection').classList.add('hidden')
  document.getElementById('backlinksSection').classList.add('hidden')
  try {
    const mem = await apiGet(`${API.memories}/${id}`)
    const meta = mem.metadata || {}
    const text = meta.text || mem.text || ''
    const tags = meta.tags || []
    document.getElementById('detailPanelTitle').textContent = text.substring(0, 36) || 'Neuron'
    document.getElementById('detailText').value = text
    document.getElementById('detailSource').value = meta.source || ''
    document.getElementById('detailImportance').value = meta.importance_score || 0.5
    document.getElementById('detailImportanceVal').textContent = (meta.importance_score || 0.5).toFixed(2)
    document.getElementById('detailTags').value = tags.join(', ')
    document.getElementById('detailId').textContent = id

    // Related memories (connections)
    const related = meta.related_memories || []
    const relatedTypes = meta.related_types || {}
    const rc = document.getElementById('detailRelated')
    if (related.length > 0) {
      rc.innerHTML = related.map(rid => {
        const cid = (typeof rid === 'string' ? rid.replace(/[\[\]]/g, '') : rid)
        const node = allNodes.find(n => n.id === cid)
        const label = node ? node.text.substring(0, 40) : cid.substring(0, 16)
        const color = node ? tagColor(node.tags) : '#55557a'
        const etype = relatedTypes[cid] || 'related'
        const typeLabel = etype !== 'related' ? `<span class="edge-type" style="font-size:8px;color:${edgeColor(etype)};margin-left:4px">${escHtml(etype)}</span>` : ''
        return `<div class="related-item" data-id="${cid}"><span class="related-dot" style="background:${color}"></span>${escHtml(label)}${typeLabel}</div>`
      }).join('')
      rc.querySelectorAll('.related-item').forEach(el => el.addEventListener('click', () => selectNode(el.dataset.id)))
    } else {
      rc.innerHTML = '<p class="text-secondary">No connections yet</p>'
    }

    // Backlinks (memories that link to this one)
    try {
      const blData = await apiGet(`${API.memories}/${id}/backlinks`)
      const bl = blData.backlinks || []
      const blEl = document.getElementById('backlinksList')
      if (bl.length > 0) {
        blEl.innerHTML = bl.map(b => {
          return `<div class="related-item" data-id="${b.id}"><span class="related-dot" style="background:${edgeColor(b.type)}"></span>${escHtml(b.text.substring(0, 40))} <span class="edge-type" style="font-size:8px;color:${edgeColor(b.type)}">${escHtml(b.type)}</span></div>`
        }).join('')
        blEl.querySelectorAll('.related-item').forEach(el => el.addEventListener('click', () => selectNode(el.dataset.id)))
        document.getElementById('backlinksSection').classList.remove('hidden')
      } else {
        blEl.innerHTML = '<p class="text-secondary">No backlinks</p>'
      }
    } catch(e) {
      document.getElementById('backlinksSection').classList.add('hidden')
    }

    const profileSec = document.getElementById('profileSection')
    const goalsSec = document.getElementById('goalsSection')
    if (tags.includes('type:profile')) {
      const pd = meta.profile_data || {}
      const prefsEl = document.getElementById('detailPrefs')
      const prefs = []
      if (pd.preferences) {
        if (pd.preferences.answer_style) prefs.push(`Answer style: ${pd.preferences.answer_style}`)
        if (pd.preferences.code_examples) prefs.push(`Code: ${pd.preferences.code_examples}`)
      }
      // Add learned preference indicators
      const learned = pd.learned_preferences || {}
      if (learned.code_examples && learned.code_examples.source === 'explicit') {
        prefs.push('⭐ Code: learned from "always/never" keyword')
      }
      if (learned.skills_for_tasks) {
        for (const cat of Object.keys(learned.skills_for_tasks)) {
          prefs.push(`⚡ ${cat}: auto-invoke available`)
        }
      }
      if (pd.learning_goals && pd.learning_goals.length > 0) {
        const goalsEl = document.getElementById('detailGoals')
        goalsEl.innerHTML = pd.learning_goals.map(g => `<div class="related-item">${escHtml(g)}</div>`).join('')
        goalsSec.classList.remove('hidden')
      }
      prefsEl.innerHTML = prefs.map(p => `<div class="related-item">${escHtml(p)}</div>`).join('')
      profileSec.classList.remove('hidden')
    }

    if (tags.some(t => t.startsWith('type:entity') && t.includes('entity_type:project'))) {
      const pg = meta.project_goals || []
      if (pg.length > 0) {
        const goalsEl = document.getElementById('detailGoals')
        goalsEl.innerHTML = pg.map(g => `<div class="related-item">${escHtml(g)}</div>`).join('')
        goalsSec.classList.remove('hidden')
      }
    }

    document.getElementById('detailContent').classList.remove('hidden')
  } catch(e) {
    document.getElementById('detailPanelTitle').textContent = 'Error'
  }
}

function hideDetail() { document.getElementById('detailPanel').classList.add('hidden'); localGraphMode = false }

function showLocalGraph(id) {
  const connected = new Set([id])
  allEdges.forEach(e => {
    if (e.source === id || e.source.id === id) connected.add(e.target.id || e.target)
    if (e.target === id || e.target.id === id) connected.add(e.source.id || e.source)
  })
  graphData = { nodes: allNodes.filter(n => connected.has(n.id)), edges: allEdges.filter(e => connected.has(e.source.id || e.source) && connected.has(e.target.id || e.target)) }
  localGraphMode = true; localNodeIds = connected; selectedNodeId = id
  updateGraph()
  setTimeout(() => selectNode(id), 150)
}

let searchTimeout = null
document.getElementById('searchInput').addEventListener('input', () => { clearTimeout(searchTimeout); searchTimeout = setTimeout(doSearch, 300) })
document.getElementById('btnClearSearch').addEventListener('click', () => { document.getElementById('searchInput').value = ''; clearSearchHighlight() })

async function doSearch() {
  const q = document.getElementById('searchInput').value.trim()
  if (!q) { clearSearchHighlight(); return }
  try {
    const data = await apiPost(API.search, { query: q, k: 50, threshold: 0 })
    highlightNodes(data.results.map(r => r.id))
  } catch(e) {}
}

function highlightNodes(ids) {
  const s = new Set(ids)
  gNode.selectAll('g.node').attr('opacity', d => s.has(d.id) ? 1 : 0.08)
  gLink.selectAll('g.link').each(function() {
    d3.select(this).select('line').attr('stroke-opacity', 0.03)
    d3.select(this).select('text').attr('opacity', 0.02)
  })
}

function clearSearchHighlight() {
  if (selectedNodeId) { selectNode(selectedNodeId); return }
  if (localGraphMode && selectedNodeId) { selectNode(selectedNodeId); return }
  gNode.selectAll('g.node').attr('opacity', 1)
  gLink.selectAll('g.link').each(function() {
    d3.select(this).select('line').attr('stroke-opacity', 0.3)
    d3.select(this).select('text').attr('opacity', 0.6)
  })
}

function renderSidebar() {
  const tagsMap = {}
  allNodes.forEach(n => (n.tags || []).forEach(t => { tagsMap[t] = (tagsMap[t] || 0) + 1 }))
  const el = document.getElementById('tagsList')
  const entries = Object.entries(tagsMap).sort((a, b) => b[1] - a[1])
  document.getElementById('tagCount').textContent = entries.length
  if (entries.length === 0) { el.innerHTML = '<p class="text-secondary" style="padding:4px;font-size:11px">No clusters</p>'; return }
  el.innerHTML = entries.map(([tag, count]) => {
    const color = tagColor([tag])
    return `<div class="tag-pill" style="margin:2px" data-tag="${escHtml(tag)}">
      <span style="width:6px;height:6px;border-radius:50%;background:${color};display:inline-block;flex-shrink:0"></span>
      ${escHtml(tag)}<span class="tag-count">${count}</span></div>`
  }).join('')
  el.querySelectorAll('.tag-pill').forEach(el2 => el2.addEventListener('click', () => filterByTag(el2.dataset.tag)))
}

function filterByTag(tag) { highlightNodes(allNodes.filter(n => (n.tags || []).includes(tag)).map(n => n.id)) }

function updateStats() {
  document.getElementById('statTotal').textContent = allNodes.length
  document.getElementById('statEdges').textContent = allEdges.length
  const dimEl = document.getElementById('statDim')
  if (dimEl && (!dimEl.textContent.trim() || dimEl.textContent.trim() === '—')) {
    dimEl.textContent = statDimension || '—'
  }
}

async function loadBrowser() {
  const container = document.getElementById('tableContainer'), loading = document.getElementById('loadingState'), empty = document.getElementById('emptyState')
  empty.classList.add('hidden'); container.classList.add('hidden'); loading.classList.remove('hidden')
  try {
    const sortParam = sortMode === 'importance' ? `sort=importance&reverse=false` : `sort=newest&reverse=true`
    const data = await apiGet(`${API.memories}?limit=${PAGE_SIZE}&offset=${currentPage * PAGE_SIZE}&${sortParam}`)
    memories = data.memories || []; const total = data.total || 0
    loading.classList.add('hidden')
    if (memories.length === 0) { empty.classList.remove('hidden'); return }
    container.classList.remove('hidden')
    document.getElementById('memoriesBody').innerHTML = memories.map(m => {
      const meta = m.metadata || {}, text = meta.text || m.text || ''
      const ts = meta.timestamp ? new Date(meta.timestamp * 1000).toLocaleString() : '—'
      const tags = (meta.tags || []).map(t => `<span class="tag-pill" style="font-size:8px;padding:1px 6px">${escHtml(t)}</span>`).join(' ')
      return `<tr><td class="col-text" data-id="${m.id}">${escHtml(text.substring(0, 100))}</td>
        <td class="col-source">${escHtml(meta.source || '—')}</td>
        <td class="col-importance">${(meta.importance_score || 0.5).toFixed(2)}</td>
        <td class="col-time">${ts}</td>
        <td class="col-tags">${tags}</td>
        <td class="col-actions"><button class="btn btn-sm btn-ghost btn-icon" data-id="${m.id}" title="View in graph"><svg width="12" height="12" viewBox="0 0 12 12" fill="none"><circle cx="6" cy="3" r="1.5" stroke="currentColor" stroke-width="1"/><circle cx="3" cy="9" r="1.5" stroke="currentColor" stroke-width="1"/><circle cx="9" cy="9" r="1.5" stroke="currentColor" stroke-width="1"/></svg></button></td></tr>`
    }).join('')
    document.querySelectorAll('.col-text, .col-actions button').forEach(el => el.addEventListener('click', () => { switchView('graph'); selectNode(el.dataset.id) }))
    const tp = Math.ceil(total / PAGE_SIZE)
    document.getElementById('pageInfo').textContent = `Page ${currentPage + 1} of ${tp}`
    document.getElementById('prevPage').disabled = currentPage === 0
    document.getElementById('nextPage').disabled = currentPage >= tp - 1
  } catch(e) {
    loading.classList.add('hidden')
    document.getElementById('errorMessage').textContent = `Failed: ${e.message}`
    document.getElementById('errorBanner').classList.remove('hidden')
  }
}

function switchView(view) {
  currentView = view
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'))
  document.querySelectorAll('.topbar-btn[data-view]').forEach(b => b.classList.remove('active'))
  if (view === 'graph') {
    document.getElementById('view-graph').classList.add('active')
    document.getElementById('btnGraphView').classList.add('active')
    setTimeout(buildGraph, 80)
  } else if (view === 'brainstorm') {
    document.getElementById('view-brainstorm').classList.add('active')
    document.getElementById('btnBrainstormView').classList.add('active')
    loadBrainstormSessions()
  } else {
    document.getElementById('view-browser').classList.add('active')
    document.getElementById('btnBrowserView').classList.add('active')
    loadBrowser()
  }
}

// Profile modal
function openProfileModal() {
  document.getElementById('profileModal').classList.remove('hidden')
  loadProfileIntoModal()
}
function closeProfileModal() { document.getElementById('profileModal').classList.add('hidden') }
function formatLearnedPrefs(learned) {
  if (!learned || Object.keys(learned).length === 0) return 'No learned preferences yet.'
  const lines = []
  for (const [key, info] of Object.entries(learned)) {
    if (key === 'skills_for_tasks') {
      for (const [cat, sinfo] of Object.entries(info)) {
        const skills = (sinfo.skills || []).join(', ')
        lines.push(`${cat} → ${skills} (${sinfo.count}x, ${(sinfo.confidence*100).toFixed(0)}% confidence)`)
      }
    } else {
      const val = info.value || '?'
      const cnt = info.count || '?'
      const conf = info.confidence ? `, ${(info.confidence*100).toFixed(0)}% confidence` : ''
      const src = info.source ? ` [${info.source}]` : ''
      lines.push(`${key}: ${val} (${cnt} observations${conf}${src})`)
    }
  }
  return lines.join('\n')
}
function loadProfileIntoModal() {
  apiGet(API.profile).then(prof => {
    const pd = prof.profile_data || {}
    document.getElementById('profileName').value = pd.name || ''
    document.getElementById('profileRole').value = pd.role || ''
    document.getElementById('profileBio').value = pd.bio || ''
    const goals = pd.learning_goals || []
    document.getElementById('profileGoals').value = goals.join(', ')
    const prefs = pd.preferences || {}
    document.getElementById('profileAnswerStyle').value = prefs.answer_style || 'concise'
    document.getElementById('profileSkills').value = prefs.skills_for_tasks ? JSON.stringify(prefs.skills_for_tasks, null, 2) : ''
    document.getElementById('profileExtraPrefs').value = prefs.extra ? JSON.stringify(prefs.extra, null, 2) : ''
    // Load learned preferences
    apiGet('/api/preferences/learned').then(r => {
      const learned = r.learned_preferences || {}
      const display = document.getElementById('learnedPrefsDisplay')
      const content = document.getElementById('learnedPrefsContent')
      if (Object.keys(learned).length > 0) {
        display.style.display = ''
        content.textContent = formatLearnedPrefs(learned)
      } else {
        display.style.display = 'none'
      }
    }).catch(() => {})
  }).catch(() => {})
}
async function saveProfile() {
  const name = document.getElementById('profileName').value.trim()
  const role = document.getElementById('profileRole').value.trim()
  const bio = document.getElementById('profileBio').value.trim()
  const goalsStr = document.getElementById('profileGoals').value.trim()
  const learningGoals = goalsStr ? goalsStr.split(',').map(s => s.trim()).filter(Boolean) : []
  const answerStyle = document.getElementById('profileAnswerStyle').value
  let skillsForTasks = {}
  let extraPrefs = {}
  try { skillsForTasks = JSON.parse(document.getElementById('profileSkills').value.trim() || '{}') } catch(e) {}
  try { extraPrefs = JSON.parse(document.getElementById('profileExtraPrefs').value.trim() || '{}') } catch(e) {}
  const body = {
    name,
    role,
    bio,
    learning_goals: learningGoals,
    preferences: {
      answer_style: answerStyle,
      skills_for_tasks: skillsForTasks,
      extra: extraPrefs,
    },
  }
  try {
    await apiPut(API.profile, body)
    closeProfileModal()
    loadAll()
  } catch(e) { alert('Save failed: ' + e.message) }
}

async function loadAll() {
  try {
    const [graphRes, statsRes] = await Promise.all([apiGet(API.graph), apiGet(API.stats)])
    allNodes = graphRes.nodes || []; allEdges = graphRes.edges || []
    graphData = { nodes: allNodes, edges: allEdges }
    document.getElementById('headerCount').textContent = allNodes.length
    const stat = statsRes
    statDimension = stat.dimension || null
    document.getElementById('statDim').textContent = statDimension || '—'
    document.getElementById('statCapacity').textContent = stat.capacity || '—'
    updateStats()
    if (currentView === 'graph') buildGraph()
    else if (currentView === 'browser') loadBrowser()
    renderLegend()
    renderSidebar()
  } catch(e) {}
}

// ── Brainstorm ──

let _brainstormSelectedSession = null
let _brainstormSvg = null, _brainstormG = null, _brainstormGLink = null, _brainstormGNode = null
let _brainstormSimulation = null
let _brainstormGraphContainer = null
let _brainstormSelectedNodeId = null
let _brainstormGraphData = null
let _brainstormSessionsCache = []

const BS_EDGE_COLORS = {
  'generated': '#55557a',
  'builds_upon': '#38BDF8',
  'synthesis': '#34D399',
  'related': '#1a1a2e',
}
const BS_EDGE_LABELS = {
  'generated': '',
  'builds_upon': 'builds on',
  'synthesis': 'synthesis',
  'related': 'related',
}

function bsEdgeColor(type) { return BS_EDGE_COLORS[type] || '#55557a' }

function isTopicNode(d) { return d.level === 0 && !d.parent_id }

function bsNodeSize(d) {
  if (isTopicNode(d)) return 22
  const imp = d.importance || 0.5
  return 7 + imp * 14
}

function bsNodeColor(d) {
  if (isTopicNode(d)) return '#F59E0B'
  if (d.synthesis) return '#34D399'
  if (d.phase === 1) return '#F59E0B'
  return '#38BDF8'
}

function initBrainstormGraph() {
  _brainstormSvg = d3.select('#brainstorm-svg')
  _brainstormSvg.selectAll('*').remove()
  _brainstormGraphContainer = document.getElementById('brainstorm-graph-container')
  const width = _brainstormGraphContainer.clientWidth, height = _brainstormGraphContainer.clientHeight
  _brainstormSvg.attr('width', width).attr('height', height)

  const zoom = d3.zoom().scaleExtent([0.1, 8]).on('zoom', (event) => _brainstormG.attr('transform', event.transform))
  _brainstormG = _brainstormSvg.append('g')
  _brainstormSvg.call(zoom)
  _brainstormSvg.on('dblclick.zoom', null)
  _brainstormSvg.on('click', (event) => {
    if (event.target === _brainstormSvg.node()) deselectBrainstormNode()
  })
  _brainstormGLink = _brainstormG.append('g')
  _brainstormGNode = _brainstormG.append('g')

  window.addEventListener('resize', () => {
    const w = _brainstormGraphContainer.clientWidth, h = _brainstormGraphContainer.clientHeight
    if (w > 0 && h > 0) _brainstormSvg.attr('width', w).attr('height', h)
  })
}

function renderBrainstormGraph(session) {
  if (!_brainstormSvg) initBrainstormGraph()
  _brainstormSvg.selectAll('*').remove()
  _brainstormG = _brainstormSvg.append('g')
  _brainstormGLink = _brainstormG.append('g')
  _brainstormGNode = _brainstormG.append('g')

  const container = _brainstormGraphContainer
  const width = container.clientWidth, height = container.clientHeight
  _brainstormSvg.attr('width', width).attr('height', height)

  const zoom = d3.zoom().scaleExtent([0.1, 8]).on('zoom', (event) => _brainstormG.attr('transform', event.transform))
  _brainstormSvg.call(zoom)
  _brainstormSvg.on('dblclick.zoom', null)
  _brainstormSvg.on('click', (event) => {
    if (event.target === _brainstormSvg.node()) deselectBrainstormNode()
  })

  const nodes = (session.nodes || []).map(n => ({ ...n }))
  const rawEdges = session.edges || []
  if (nodes.length === 0) return

  const topicNode = nodes.find(n => isTopicNode(n))
  _brainstormGraphData = { nodes, edges: rawEdges }

  if (_brainstormSimulation) _brainstormSimulation.stop()

  const links = rawEdges.map(e => ({ source: e.source, target: e.target, type: e.type, reasoning: e.reasoning || '' }))

  _brainstormSimulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d => d.id).distance(100).strength(0.2))
    .force('charge', d3.forceManyBody().strength(-200))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('collision', d3.forceCollide().radius(d => bsNodeSize(d) + 8))
    .alphaDecay(0.02)

  // Edges
  const linkGroup = _brainstormGLink.selectAll('g.bs-link').data(links).join('g').attr('class', 'bs-link')
  linkGroup.append('line')
    .attr('stroke', d => {
      const etype = d.type || 'generated'
      if (topicNode) {
        const sid = d.source.id || d.source, tid = d.target.id || d.target
        if (sid === topicNode.id || tid === topicNode.id) return 'rgba(245, 158, 11, 0.2)'
      }
      if (etype !== 'generated') return bsEdgeColor(etype)
      return '#1a1a2e'
    })
    .attr('stroke-width', d => (d.type && d.type !== 'generated' && d.type !== 'related') ? 1.5 : 0.8)
    .attr('stroke-opacity', d => (d.type && d.type !== 'generated' && d.type !== 'related') ? 0.5 : 0.25)
    .attr('stroke-dasharray', d => d.type === 'related' ? '3,3' : 'none')
    .attr('data-type', d => d.type || 'generated')

  linkGroup.append('text')
    .text(d => BS_EDGE_LABELS[d.type] || '')
    .attr('font-size', 8)
    .attr('fill', d => bsEdgeColor(d.type))
    .attr('font-family', "'JetBrains Mono', monospace")
    .attr('dy', -6)
    .attr('text-anchor', 'middle')
    .attr('opacity', d => BS_EDGE_LABELS[d.type] ? 0.7 : 0)
    .attr('paint-order', 'stroke')
    .attr('stroke', '#08080E')
    .attr('stroke-width', 4)
    .style('pointer-events', 'none')

  // Nodes
  const node = _brainstormGNode.selectAll('g.bs-node').data(nodes).join('g')
    .attr('class', 'bs-node')
    .attr('transform', d => `translate(${d.x || width/2},${d.y || height/2})`)
    .style('cursor', 'pointer')

  node.each(function(d) {
    const el = d3.select(this)
    el.selectAll('*').remove()
    const size = bsNodeSize(d)
    if (isTopicNode(d)) {
      const s = size
      el.append('rect')
        .attr('x', -s * 0.7).attr('y', -s * 0.7)
        .attr('width', s * 1.4).attr('height', s * 1.4)
        .attr('transform', 'rotate(45)')
        .attr('fill', 'url(#bs-topic-grad)')
        .attr('stroke', '#F59E0B').attr('stroke-width', 2)
        .attr('stroke-opacity', d.id === _brainstormSelectedNodeId ? 0.9 : 0.4)
        .style('filter', 'drop-shadow(0 0 8px rgba(245, 158, 11, 0.3))')
    } else {
      el.append('circle')
        .attr('r', size)
        .attr('fill', d.synthesis ? 'rgba(52, 211, 153, 0.2)' : d.phase === 1 ? 'rgba(245, 158, 11, 0.15)' : 'rgba(56, 189, 248, 0.15)')
        .attr('stroke', _brainstormSelectedNodeId === d.id ? bsNodeColor(d) : bsNodeColor(d))
        .attr('stroke-width', _brainstormSelectedNodeId === d.id ? 2.5 : 1.5)
        .attr('stroke-opacity', _brainstormSelectedNodeId === d.id ? 0.9 : 0.3)
        .style('filter', d.synthesis ? 'drop-shadow(0 0 6px rgba(52, 211, 153, 0.25))' : 'none')
    }
  })

  // Labels
  const label = _brainstormGNode.selectAll('text.bs-label').data(nodes).join('text')
    .attr('class', 'bs-label')
    .text(d => {
      if (isTopicNode(d)) return 'Topic'
      const t = d.text || ''
      const clean = t.replace(/^Here are some/i, '').replace(/^Here's/i, '').replace(/^Okay,\s*/i, '').trim()
      return clean.length > 22 ? clean.slice(0, 22) + '…' : clean
    })
    .attr('font-size', d => isTopicNode(d) ? 9 : 8)
    .attr('dx', 0)
    .attr('dy', d => bsNodeSize(d) + 12)
    .attr('text-anchor', 'middle')
    .attr('fill', '#55557a')
    .style('pointer-events', 'none')
    .style('font-family', "'JetBrains Mono', monospace")

  // Phase/synthesis badge
  const badge = _brainstormGNode.selectAll('text.bs-badge').data(nodes).join('text')
    .attr('class', 'bs-badge')
    .text(d => {
      if (isTopicNode(d)) return '★'
      if (d.synthesis) return 'Σ'
      return d.phase === 1 ? 'P1' : 'P2'
    })
    .attr('font-size', d => isTopicNode(d) ? 10 : 7)
    .attr('dx', d => bsNodeSize(d) + 5)
    .attr('dy', 3)
    .attr('fill', d => bsNodeColor(d))
    .attr('font-weight', d => isTopicNode(d) ? 700 : 500)
    .attr('font-family', "'JetBrains Mono', monospace")
    .style('pointer-events', 'none')

  // Events
  node.on('click', (event, d) => {
    event.stopPropagation()
    selectBrainstormNode(d.id)
  })
  node.on('mouseenter', (event, d) => {
    if (_brainstormSelectedNodeId === d.id) return
    showBrainstormTooltip(event, d)
    highlightBrainstormConnections(d.id)
  })
  node.on('mouseleave', () => {
    hideBrainstormTooltip()
    clearBrainstormHighlight()
  })

  linkGroup.on('mouseenter', (event, d) => {
    d3.select(event.currentTarget).select('line')
      .attr('stroke', '#F59E0B').attr('stroke-width', 1.5).attr('stroke-opacity', 0.5)
    d3.select(event.currentTarget).select('text')
      .attr('fill', '#F59E0B').attr('opacity', 1)
  })
  linkGroup.on('mouseleave', (event, d) => {
    const etype = d.type || 'generated'
    d3.select(event.currentTarget).select('line')
      .attr('stroke', etype !== 'generated' && etype !== 'related' ? bsEdgeColor(etype) : '#1a1a2e')
      .attr('stroke-width', (etype && etype !== 'generated' && etype !== 'related') ? 1.5 : 0.8)
      .attr('stroke-opacity', (etype && etype !== 'generated' && etype !== 'related') ? 0.5 : 0.25)
    d3.select(event.currentTarget).select('text')
      .attr('fill', bsEdgeColor(etype)).attr('opacity', BS_EDGE_LABELS[etype] ? 0.7 : 0)
  })

  // Tick
  _brainstormSimulation.on('tick', () => {
    if (topicNode) { topicNode.fx = width / 2; topicNode.fy = height / 2 }
    linkGroup.selectAll('line')
      .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x).attr('y2', d => d.target.y)
    linkGroup.selectAll('text')
      .attr('x', d => (d.source.x + d.target.x) / 2)
      .attr('y', d => (d.source.y + d.target.y) / 2)
    node.attr('transform', d => `translate(${d.x},${d.y})`)
    label.attr('x', d => d.x).attr('y', d => d.y)
    badge.attr('x', d => d.x).attr('y', d => d.y)
  })

  // Defs
  const defs = _brainstormSvg.select('defs')
  if (defs.empty()) _brainstormSvg.insert('defs', ':first-child')
  const grad = _brainstormSvg.select('defs').selectAll('#bs-topic-grad').data([0])
  grad.join('enter').append('radialGradient').attr('id', 'bs-topic-grad')
    .attr('cx', '30%').attr('cy', '30%').attr('r', '70%')
    .each(function() {
      d3.select(this).append('stop').attr('offset', '0%').attr('stop-color', '#FCD34D').attr('stop-opacity', 0.8)
      d3.select(this).append('stop').attr('offset', '100%').attr('stop-color', '#F59E0B').attr('stop-opacity', 0.3)
    })

  renderBrainstormLegend()
}

function selectBrainstormNode(id) {
  if (!id) return
  _brainstormSelectedNodeId = id
  const connected = new Set([id])
  const edges = _brainstormGraphData.edges || []
  edges.forEach(e => {
    if ((e.source.id || e.source) === id) connected.add(e.target.id || e.target)
    if ((e.target.id || e.target) === id) connected.add(e.source.id || e.source)
  })
  _brainstormGNode.selectAll('g.bs-node').transition().duration(300)
    .attr('opacity', d => connected.has(d.id) ? 1 : 0.12)
  _brainstormGNode.selectAll('circle, rect').transition().duration(300)
    .attr('opacity', d => connected.has(d.id) ? 1 : 0.12)
  _brainstormGLink.selectAll('g.bs-link').each(function(d) {
    const el = d3.select(this)
    const sid = d.source.id || d.source, tid = d.target.id || d.target
    const isConnected = (sid === id || tid === id)
    el.select('line').transition().duration(300)
      .attr('stroke-opacity', isConnected ? 0.6 : 0.04)
      .attr('stroke-width', isConnected ? 2 : 0.8)
    el.select('text').transition().duration(300)
      .attr('opacity', isConnected ? 1 : 0.01)
  })
  showBrainstormNodeDetail(id)
}

function deselectBrainstormNode() {
  _brainstormSelectedNodeId = null
  if (!_brainstormGNode) return
  _brainstormGNode.selectAll('g.bs-node').transition().duration(300).attr('opacity', 1)
  _brainstormGNode.selectAll('circle, rect').transition().duration(300).attr('opacity', 1)
  _brainstormGLink.selectAll('g.bs-link').each(function(d) {
    const etype = d.type || 'generated'
    d3.select(this).select('line').transition().duration(300)
      .attr('stroke-opacity', (etype && etype !== 'generated' && etype !== 'related') ? 0.5 : 0.25)
      .attr('stroke-width', (etype && etype !== 'generated' && etype !== 'related') ? 1.5 : 0.8)
    d3.select(this).select('text').transition().duration(300)
      .attr('opacity', BS_EDGE_LABELS[etype] ? 0.7 : 0)
  })
  hideBrainstormDetail()
}

function highlightBrainstormConnections(id) {
  const connected = new Set([id])
  const edges = _brainstormGraphData.edges || []
  edges.forEach(e => {
    if ((e.source.id || e.source) === id) connected.add(e.target.id || e.target)
    if ((e.target.id || e.target) === id) connected.add(e.source.id || e.source)
  })
  _brainstormGNode.selectAll('g.bs-node').attr('opacity', d => connected.has(d.id) ? 1 : 0.1)
  _brainstormGLink.selectAll('g.bs-link').each(function(d) {
    const el = d3.select(this)
    const sid = d.source.id || d.source, tid = d.target.id || d.target
    const isConnected = (sid === id || tid === id)
    el.select('line').attr('stroke-opacity', isConnected ? 0.5 : 0.03)
    el.select('text').attr('opacity', isConnected ? 1 : 0.01)
  })
}

function clearBrainstormHighlight() {
  if (_brainstormSelectedNodeId) { selectBrainstormNode(_brainstormSelectedNodeId); return }
  _brainstormGNode.selectAll('g.bs-node').attr('opacity', 1)
  _brainstormGLink.selectAll('g.bs-link').each(function(d) {
    const etype = d.type || 'generated'
    d3.select(this).select('line').attr('stroke-opacity', (etype && etype !== 'generated' && etype !== 'related') ? 0.5 : 0.25)
    d3.select(this).select('text').attr('opacity', BS_EDGE_LABELS[etype] ? 0.7 : 0)
  })
}

function showBrainstormTooltip(event, d) {
  const tt = document.getElementById('brainstorm-graph-tooltip')
  if (!tt) return
  tt.classList.remove('hidden')
  const label = escHtml((d.text || '').substring(0, 60))
  const phaseLabel = isTopicNode(d) ? 'Topic' : d.synthesis ? 'Synthesis' : d.phase === 1 ? 'Phase 1' : 'Phase 2'
  const phaseColor = isTopicNode(d) ? '#F59E0B' : d.synthesis ? '#34D399' : d.phase === 1 ? '#F59E0B' : '#38BDF8'
  tt.innerHTML = `
    <div class="tt-title"><span style="color:${phaseColor}">${phaseLabel}</span> — ${label}</div>
    <div class="tt-meta">${((d.importance || 0.5) * 10).toFixed(1)}/10 significance</div>`
  const rect = _brainstormGraphContainer.getBoundingClientRect()
  const cx = event.clientX || event.sourceEvent?.clientX || 0
  const cy = event.clientY || event.sourceEvent?.clientY || 0
  tt.style.left = (cx - rect.left + 14) + 'px'
  tt.style.top = (cy - rect.top - 12) + 'px'
}

function hideBrainstormTooltip() {
  const tt = document.getElementById('brainstorm-graph-tooltip')
  if (tt) tt.classList.add('hidden')
}

function renderBrainstormLegend() {
  const legend = document.getElementById('brainstormGraphLegend')
  if (!legend) return
  const edgeTypesPresent = new Set()
  ;(_brainstormGraphData.edges || []).forEach(e => {
    const t = e.type || 'generated'
    if (t !== 'generated') edgeTypesPresent.add(t)
  })
  let html = ''
  html += '<div class="legend-item"><span class="legend-dot" style="background:#F59E0B"></span>Phase 1</div>'
  html += '<div class="legend-item"><span class="legend-dot" style="background:#38BDF8"></span>Phase 2</div>'
  html += '<div class="legend-item"><span class="legend-dot" style="background:#34D399"></span>Synthesis</div>'
  if (edgeTypesPresent.size > 0) {
    html += '<div class="legend-sep"></div>'
    edgeTypesPresent.forEach(t => {
      html += `<div class="legend-item"><span class="legend-dot" style="background:${bsEdgeColor(t)};border-radius:2px;height:3px;width:12px"></span>${escHtml(t)}</div>`
    })
  }
  legend.innerHTML = html
}

async function loadBrainstormSessions() {
  const listEl = document.getElementById('brainstormSessionList')
  const detailPanel = document.getElementById('brainstorm-detail-panel')
  detailPanel.classList.add('hidden')
  try {
    const data = await apiGet(API.brainstorm + '/sessions')
    const sessions = data.sessions || []
    _brainstormSessionsCache = sessions
    if (sessions.length === 0) {
      listEl.innerHTML = '<p class="text-secondary" style="padding:8px;font-size:11px">No topics yet</p>'
      deselectBrainstormSession()
      return
    }
    if (_brainstormSelectedSession && !sessions.find(s => s.id === _brainstormSelectedSession.id)) {
      deselectBrainstormSession()
    }
    renderBrainstormSidebar()
    if (!_brainstormSelectedSession) {
      deselectBrainstormSession()
    }
  } catch (e) {
    listEl.innerHTML = `<p class="text-secondary" style="padding:8px;font-size:11px">Failed to load: ${e.message}</p>`
  }
}

function renderBrainstormSidebar() {
  const listEl = document.getElementById('brainstormSessionList')
  const sessions = _brainstormSessionsCache
  listEl.innerHTML = sessions.map(s => {
    const date = s.created_at ? s.created_at.slice(0, 10) : '?'
    const nodeCount = (s.nodes || []).length
    const edgeCount = (s.edges || []).length
    const isActive = _brainstormSelectedSession && _brainstormSelectedSession.id === s.id
    return `<div class="bs-session-item ${isActive ? 'active' : ''}" data-id="${s.id}">
      <div style="display:flex;align-items:center;justify-content:space-between;width:100%">
        <div style="flex:1;min-width:0;padding-right:4px">
          <div class="bs-session-title">${isActive ? '<span class="bs-session-active-indicator">✓</span> ' : ''}${escHtml(s.topic || 'Untitled')}</div>
          <div class="bs-session-meta">${date} · ${nodeCount} ideas · ${edgeCount} connections</div>
        </div>
        <button class="bs-session-delete" data-id="${s.id}" title="Delete topic">
          <svg width="10" height="10" viewBox="0 0 10 10" fill="none"><path d="M2 2l6 6M8 2l-6 6" stroke="currentColor" stroke-width="1.2"/></svg>
        </button>
      </div>
    </div>`
  }).join('')
  listEl.querySelectorAll('.bs-session-item').forEach(el => {
    el.addEventListener('click', (e) => {
      if (e.target.closest('.bs-session-delete')) return
      selectBrainstormSession(el.dataset.id)
    })
  })
  listEl.querySelectorAll('.bs-session-delete').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation()
      const sid = btn.dataset.id
      if (!confirm('Delete this brainstorm session?')) return
      apiDelete(API.brainstorm + '/session/' + sid).then(() => {
        if (_brainstormSelectedSession && _brainstormSelectedSession.id === sid) {
          _brainstormSelectedSession = null
        }
        loadBrainstormSessions()
      }).catch(e => alert('Delete failed: ' + e.message))
    })
  })
}

function deselectBrainstormSession() {
  _brainstormSelectedSession = null
  _brainstormSelectedNodeId = null
  if (_brainstormSimulation) _brainstormSimulation.stop()
  document.getElementById('brainstorm-empty').classList.remove('hidden')
  document.getElementById('brainstorm-graph-container').classList.add('hidden')
  document.getElementById('brainstorm-summary').classList.add('hidden')
  document.getElementById('brainstorm-detail-panel').classList.add('hidden')
  document.querySelectorAll('.bs-session-item').forEach(el => el.classList.remove('active'))
}

async function selectBrainstormSession(sessionId) {
  if (_brainstormSelectedSession && _brainstormSelectedSession.id === sessionId) {
    deselectBrainstormSession()
    return
  }
  try {
    const session = await apiGet(API.brainstorm + '/session/' + sessionId)
    _brainstormSelectedSession = session
    _brainstormSelectedNodeId = null
    document.getElementById('brainstorm-empty').classList.add('hidden')
    document.getElementById('brainstorm-graph-container').classList.remove('hidden')
    document.getElementById('brainstorm-summary').classList.remove('hidden')
    document.getElementById('brainstorm-detail-panel').classList.add('hidden')
    renderBrainstormSidebar()
    renderBrainstormGraph(session)
    renderBrainstormSummary(session)
  } catch (e) {
    console.error('Failed to load session:', e)
  }
}



function showBrainstormNodeDetail(nodeId) {
  const panel = document.getElementById('brainstorm-detail-panel')
  const body = document.getElementById('bsDetailBody')
  const loading = document.getElementById('bsDetailLoading')
  const content = document.getElementById('bsDetailContent')
  panel.classList.remove('hidden')
  loading.classList.remove('hidden')
  content.classList.add('hidden')

  // Find the node in the session data
  const node = (_brainstormSelectedSession.nodes || []).find(n => n.id === nodeId)
  if (!node) {
    loading.classList.add('hidden')
    return
  }

  document.getElementById('bsDetailPanelTitle').textContent = (node.text || '').substring(0, 36) || 'Idea'

  // Convert old sessions that have scores as nested dict vs flat
  const scores = node.scores || {}
  const sNovelty = scores.novelty || scores.novelty === 0 ? scores.novelty : 0.5
  const sFeasibility = scores.feasibility || scores.feasibility === 0 ? scores.feasibility : 0.5
  const sRelevance = scores.relevance || scores.relevance === 0 ? scores.relevance : 0.8
  const sCoherence = scores.coherence || scores.coherence === 0 ? scores.coherence : 0.7

  // Idea text
  document.getElementById('bsDetailText').textContent = node.text || 'No content'

  // Phase
  const phaseEl = document.getElementById('bsDetailPhase')
  if (isTopicNode(node)) {
    phaseEl.textContent = '★ Topic — Core Question'
    phaseEl.style.color = '#F59E0B'
  } else if (node.synthesis) {
    phaseEl.textContent = 'Σ Synthesis — Cross-Pollination'
    phaseEl.style.color = '#34D399'
  } else {
    phaseEl.textContent = node.phase === 1 ? 'Phase 1 — Rough Draft' : 'Phase 2 — Expert Plan'
    phaseEl.style.color = node.phase === 1 ? '#F59E0B' : '#38BDF8'
  }

  // Thinking process (reasoning chain)
  const thinkingEl = document.getElementById('bsDetailThinkingChain')
  let reasoningChainHtml = ''
  // Show how this idea was derived
  if (node.reasoning && !isTopicNode(node)) {
    reasoningChainHtml += `<div class="thinking-step"><span class="thinking-arrow">→</span> ${escHtml(node.reasoning)}</div>`
  }
  // Show what this idea built upon (parent)
  if (node.parent_id) {
    const parent = (_brainstormSelectedSession.nodes || []).find(n => n.id === node.parent_id)
    if (parent) {
      const parentPreview = (parent.text || '').substring(0, 80)
      reasoningChainHtml += `<div class="thinking-step"><span class="thinking-arrow">↳</span> Built upon: <span class="thinking-ref">${escHtml(parentPreview)}</span></div>`
    }
  }
  // Show synthesis sources
  if (node.synthesis && node.synthesis_sources && node.synthesis_sources.length > 0) {
    const synSources = node.synthesis_sources.map(sid => {
      const sn = (_brainstormSelectedSession.nodes || []).find(n => n.id === sid)
      return sn ? (sn.text || '').substring(0, 50) : sid.substring(0, 8)
    })
    reasoningChainHtml += `<div class="thinking-step"><span class="thinking-arrow">Σ</span> Combined: ${synSources.map(s => `<span class="thinking-ref">${escHtml(s)}</span>`).join(' + ')}</div>`
  }
  thinkingEl.innerHTML = reasoningChainHtml || '<span class="text-secondary" style="font-size:11px">No reasoning recorded</span>'

  // Scores
  const scoresEl = document.getElementById('bsDetailScores')
  const scoreLabels = { novelty: 'Novelty', feasibility: 'Feasibility', relevance: 'Relevance', coherence: 'Coherence' }
  const svals = { novelty: sNovelty, feasibility: sFeasibility, relevance: sRelevance, coherence: sCoherence }
  scoresEl.innerHTML = Object.entries(scoreLabels).map(([key, label]) => {
    const val = svals[key] || 0
    return `<div class="bs-score"><span class="bs-score-label">${label}</span><span class="bs-score-bar"><span class="bs-score-fill" style="width:${val * 100}%;background:${val > 0.7 ? '#34D399' : val > 0.4 ? '#F59E0B' : '#FB7185'}"></span></span><span class="bs-score-val">${(val * 10).toFixed(0)}/10</span></div>`
  }).join('')

  // Skills
  const skillsEl = document.getElementById('bsDetailSkills')
  const skills = node.skills_used || []
  skillsEl.innerHTML = skills.length > 0
    ? skills.map(s => `<span class="bs-skill-pill">${escHtml(typeof s === 'string' ? s.split('/').pop() || s : s)}</span>`).join('')
    : '<span class="text-secondary">None</span>'

  // Connected ideas
  const connEl = document.getElementById('bsDetailConnections')
  const allEdges = _brainstormSelectedSession.edges || []
  const connectedEdges = allEdges.filter(e => {
    const sid = e.source.id || e.source, tid = e.target.id || e.target
    return sid === nodeId || tid === nodeId
  })
  if (connectedEdges.length > 0) {
    connEl.innerHTML = connectedEdges.map(e => {
      const sid = e.source.id || e.source, tid = e.target.id || e.target
      const otherId = sid === nodeId ? tid : sid
      const otherNode = (_brainstormSelectedSession.nodes || []).find(n => n.id === otherId)
      const otherLabel = otherNode ? (otherNode.text || '').substring(0, 40) : otherId.substring(0, 12)
      const color = otherNode ? bsNodeColor(otherNode) : '#55557a'
      const etype = e.type || 'generated'
      const typeLabel = etype !== 'generated' ? `<span class="edge-type" style="font-size:8px;color:${bsEdgeColor(etype)};margin-left:4px">${escHtml(etype)}</span>` : ''
      return `<div class="related-item" data-id="${otherId}"><span class="related-dot" style="background:${color}"></span>${escHtml(otherLabel)}${typeLabel}</div>`
    }).join('')
    connEl.querySelectorAll('.related-item').forEach(el => el.addEventListener('click', () => {
      selectBrainstormNode(el.dataset.id)
    }))
  } else {
    connEl.innerHTML = '<p class="text-secondary">No connections</p>'
  }

  // Source memories
  const sourcesEl = document.getElementById('bsDetailSources')
  const sources = node.source_memories || []
  sourcesEl.innerHTML = sources.length > 0
    ? sources.map(s => `<code class="bs-source-id">${escHtml(s.slice(0, 12))}</code>`).join(' ')
    : '<span class="text-secondary">None</span>'

  loading.classList.add('hidden')
  content.classList.remove('hidden')
}

function hideBrainstormDetail() {
  document.getElementById('brainstorm-detail-panel').classList.add('hidden')
}

function renderBrainstormSummary(session) {
  const topic = session.topic || 'Topic'
  const date = session.created_at ? new Date(session.created_at).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' }) : ''
  const nodes = session.nodes || []
  const edges = session.edges || []
  const phase2Nodes = nodes.filter(n => n.phase === 2 && !n.synthesis)
  const synthesisNodes = nodes.filter(n => n.synthesis)

  document.getElementById('bsSummaryTopic').textContent = topic
  document.getElementById('bsSummaryDate').textContent = date
  document.getElementById('bsSummaryNodeCount').textContent = nodes.length
  document.getElementById('bsSummaryEdgeCount').textContent = edges.length

  // Idea preview cards
  const ideasEl = document.getElementById('bsSummaryIdeas')
  if (phase2Nodes.length > 0) {
    ideasEl.innerHTML = phase2Nodes.map(n => {
      const text = (n.text || '').substring(0, 80)
      const scores = n.scores || {}
      const avg = ((scores.novelty || 0.5) + (scores.feasibility || 0.5)) / 2
      return `<div class="bs-summary-idea-card" data-id="${n.id}">
        <div class="bs-summary-idea-text">${escHtml(text)}</div>
        <div class="bs-summary-idea-meta">
          <span style="color:#38BDF8">P2</span>
          <span>${(avg * 10).toFixed(0)}/10 avg</span>
        </div>
      </div>`
    }).join('')
    ideasEl.querySelectorAll('.bs-summary-idea-card').forEach(el => {
      el.addEventListener('click', () => selectBrainstormNode(el.dataset.id))
    })
  } else {
    ideasEl.innerHTML = '<span class="text-secondary" style="font-size:11px">No Phase 2 ideas generated yet.</span>'
  }

  // Synthesis preview
  const synEl = document.getElementById('bsSummarySynthesis')
  if (synthesisNodes.length > 0) {
    const synText = (synthesisNodes[0].text || '').substring(0, 200)
    synEl.textContent = 'Synthesis: ' + synText + (synText.length >= 200 ? '...' : '')
    synEl.classList.remove('hidden')
  } else {
    synEl.classList.add('hidden')
  }
}

function showFullBrainstormSummary() {
  const session = _brainstormSelectedSession
  if (!session) return
  const nodes = session.nodes || []
  const edges = session.edges || []
  const phase2Nodes = nodes.filter(n => n.phase === 2 && !n.synthesis)
  const synthesisNodes = nodes.filter(n => n.synthesis)
  const phase1Nodes = nodes.filter(n => n.phase === 1 && !n.synthesis)

  document.getElementById('bsFullSummaryTitle').textContent = (session.topic || 'Brainstorm') + ' — Full Summary'

  // Info
  const date = session.created_at ? new Date(session.created_at).toLocaleDateString() : '?'
  document.getElementById('bsFullSummaryInfo').innerHTML = `
    <div class="detail-section">
      <label class="detail-label">Topic</label>
      <p style="font-size:15px;font-weight:600;color:var(--amber)">${escHtml(session.topic || 'Untitled')}</p>
      <p style="font-size:11px;color:var(--text-muted);margin-top:4px">${date} · ${nodes.length} ideas · ${edges.length} connections</p>
    </div>
  `

  // Ideas
  const ideasEl = document.getElementById('bsFullSummaryIdeas')
  let ideasHtml = '<label class="detail-label" style="margin-top:16px;margin-bottom:8px;display:block">Ideas</label>'

  for (const group of [
    { label: 'Phase 2 — Expert Plans', nodes: phase2Nodes, color: '#38BDF8' },
    { label: 'Phase 1 — Rough Drafts', nodes: phase1Nodes, color: '#F59E0B' },
  ]) {
    if (group.nodes.length === 0) continue
    ideasHtml += `<label class="detail-label" style="margin-top:12px;margin-bottom:6px;display:block;color:${group.color}">${group.label}</label>`
    ideasHtml += group.nodes.map(n => {
      const scores = n.scores || {}
      const skills = n.skills_used || []
      const scoreHtml = Object.entries(scores).filter(([k]) => ['novelty','feasibility','relevance','coherence'].includes(k))
        .map(([k, v]) => `<span style="margin-right:6px">${k}: ${((v || 0) * 10).toFixed(0)}/10</span>`).join('')
      const skillsHtml = skills.map(s => `<span class="bs-skill-pill">${escHtml(typeof s === 'string' ? s.split('/').pop() || s : s)}</span>`).join(' ')
      return `<div class="bs-full-idea">
        <div class="bs-full-idea-text">${escHtml(n.text || '')}</div>
        <div class="bs-full-idea-meta">
          <span class="bs-phase-badge" style="color:${group.color}">${n.phase === 1 ? 'P1' : 'P2'}</span>
          <div class="bs-full-idea-scores">${scoreHtml}</div>
          ${skillsHtml ? `<span>⚡ ${skillsHtml}</span>` : ''}
        </div>
      </div>`
    }).join('')
  }
  ideasEl.innerHTML = ideasHtml

  // Synthesis
  const synEl = document.getElementById('bsFullSummarySynthesis')
  if (synthesisNodes.length > 0) {
    synEl.innerHTML = '<label class="detail-label" style="margin-top:16px;margin-bottom:8px;display:block;color:#34D399">Synthesis</label>\n' +
      synthesisNodes.map(n => `<div class="bs-full-synthesis">${escHtml(n.text || '')}</div>`).join('')
    synEl.classList.remove('hidden')
  } else {
    synEl.classList.add('hidden')
  }

  document.getElementById('brainstorm-summary-full').classList.remove('hidden')
}

function hideFullBrainstormSummary() {
  document.getElementById('brainstorm-summary-full').classList.add('hidden')
}

document.getElementById('btnCloseBrainstormDetail').addEventListener('click', hideBrainstormDetail)
document.getElementById('btnShowMoreSummary').addEventListener('click', showFullBrainstormSummary)
document.getElementById('btnCloseFullSummary').addEventListener('click', hideFullBrainstormSummary)

document.getElementById('btnNewBrainstorm').addEventListener('click', () => {
  const topic = prompt('What should we brainstorm about?', '')
  if (!topic || !topic.trim()) return
  apiPost(API.brainstorm, { topic: topic.trim(), n_ideas: 5 })
    .then(data => {
      if (data.session_id) {
        loadBrainstormSessions()
      }
    })
    .catch(e => alert('Brainstorm failed: ' + e.message))
})

function escHtml(s) { if (!s) return ''; const d = document.createElement('div'); d.textContent = s; return d.innerHTML }

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('btnGraphView').addEventListener('click', () => switchView('graph'))
  document.getElementById('btnBrowserView').addEventListener('click', () => switchView('browser'))
  document.getElementById('btnBrainstormView').addEventListener('click', () => switchView('brainstorm'))
  document.getElementById('btnProfileEdit').addEventListener('click', openProfileModal)
  document.getElementById('btnAddMemory').addEventListener('click', () => document.getElementById('addModal').classList.remove('hidden'))
  document.getElementById('btnEmptyAdd').addEventListener('click', () => document.getElementById('addModal').classList.remove('hidden'))
  document.getElementById('modalClose').addEventListener('click', () => document.getElementById('addModal').classList.add('hidden'))
  document.getElementById('modalCancel').addEventListener('click', () => document.getElementById('addModal').classList.add('hidden'))
  document.getElementById('profileModalClose').addEventListener('click', closeProfileModal)
  document.getElementById('profileCancel').addEventListener('click', closeProfileModal)
  document.getElementById('profileSave').addEventListener('click', saveProfile)

  document.getElementById('modalSave').addEventListener('click', async () => {
    const text = document.getElementById('modalText').value.trim()
    if (!text) return
    const source = document.getElementById('modalSource').value.trim() || 'manual'
    const importance = parseFloat(document.getElementById('modalImportance').value)
    const tagsStr = document.getElementById('modalTags').value.trim()
    const tags = tagsStr ? tagsStr.split(',').map(t => t.trim()).filter(Boolean) : undefined
    try {
      await apiPost(API.memories, { text, source, importance, tags })
      document.getElementById('addModal').classList.add('hidden')
      await loadAll()
    } catch(e) { alert('Failed: ' + e.message) }
  })

  document.getElementById('modalImportance').addEventListener('input', () => {
    document.getElementById('modalImportanceVal').textContent = parseFloat(document.getElementById('modalImportance').value).toFixed(2)
  })
  document.getElementById('detailImportance').addEventListener('input', () => {
    document.getElementById('detailImportanceVal').textContent = parseFloat(document.getElementById('detailImportance').value).toFixed(2)
  })
  document.getElementById('btnCloseDetail').addEventListener('click', deselectNode)
  document.getElementById('btnShowLocalGraph').addEventListener('click', () => { if (selectedNodeId) showLocalGraph(selectedNodeId) })
  document.getElementById('btnSaveMemory').addEventListener('click', async () => {
    if (!selectedNodeId) return
    try {
      await apiPut(`${API.memories}/${selectedNodeId}`, {
        text: document.getElementById('detailText').value.trim(),
        source: document.getElementById('detailSource').value.trim(),
        importance: parseFloat(document.getElementById('detailImportance').value),
        tags: document.getElementById('detailTags').value.trim().split(',').map(t => t.trim()).filter(Boolean),
      })
    } catch(e) { alert('Save failed: ' + e.message) }
  })
  document.getElementById('btnDeleteMemory').addEventListener('click', async () => {
    if (!selectedNodeId || !confirm('Delete this memory?')) return
    try { await apiDelete(`${API.memories}/${selectedNodeId}`); deselectNode(); await loadAll() } catch(e) { alert('Delete failed: ' + e.message) }
  })
  document.getElementById('prevPage').addEventListener('click', () => { currentPage = Math.max(0, currentPage - 1); loadBrowser() })
  document.getElementById('nextPage').addEventListener('click', () => { currentPage++; loadBrowser() })
  document.getElementById('errorRetry').addEventListener('click', loadBrowser)
  document.querySelectorAll('.sort-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.sort-btn').forEach(b => b.classList.remove('active'))
      btn.classList.add('active')
      sortMode = btn.dataset.sort
      currentPage = 0
      loadBrowser()
    })
  })
  document.getElementById('tagsHeader').addEventListener('click', () => {
    document.getElementById('tagsHeader').classList.toggle('collapsed')
    document.getElementById('tagsList').classList.toggle('hidden')
  })
  document.getElementById('nodeLimitSlider').addEventListener('input', () => {
    document.getElementById('nodeLimitVal').textContent = document.getElementById('nodeLimitSlider').value
    buildGraph()
  })
  document.getElementById('minImpSlider').addEventListener('input', () => {
    document.getElementById('minImpVal').textContent = parseFloat(document.getElementById('minImpSlider').value).toFixed(2)
    buildGraph()
  })

  initGraph()
  switchView('graph')
  loadAll()

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      if (!document.getElementById('brainstorm-summary-full').classList.contains('hidden')) hideFullBrainstormSummary()
      else if (!document.getElementById('addModal').classList.contains('hidden')) document.getElementById('addModal').classList.add('hidden')
      else if (!document.getElementById('profileModal').classList.contains('hidden')) closeProfileModal()
      else deselectNode()
    }
    if (e.ctrlKey && e.key === 'f') { e.preventDefault(); document.getElementById('searchInput').focus() }
  })
})
