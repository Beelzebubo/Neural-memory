// ── Neural Memory — Lino UI ──

const API = {
  memories: '/api/memories',
  graph: '/api/graph',
  search: '/api/search',
  stats: '/api/stats',
  profile: '/api/profile',
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

function escHtml(s) { if (!s) return ''; const d = document.createElement('div'); d.textContent = s; return d.innerHTML }

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('btnGraphView').addEventListener('click', () => switchView('graph'))
  document.getElementById('btnBrowserView').addEventListener('click', () => switchView('browser'))
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
      if (!document.getElementById('addModal').classList.contains('hidden')) document.getElementById('addModal').classList.add('hidden')
      else if (!document.getElementById('profileModal').classList.contains('hidden')) closeProfileModal()
      else deselectNode()
    }
    if (e.ctrlKey && e.key === 'f') { e.preventDefault(); document.getElementById('searchInput').focus() }
  })
})
