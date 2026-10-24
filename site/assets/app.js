/* Claims fraud triage dashboard.
 * Reads the JSON embedded in the page; when served by the API it also goes live
 * (copilot chat, scoring form, FNOL feed). Every data-derived string goes in via
 * textContent, never innerHTML. */
(() => {
  'use strict';

  const DATA = JSON.parse(document.getElementById('dashboard-data').textContent);
  const M = DATA.metrics;
  const POLICY = M.policy;
  const CHAMP = M.candidates[M.champion.candidate].test;
  const RULES = M.candidates.rules.test;
  const state = { live: false, selected: null, ring: 0, rendered: new Set(), health: null };

  /* ---------- small helpers ---------------------------------------------------------------- */
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
      if (value === null || value === undefined || value === false) continue;
      if (key === 'class') node.className = value;
      else if (key === 'text') node.textContent = value;
      else if (key === 'style') node.setAttribute('style', value);
      else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value === true ? '' : value);
    }
    for (const child of children.flat()) {
      if (child === null || child === undefined || child === false) continue;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  const money = (value) => {
    const abs = Math.abs(value);
    const sign = value < 0 ? '−' : '';
    if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(2).replace(/\.?0+$/, '')}M`;
    if (abs >= 1e4) return `${sign}$${Math.round(abs / 1e3)}K`;
    if (abs >= 1e3) return `${sign}$${(abs / 1e3).toFixed(1).replace(/\.0$/, '')}K`;
    return `${sign}$${Math.round(abs)}`;
  };
  const moneyFull = (value) => `${value < 0 ? '−' : ''}$${Math.round(Math.abs(value)).toLocaleString('en-CA')}`;
  const pct = (value, digits = 0) => `${(value * 100).toFixed(digits)}%`;
  const num = (value, digits = 1) => Number(value).toFixed(digits).replace(/\.0+$/, '');
  const label = (text) => String(text).replace(/_/g, ' ').replace(/^\w/, (c) => c.toUpperCase());
  const plural = (n, word) => `${n} ${word}${Number(n) === 1 ? '' : 's'}`;
  const shortDate = (iso) => new Date(`${iso}T00:00:00`).toLocaleString('en-CA', { month: 'short', day: 'numeric' });
  const monthName = (ym) => new Date(`${ym}-01T00:00:00`).toLocaleString('en-CA', { month: 'short', year: 'numeric' });
  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  const FEATURE_LABELS = {
    narrative_risk: 'Narrative resembles fraud notes', shop_avg_repair_ratio_180d: "Shop's avg repair-to-value",
    repair_to_value: 'Repair vs vehicle value', log_repair_estimate: 'Repair estimate', loss_type: 'Loss type',
    shop_prior_claims_90d: 'Body shop volume (90d)', log_total_claimed: 'Total claimed', policy_tenure_days: 'Policy age',
    n_injured: 'People injured', vehicle_value: 'Vehicle value', report_delay_days: 'Days to report',
    prior_claims_3y: 'Prior claims (3y)', vehicle_age: 'Vehicle age', loss_hour: 'Hour of loss', injured_share: 'Share injured',
    rep_clinic_prior_claims_365d: 'Rep + clinic pairings', phone_prior_claims_365d: 'Phone reuse (1y)',
    address_prior_claims_365d: 'Address reuse (1y)', clinic_prior_claims_90d: 'Clinic volume (90d)', province: 'Province',
    soft_tissue_only: 'Soft-tissue only', represented: 'Represented', log_injury_reserve: 'Injury reserve',
    tow_fee: 'Towing & storage fees', storage_days: 'Storage days', tow_distance_km: 'Tow distance',
    tow_avg_storage_days_180d: "Tow operator's avg storage", tow_prior_claims_90d: 'Tow operator volume (90d)',
    total_loss: 'Total loss', night_loss: 'Night-time loss', n_occupants: 'Occupants', police_report: 'Police report',
    witnesses: 'Witnesses',
  };
  const featureLabel = (name) => FEATURE_LABELS[name] || label(name);
  const CANDIDATE_NAMES = { rules: 'Red-flag scorecard', logreg: 'Logistic regression', lightgbm: 'LightGBM' };
  const BAND_CLASS = { Investigate: 'band-investigate', Review: 'band-review', Standard: 'band-standard', 'Fast-track': 'band-fast' };
  const bandChip = (band) => el('span', { class: `chip band ${BAND_CLASS[band] || ''}`, text: band });

  const ICONS = {
    alert: 'M12 3 2 20h20L12 3Zm0 6v5m0 3v.5',
    warn: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18Zm0 5v5m0 3v.5',
    stable: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18Zm-4 9 3 3 5-6',
  };
  function statusBadge(status) {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('aria-hidden', 'true');
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', ICONS[status]);
    path.setAttribute('fill', 'none');
    path.setAttribute('stroke', 'currentColor');
    path.setAttribute('stroke-width', '2');
    path.setAttribute('stroke-linecap', 'round');
    path.setAttribute('stroke-linejoin', 'round');
    svg.append(path);
    const text = { alert: 'Alert', warn: 'Watch', stable: 'Stable' }[status];
    return el('span', { class: `status status-${status}` }, svg, text);
  }

  function tile(labelText, value, delta, tone) {
    const deltaNode = delta ? el('div', { class: 'tile-delta' }) : null;
    if (delta) {
      if (tone) deltaNode.append(el('span', { class: tone, text: delta[0] }), ` ${delta[1] || ''}`);
      else deltaNode.textContent = delta.join(' ');
    }
    return el('div', { class: 'tile' }, el('div', { class: 'tile-label', text: labelText }),
      el('div', { class: 'tile-value', text: value }), deltaNode);
  }

  function table(columns, rows, className = 'data-table compact-table') {
    const thead = el('thead', {}, el('tr', {}, columns.map((c) => el('th', { scope: 'col', class: c.num ? 'num' : null, text: c.label }))));
    const tbody = el('tbody', {}, rows.map((row) => el('tr', {}, columns.map((c) => {
      const value = c.render ? c.render(row) : row[c.key];
      return el('td', { class: c.num ? 'num' : null }, value instanceof Node ? value : String(value ?? '—'));
    }))));
    return el('table', { class: className }, thead, tbody);
  }

  function meter(value, max = 1) {
    return el('div', { class: 'meter', role: 'presentation' },
      el('span', { style: `width:${Math.max(2, Math.min(100, (value / max) * 100)).toFixed(1)}%` }));
  }

  /* ---------- Plotly theme ---------------------------------------------------------------------- */
  const PLOT_CONFIG = { displayModeBar: false, responsive: true };
  const charts = new Map();

  function niceStep(span, count = 5) {
    const raw = span / count;
    const power = 10 ** Math.floor(Math.log10(raw || 1));
    const n = raw / power;
    return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10) * power;
  }
  function moneyAxis(values) {
    const lo = Math.min(0, ...values);
    const hi = Math.max(...values);
    const step = niceStep(hi - lo);
    const tickvals = [];
    for (let v = Math.floor(lo / step) * step; v <= hi + step * 0.5; v += step) tickvals.push(v);
    return { tickvals, ticktext: tickvals.map((v) => (v === 0 ? '$0' : money(v))) };
  }

  function layout(extra = {}) {
    const text2 = cssVar('--text-secondary');
    const muted = cssVar('--axis-muted');
    const grid = cssVar('--grid');
    const axis = cssVar('--axis');
    const axisBase = {
      gridcolor: grid, gridwidth: 1, linecolor: axis, linewidth: 1, zeroline: false, ticks: '',
      tickfont: { color: muted, size: 11.5 }, title: { font: { color: text2, size: 12 } }, automargin: true,
    };
    const base = {
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: { family: cssVar('--font') || 'system-ui', color: text2, size: 12 },
      margin: { l: 56, r: 16, t: 8, b: 44 },
      hoverlabel: { bgcolor: cssVar('--surface-2'), bordercolor: cssVar('--border-strong'),
        font: { color: cssVar('--text-primary'), family: cssVar('--font'), size: 12 } },
      legend: { orientation: 'h', x: 0, y: 1.12, xanchor: 'left', font: { color: text2, size: 12 }, bgcolor: 'rgba(0,0,0,0)' },
      showlegend: false,
    };
    const merged = { ...base, ...extra };
    for (const key of ['xaxis', 'yaxis']) {
      const own = extra[key] || {};
      merged[key] = { ...axisBase, ...own, title: { ...axisBase.title, ...(own.title || {}) } };
    }
    if (merged.showlegend && window.innerWidth < 640) {
      // On a phone a top legend wraps over the plot; put it under the x-axis instead.
      const margin = merged.margin || {};
      merged.legend = { ...merged.legend, y: -0.22, yanchor: 'top' };
      merged.margin = { ...margin, t: 12, b: (margin.b || 44) + 64 };
    }
    return merged;
  }

  function chart(id, render) {
    charts.set(id, render);
    const node = document.getElementById(id);
    if (node && node.offsetParent !== null) render(node);
  }
  function rerenderVisible() {
    for (const [id, render] of charts) {
      const node = document.getElementById(id);
      if (node && node.offsetParent !== null) render(node);
    }
  }

  /* ---------- theme toggle ------------------------------------------------------------------------ */
  const THEME_KEY = 'fraudops-theme';
  function effectiveDark() {
    const forced = document.documentElement.dataset.theme;
    if (forced) return forced === 'dark';
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  }
  function applyTheme(theme) {
    if (theme) document.documentElement.dataset.theme = theme;
    else delete document.documentElement.dataset.theme;
    rerenderVisible();
  }
  try {
    const saved = localStorage.getItem(THEME_KEY);
    if (saved === 'light' || saved === 'dark') document.documentElement.dataset.theme = saved;
  } catch (_) { /* storage may be unavailable */ }
  $('#theme-toggle').addEventListener('click', () => {
    const next = effectiveDark() ? 'light' : 'dark';
    try { localStorage.setItem(THEME_KEY, next); } catch (_) { /* ignore */ }
    applyTheme(next);
  });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    if (!document.documentElement.dataset.theme) rerenderVisible();
  });

  /* ---------- tabs ------------------------------------------------------------------------------------ */
  const TABS = $$('.tabs [role="tab"]');
  function showTab(name, push = true) {
    if (!document.getElementById(name)) name = 'overview';
    for (const tab of TABS) {
      const active = tab.dataset.tab === name;
      tab.setAttribute('aria-selected', String(active));
      tab.tabIndex = active ? 0 : -1;
      document.getElementById(tab.dataset.tab).hidden = !active;
    }
    if (push) {
      history.replaceState(null, '', `#${name}`);
      window.scrollTo(0, 0);
    }
    requestAnimationFrame(rerenderVisible);
  }
  TABS.forEach((tab, index) => {
    tab.addEventListener('click', () => showTab(tab.dataset.tab));
    tab.addEventListener('keydown', (event) => {
      if (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft') return;
      const next = TABS[(index + (event.key === 'ArrowRight' ? 1 : TABS.length - 1)) % TABS.length];
      next.focus();
      showTab(next.dataset.tab);
    });
  });

  for (const node of $$('[data-bind="capacity"]')) node.textContent = String(POLICY.capacity);
  for (const node of $$('[data-bind="psi-alert"]')) node.textContent = String(DATA.drift.thresholds.alert);

  /* ---------- Overview ------------------------------------------------------------------------------- */
  function renderOverview() {
    const uplift = CHAMP.net_savings - RULES.net_savings;
    const months = M.data.test.months;
    $('#hero-value').textContent = `${uplift >= 0 ? '+' : ''}${money(uplift)}`;
    $('#hero-note').textContent = `About ${money(uplift * 12)} a year with the same ${POLICY.capacity} files a month. `
      + `Held-out test months ${monthName(months[0])} to ${monthName(months[1])}.`;

    const fast = DATA.latest.hidden_truth.fast_track;
    const bands = DATA.latest.bands;
    const fastShare = (bands['Fast-track'] || 0) / DATA.latest.claims;
    const honestDrop = 1 - CHAMP.honest_investigated / RULES.honest_investigated;
    $('#kpis').replaceChildren(
      tile('Fraud dollars caught per month', money(CHAMP.fraud_dollars_caught),
        [`+${money(CHAMP.fraud_dollars_caught - RULES.fraud_dollars_caught)}`, `vs. scorecard (${money(RULES.fraud_dollars_caught)})`], 'up-good'),
      tile('Opened files that were fraud', pct(CHAMP.hit_rate),
        [`${pct(CHAMP.hit_rate - RULES.hit_rate)} pts`, `vs. ${pct(RULES.hit_rate)} for the scorecard`], 'up-good'),
      tile('Honest customers investigated per month', num(CHAMP.honest_investigated),
        [`−${pct(honestDrop)}`, `vs. ${num(RULES.honest_investigated)} with the scorecard`], 'down-good'),
      tile(`Claims fast-tracked in ${monthName(DATA.month)}`, pct(fastShare),
        [`${fast.fraud_claims_missed} frauds (${money(fast.fraud_dollars_missed)}) slipped through`, '— hindsight audit'], 'bad'),
    );

    chart('chart-capacity', (node) => {
      const curve = M.capacity_curve;
      const all = [...curve.model, ...curve.rules, ...curve.random];
      const ticks = moneyAxis(all);
      const hover = (name) => `<b>%{customdata}</b>  ${name}<extra></extra>`;
      const series = [
        { key: 'random', name: 'Random files', color: cssVar('--deemph') },
        { key: 'rules', name: 'Red-flag scorecard (today)', color: cssVar('--series-2') },
        { key: 'model', name: 'Model, ranked by expected saving', color: cssVar('--series-1') },
      ];
      const traces = series.map((s) => ({
        x: curve.capacity, y: curve[s.key], customdata: curve[s.key].map(moneyFull), name: s.name, type: 'scatter',
        mode: 'lines', line: { color: s.color, width: 2, shape: 'spline', smoothing: 0.4 }, hovertemplate: hover(s.name),
      }));
      const at = curve.capacity.indexOf(POLICY.capacity);
      traces.push({
        x: [POLICY.capacity, POLICY.capacity], y: [curve.model[at], curve.rules[at]], mode: 'markers', type: 'scatter',
        marker: { size: 9, color: [cssVar('--series-1'), cssVar('--series-2')], line: { color: cssVar('--surface-1'), width: 2 } },
        hoverinfo: 'skip', showlegend: false,
      });
      Plotly.react(node, traces, layout({
        showlegend: true, hovermode: 'x unified', margin: { l: 60, r: 16, t: 34, b: 44 },
        legend: { orientation: 'h', x: 0, y: 1.16, traceorder: 'reversed', font: { color: cssVar('--text-secondary'), size: 12 } },
        xaxis: { title: { text: 'Files opened per month' }, showgrid: false, range: [0, curve.capacity.at(-1)] },
        yaxis: { tickvals: ticks.tickvals, ticktext: ticks.ticktext },
        shapes: [{ type: 'line', x0: POLICY.capacity, x1: POLICY.capacity, yref: 'paper', y0: 0, y1: 1,
          line: { color: cssVar('--axis'), width: 1 } }],
        annotations: [{ x: POLICY.capacity, y: 1, yref: 'paper', text: `Capacity today: ${POLICY.capacity}`, showarrow: false,
          xanchor: 'left', yanchor: 'top', xshift: 6, font: { size: 11.5, color: cssVar('--text-secondary') } }],
      }), PLOT_CONFIG);
    });
    const curve = M.capacity_curve;
    $('#table-capacity').replaceChildren(table(
      [{ label: 'Files / month', key: 'c', num: true }, { label: 'Model', key: 'm', num: true },
        { label: 'Scorecard', key: 'r', num: true }, { label: 'Random', key: 'x', num: true }],
      curve.capacity.map((c, i) => ({ c, m: moneyFull(curve.model[i]), r: moneyFull(curve.rules[i]), x: moneyFull(curve.random[i]) })),
    ));

    const order = [['Investigate', '--seq-550', 'SIU queue'], ['Review', '--seq-400', 'adjuster red-flag review'],
      ['Standard', '--seq-250', 'normal handling'], ['Fast-track', '--seq-150', 'straight-through payment']];
    $('#bands-caption').textContent = `${DATA.latest.claims} claims scored for ${monthName(DATA.month)}.`;
    const bar = el('div', { class: 'bandbar', role: 'img',
      'aria-label': order.map(([name]) => `${name} ${bands[name] || 0}`).join(', ') });
    for (const [name, color] of order) {
      if (bands[name]) bar.append(el('span', { style: `width:${(bands[name] / DATA.latest.claims) * 100}%;background:var(${color})`, title: `${name}: ${bands[name]}` }));
    }
    $('#bands').replaceChildren(bar, el('div', { class: 'band-legend' }, order.map(([name, color, what]) =>
      el('div', {}, el('i', { style: `background:var(${color})` }), el('span', {}, name, el('small', { text: what })),
        el('b', { text: String(bands[name] || 0) })))));

    const biz = DATA.business;
    $('#assumptions').replaceChildren(...[
      ['SIU capacity', `${biz.siu.capacity_per_month} files / month`],
      ['Cost per investigation', moneyFull(biz.siu.investigation_cost)],
      ['Recovery when fraud is confirmed', pct(biz.siu.recovery_rate)],
      ['Fast-track below', pct(biz.triage.fast_track_below)],
      ['Adjuster review from', pct(biz.triage.review_above)],
    ].flatMap(([k, v]) => [el('dt', { text: k }), el('dd', { text: v })]));
  }

  /* ---------- Queue ------------------------------------------------------------------------------------- */
  const TITLES = DATA.titles || {};
  const tagTitle = (id) => (TITLES[id] ? `${id} · ${TITLES[id].title}` : id);

  function renderQueue() {
    const body = $('#queue-table tbody');
    body.replaceChildren(...DATA.queue.map((claim) => {
      const row = el('tr', { 'data-claim': claim.claim_id, tabindex: '0', 'aria-selected': 'false',
        class: claim.rank === POLICY.capacity + 1 ? 'cutoff' : null },
      el('td', { class: 'num', text: String(claim.rank) }),
      el('td', {}, el('span', { class: 'claim-id', text: claim.claim_id }), el('span', { class: 'sub', text: shortDate(claim.reported) })),
      el('td', {}, label(claim.loss_type), el('span', { class: 'sub', text: `${claim.city}, ${claim.province}` })),
      el('td', { class: 'num', text: money(claim.total_claimed) }),
      el('td', {}, el('div', { class: 'prob' }, el('b', { text: pct(claim.probability) }), meter(claim.probability))),
      el('td', { class: 'num', text: money(claim.expected_value) }),
      el('td', {}, el('div', { class: 'links' }, bandChip(claim.band), claim.tags.map((t) => el('span', { class: 'chip', title: tagTitle(t), text: t })))));
      row.addEventListener('click', () => selectClaim(claim.claim_id));
      row.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); selectClaim(claim.claim_id); } });
      return row;
    }));
    selectClaim(DATA.queue[0].claim_id);
  }

  function highlighted(text, terms) {
    const node = el('p', { class: 'notes' });
    if (!terms.length) { node.textContent = text; return node; }
    const pattern = new RegExp(`(${terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'gi');
    let last = 0;
    text.replace(pattern, (match, _g, offset) => {
      node.append(document.createTextNode(text.slice(last, offset)), el('mark', { text: match }));
      last = offset + match.length;
      return match;
    });
    node.append(document.createTextNode(text.slice(last)));
    return node;
  }

  function reasonRows(items, direction, max) {
    return items.map((item) => el('div', { class: `reason ${direction}` }, el('span', { text: item.text }),
      el('div', { class: 'bar', title: `${item.weight > 0 ? '+' : ''}${item.weight} log-odds` },
        el('span', { style: `width:${Math.max(4, (Math.abs(item.weight) / max) * 100)}%` }))));
  }

  function renderAnswer(answer, recorded) {
    const meta = el('div', { class: 'answer-meta' },
      el('span', { class: 'chip', text: answer.mode === 'generated' ? 'Generated' : answer.mode === 'abstained' ? 'Declined' : 'Evidence only' }),
      el('span', { class: 'chip', text: `${answer.retrieval} retrieval` }),
      answer.model ? el('span', { class: 'chip', text: answer.model }) : null,
      el('span', { class: 'chip', text: `${(answer.elapsed_ms / 1000).toFixed(1)} s` }),
      el('span', { class: 'chip', text: 'Local model only' }),
      recorded ? el('span', { class: 'chip', text: 'Recorded answer' }) : null);
    const sources = el('div', { class: 'sources' }, answer.sources.map((s) => el('div', { class: 'source' },
      el('b', { text: `[${s.label}]` }), el('div', {}, `${s.title} — ${s.heading}`, el('span', { class: 'sub', text: s.snippet })))));
    return el('div', {}, el('div', { class: 'answer', text: answer.answer }), meta,
      el('details', { class: 'table-view' }, el('summary', { text: `Sources (${answer.sources.length})` }), sources));
  }

  async function askCopilot(question, claimId) {
    const response = await fetch('api/copilot/ask', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ question, claim_id: claimId || null }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || `HTTP ${response.status}`);
    return body;
  }

  function selectClaim(claimId) {
    const claim = DATA.queue.find((c) => c.claim_id === claimId);
    if (!claim) return;
    state.selected = claimId;
    for (const row of $$('#queue-table tbody tr')) row.setAttribute('aria-selected', String(row.dataset.claim === claimId));
    const f = claim.facts;
    const facts = [
      ['Policy age', plural(f.policy_tenure_days, 'day')], ['Reported after', plural(f.report_delay_days, 'day')],
      ['Injured / occupants', `${f.n_injured} / ${f.n_occupants}`], ['Soft-tissue only', f.soft_tissue_only ? 'Yes' : 'No'],
      ['Police report', f.police_report ? 'Yes' : 'No'], ['Witnesses', String(f.witnesses)],
      ['Prior claims (3y)', String(f.prior_claims_3y)], ['Repair estimate', moneyFull(f.repair_estimate)],
      ['Injury reserve', moneyFull(f.injury_reserve)], ['Towing & storage', moneyFull(f.tow_fee)],
      ['Clinic', f.clinic_id || '—'], ['Representative', f.rep_id || '—'],
      ['Body shop', f.body_shop_id || '—'], ['Tow operator', f.tow_company_id || '—'], ['Loss hour', `${String(f.loss_hour).padStart(2, '0')}:00`],
    ];
    const max = Math.max(...claim.reasons_for.map((r) => Math.abs(r.weight)), ...claim.reasons_against.map((r) => Math.abs(r.weight)), 0.01);
    const terms = claim.reasons_for.flatMap((r) => [...r.text.matchAll(/'([^']+)'/g)].map((m) => m[1]));
    const recorded = (DATA.copilot_examples || []).find((e) => e.claim_id === claimId);
    const copilotBox = el('div', { id: 'claim-copilot' });
    if (state.live) {
      const button = el('button', { class: 'primary', type: 'button', text: 'Ask the copilot what to check first' });
      button.addEventListener('click', async () => {
        button.disabled = true;
        button.textContent = 'Thinking…';
        try {
          copilotBox.replaceChildren(renderAnswer(await askCopilot('Why is this claim flagged, and what should the investigator check first?', claimId)));
        } catch (error) {
          copilotBox.replaceChildren(el('p', { class: 'muted', text: `Copilot unavailable: ${error.message}` }));
        }
      });
      copilotBox.append(button);
    } else if (recorded) {
      copilotBox.append(renderAnswer(recorded, true));
    } else {
      copilotBox.append(el('p', { class: 'muted', text: 'Serve this page with the API (fraudops serve) to ask the copilot about this claim.' }));
    }
    $('#claim-detail').replaceChildren(
      el('div', { class: 'detail-head' },
        el('div', {}, el('h2', { text: claim.claim_id }), el('span', { class: 'sub', text: `${label(claim.loss_type)} · ${claim.city}, ${claim.province} · reported ${claim.reported}` })),
        bandChip(claim.band)),
      el('div', { class: 'mini-kpis' },
        el('div', {}, el('dt', { text: 'Fraud probability' }), el('dd', { text: pct(claim.probability) })),
        el('div', {}, el('dt', { text: 'Expected saving' }), el('dd', { text: money(claim.expected_value) })),
        el('div', {}, el('dt', { text: 'Claimed' }), el('dd', { text: money(claim.total_claimed) })),
        el('div', {}, el('dt', { text: 'Queue rank' }), el('dd', { text: `#${claim.rank}` }))),
      el('div', {}, el('h3', { text: 'Why the model ranks it here' }),
        el('div', { class: 'reason-key' }, el('span', {}, el('i', { style: 'background:var(--pos)' }), 'raises risk'),
          el('span', {}, el('i', { style: 'background:var(--neg)' }), 'lowers risk')),
        el('div', { class: 'reasons', style: 'margin-top:8px' }, reasonRows(claim.reasons_for, 'up', max), reasonRows(claim.reasons_against, 'down', max))),
      claim.tags.length ? el('div', {}, el('h3', { text: 'Linked to' }),
        el('div', { class: 'links', style: 'margin-top:6px' }, claim.tags.map((t) => el('span', { class: 'chip', text: tagTitle(t) })))) : null,
      el('dl', { class: 'facts' }, facts.map(([k, v]) => el('div', {}, el('dt', { text: k }), el('dd', { text: v })))),
      el('div', {}, el('h3', { text: 'Adjuster notes', style: 'margin-bottom:6px' }), highlighted(claim.narrative, terms)),
      el('div', {}, el('h3', { text: 'Copilot', style: 'margin-bottom:6px' }), copilotBox),
    );
  }

  /* ---------- Patterns ----------------------------------------------------------------------------- */
  const UNIT = {
    tow_fee: (v) => moneyFull(v), storage_days: (v) => `${num(v, 0)} day${Math.round(v) === 1 ? '' : 's'}`, tow_distance_km: (v) => `${num(v, 0)} km`,
    log_total_claimed: (v) => money(Math.expm1(v)), log_injury_reserve: (v) => money(Math.expm1(v)),
    log_repair_estimate: (v) => money(Math.expm1(v)),
  };
  const unit = (feature, value) => (UNIT[feature] ? UNIT[feature](value) : num(value));

  function renderEmerging() {
    const box = $('#emerging');
    const patterns = DATA.segments.emerging || [];
    if (!patterns.length) {
      box.replaceChildren(el('div', { class: 'card' }, el('h2', { text: 'No emerging pattern this month' }),
        el('p', { class: 'muted', text: 'No kind of claim became markedly more common than in the last 12 months.' })));
      return;
    }
    box.replaceChildren(...patterns.map((p) => {
      const title = (TITLES[p.id] && TITLES[p.id].title) || 'New pattern';
      const kicker = el('div', { class: 'alert-kicker' }, statusBadge('alert').firstChild, `Emerging pattern ${p.id} · ${monthName(DATA.month)}`);
      const truth = p.hidden_truth_schemes || {};
      const topScheme = Object.entries(truth).sort((a, b) => b[1] - a[1])[0];
      const chartId = `chart-trend-${p.id}`;
      const providers = Object.entries(p.shared_providers).map(([kind, info]) => `${label(kind)} ${info.value} on ${pct(info.share)} of them`);
      const card = el('section', { class: 'card alert-card', 'aria-label': `Emerging pattern ${p.id}` },
        el('div', { class: 'alert-head' }, el('div', {}, kicker, el('h2', { class: 'alert-title', text: title }))),
        el('div', { class: 'alert-body' },
          el('div', {},
            el('dl', { class: 'mini-kpis' },
              el('div', {}, el('dt', { text: 'Claims this month' }), el('dd', { text: String(p.claims) })),
              el('div', {}, el('dt', { text: 'Claimed' }), el('dd', { text: money(p.total_claimed) })),
              el('div', {}, el('dt', { text: 'Avg. fraud probability' }), el('dd', { text: pct(p.avg_probability, 1) })),
              el('div', {}, el('dt', { text: 'In the SIU queue' }), el('dd', { text: String(p.in_queue) }))),
            el('table', { class: 'compare' },
              el('thead', {}, el('tr', {}, el('th', { text: 'What sets them apart' }), el('th', { text: 'Last 12 months' }), el('th', { text: 'These claims' }))),
              el('tbody', {}, p.distinctive.slice(0, 3).map((d) => el('tr', {}, el('td', { text: featureLabel(d.feature) }),
                el('td', { text: unit(d.feature, d.reference_mean) }), el('td', { text: unit(d.feature, d.current_mean) }))))),
            el('p', { class: 'action muted' }, el('strong', { text: 'Shared: ' }), providers.join('; ') || 'no single provider',
              '. ', el('strong', { text: 'Notes mention: ' }), p.top_terms.join(', '), '.'),
            el('p', { class: 'action' }, el('strong', { text: 'Why it matters: ' }),
              `the model gives these claims ${pct(p.avg_probability, 1)} on average, so only ${p.in_queue} reached the queue. It has never been trained on this kind of claim; the density check caught it instead. `,
              el('strong', { text: 'Next step: ' }), 'refer the group to the SIU lead as one pattern, and label the outcomes so the next model version learns it.'),
            topScheme ? el('p', { class: 'muted', style: 'margin-top:8px;font-size:12.5px' },
              `Hindsight check (synthetic data only): ${pct(topScheme[1] / p.claims)} of these claims belong to the generator's hidden "${topScheme[0].replace(/_/g, ' ')}" scheme.`) : null),
          el('figure', {}, el('figcaption', {}, el('h3', { text: p.trend ? `Claims involving ${p.trend.provider}, by month` : 'Trend' }),
            el('p', { text: 'The operator barely existed in the data until two months ago.' })),
            el('div', { class: 'chart', id: chartId, style: 'height:240px' }))));
      if (p.trend) {
        chart(chartId, (node) => {
          const last = p.trend.months.length - 1;
          Plotly.react(node, [{
            x: p.trend.months.map(monthName), y: p.trend.claims, type: 'bar', hovertemplate: '%{x}: <b>%{y}</b> claims<extra></extra>',
            marker: { color: p.trend.claims.map((_, i) => (i === last ? cssVar('--series-2') : cssVar('--deemph'))), cornerradius: 4 },
            text: p.trend.claims.map((v, i) => (i === last || (v > 0 && i >= last - 2) ? String(v) : '')), textposition: 'outside',
            textfont: { color: cssVar('--text-secondary'), size: 11.5 }, cliponaxis: false,
          }], layout({ bargap: 0.45, margin: { l: 36, r: 8, t: 16, b: 52 }, xaxis: { showgrid: false, tickangle: -45 },
            yaxis: { rangemode: 'tozero' } }), PLOT_CONFIG);
        });
      }
      return card;
    }));
  }

  function renderMap() {
    chart('chart-map', (node) => {
      const m = DATA.map;
      const idx = { ref: [], cur: [], emerging: [] };
      m.x.forEach((_, i) => {
        if (!m.current[i]) idx.ref.push(i);
        else if (m.emerging[i]) idx.emerging.push(i);
        else idx.cur.push(i);
      });
      const pick = (indices, key) => indices.map((i) => m[key][i]);
      const hover = (indices) => indices.map((i) => `${m.claim_id[i]} · ${label(m.loss_type[i])}<br>${moneyFull(m.total_claimed[i])} · fraud probability ${pct(m.probability[i])}`);
      const ring = cssVar('--surface-1');
      const traces = [
        { name: 'Last 12 months (sample)', indices: idx.ref, color: cssVar('--deemph'), size: 5, opacity: 0.55, line: 0 },
        { name: `${monthName(DATA.month)} claims`, indices: idx.cur, color: cssVar('--series-1'), size: 7, opacity: 0.9, line: 1 },
        { name: `Emerging pattern (${idx.emerging.length})`, indices: idx.emerging, color: cssVar('--series-2'), size: 10, opacity: 1, line: 2 },
      ].map((t) => ({
        x: pick(t.indices, 'x'), y: pick(t.indices, 'y'), text: hover(t.indices), name: t.name, type: 'scatter', mode: 'markers',
        hovertemplate: '%{text}<extra></extra>',
        marker: { size: t.size, color: t.color, opacity: t.opacity, line: { color: ring, width: t.line } },
      }));
      const annotations = (DATA.segments.segments || []).map((s) => {
        const xs = []; const ys = [];
        idx.ref.forEach((i) => { if (m.segment[i] === s.id) { xs.push(m.x[i]); ys.push(m.y[i]); } });
        xs.sort((a, b) => a - b); ys.sort((a, b) => a - b);
        const median = (arr) => arr[Math.floor(arr.length / 2)];
        return { x: median(xs), y: median(ys), text: `S${s.id}`, hovertext: s.name, showarrow: false,
          font: { size: 12, color: cssVar('--text-primary') }, bgcolor: cssVar('--surface-1'), borderpad: 3, opacity: 0.92 };
      });
      Plotly.react(node, traces, layout({
        showlegend: true, hovermode: 'closest', margin: { l: 8, r: 8, t: 30, b: 8 },
        legend: { orientation: 'h', x: 0, y: 1.06, font: { color: cssVar('--text-secondary'), size: 12 } },
        xaxis: { visible: false }, yaxis: { visible: false }, annotations,
      }), PLOT_CONFIG);
    });
  }

  function logTicks(values, money) {
    const lo = Math.min(...values);
    const hi = Math.max(...values);
    const candidates = [1, 2, 3, 5, 7, 10, 14, 20, 30, 50, 60, 100, 200, 300, 500, 1000, 2000, 3000, 5000, 10000, 20000, 30000, 50000, 100000];
    const inRange = candidates.filter((v) => v >= lo * 0.8 && v <= hi * 1.25);
    const step = Math.max(1, Math.ceil(inRange.length / 6));
    const tickvals = inRange.filter((_, i) => i % step === 0);
    return { tickvals, ticktext: tickvals.map((v) => (money ? money(v) : String(v))) };
  }

  function renderPatternView() {
    const view = DATA.segments.pattern_view;
    if (!view) {
      $('#chart-pattern-view').closest('figure').hidden = true;
      return;
    }
    const AXIS = { tow_fee: 'Towing and storage fees', storage_days: 'Days in storage', tow_distance_km: 'Tow distance (km)' };
    const xName = AXIS[view.x_feature] || featureLabel(view.x_feature);
    const yName = AXIS[view.y_feature] || featureLabel(view.y_feature);
    const pattern = (DATA.segments.emerging || []).find((p) => p.id === view.pattern);
    $('#pattern-view-title').textContent = `${view.pattern} on its two most distinctive features`;
    $('#pattern-view-caption').textContent = `${xName} against ${yName.toLowerCase()}, log scales; claims with a zero value are left out.`
      + (pattern ? ` ${pattern.claims} pattern claims sit far from last year's cloud.` : '');
    chart('chart-pattern-view', (node) => {
      const isMoney = (f) => f === 'tow_fee' || f.includes('claimed') || f.includes('estimate') || f.includes('reserve');
      const groups = [
        ['reference', 'Last 12 months (sample)', cssVar('--deemph'), 5, 0.6, 0],
        ['current', `${monthName(DATA.month)} claims`, cssVar('--series-1'), 7, 0.9, 1],
        ['pattern', `Pattern ${view.pattern}`, cssVar('--series-2'), 10, 1, 2],
      ];
      const traces = groups.map(([key, name, color, size, opacity, ring]) => {
        const idx = view.group.map((g, i) => (g === key ? i : -1)).filter((i) => i >= 0);
        return {
          x: idx.map((i) => view.x[i]), y: idx.map((i) => view.y[i]), type: 'scatter', mode: 'markers', name,
          text: idx.map((i) => `${view.claim_id[i]}<br>${xName}: ${isMoney(view.x_feature) ? moneyFull(view.x[i]) : view.x[i]}<br>${yName}: ${isMoney(view.y_feature) ? moneyFull(view.y[i]) : view.y[i]}`),
          hovertemplate: '%{text}<extra></extra>',
          marker: { size, color, opacity, line: { color: cssVar('--surface-1'), width: ring } },
        };
      });
      const xt = logTicks(view.x, isMoney(view.x_feature) ? money : null);
      const yt = logTicks(view.y, isMoney(view.y_feature) ? money : null);
      Plotly.react(node, traces, layout({
        showlegend: true, hovermode: 'closest', margin: { l: 56, r: 12, t: 64, b: 48 },
        legend: { orientation: 'h', x: 0, y: 1.02, yanchor: 'bottom', font: { color: cssVar('--text-secondary'), size: 12 } },
        xaxis: { type: 'log', title: { text: xName }, tickvals: xt.tickvals, ticktext: xt.ticktext },
        yaxis: { type: 'log', title: { text: yName }, tickvals: yt.tickvals, ticktext: yt.ticktext },
      }), PLOT_CONFIG);
    });
  }

  function renderSegments() {
    const segs = [...(DATA.segments.segments || [])].sort((a, b) => b.confirmed_fraud_rate - a.confirmed_fraud_rate);
    const maxRate = Math.max(...segs.map((s) => s.confirmed_fraud_rate), 0.01);
    $('#segments-table').replaceChildren(table([
      { label: 'Segment', render: (s) => el('span', {}, el('b', { text: `S${s.id} ` }), s.name) },
      { label: 'Fraud rate', render: (s) => el('div', { class: 'prob' }, el('b', { text: pct(s.confirmed_fraud_rate, 1) }), meter(s.confirmed_fraud_rate, maxRate)) },
      { label: 'Avg. claim', num: true, render: (s) => money(s.avg_claimed) },
      { label: 'Share now (last yr)', num: true, render: (s) => `${pct(s.share_current, 1)} (${pct(s.share_reference, 1)})` },
    ], segs));
  }

  function renderRings() {
    const nets = DATA.rings.networks || [];
    const rings = DATA.rings.rings || [];
    const picker = $('#ring-picker');
    if (!nets.length) { picker.replaceChildren(); $('#ring-facts').replaceChildren(el('p', { class: 'muted', text: 'No linked groups found.' })); return; }
    picker.replaceChildren(...nets.map((n, i) => {
      const b = el('button', { type: 'button', role: 'radio', 'aria-checked': String(i === state.ring), text: n.ring });
      b.addEventListener('click', () => { state.ring = i; renderRings(); });
      return b;
    }));
    const net = nets[state.ring];
    const ring = rings.find((r) => r.id === net.ring);
    chart('chart-ring', (node) => {
      const byId = Object.fromEntries(net.nodes.map((n) => [n.id, n]));
      const ex = []; const ey = [];
      for (const [a, b] of net.edges) { ex.push(byId[a].x, byId[b].x, null); ey.push(byId[a].y, byId[b].y, null); }
      const claims = net.nodes.filter((n) => n.kind === 'claim');
      const entities = net.nodes.filter((n) => n.kind !== 'claim');
      const ringColor = cssVar('--surface-1');
      const claimTrace = (subset, name, color) => ({
        x: subset.map((n) => n.x), y: subset.map((n) => n.y), type: 'scatter', mode: 'markers', name,
        text: subset.map((n) => `${n.id} · ${n.month}<br>${moneyFull(n.total_claimed)} · fraud probability ${pct(n.probability)}`),
        hovertemplate: '%{text}<extra></extra>',
        marker: { size: subset.map((n) => 8 + Math.sqrt(n.total_claimed / 4000)), color, line: { color: ringColor, width: 2 } },
      });
      Plotly.react(node, [
        { x: ex, y: ey, type: 'scatter', mode: 'lines', line: { color: cssVar('--axis-muted'), width: 1 }, opacity: 0.45, hoverinfo: 'skip', showlegend: false },
        { x: entities.map((n) => n.x), y: entities.map((n) => n.y), type: 'scatter', mode: 'markers', name: 'Shared phone or address',
          text: entities.map((n) => (n.kind === 'phone' ? 'Shared phone number' : 'Shared address')), hovertemplate: '%{text}<extra></extra>',
          marker: { size: 9, symbol: entities.map((n) => (n.kind === 'phone' ? 'diamond' : 'square')), color: cssVar('--axis-muted'), line: { color: ringColor, width: 2 } } },
        claimTrace(claims.filter((n) => !n.current), 'Earlier claims', cssVar('--series-1')),
        claimTrace(claims.filter((n) => n.current), `${monthName(DATA.month)} claims`, cssVar('--series-2')),
      ], layout({ showlegend: true, hovermode: 'closest', margin: { l: 8, r: 8, t: 28, b: 8 },
        legend: { orientation: 'h', x: 0, y: 1.08, font: { color: cssVar('--text-secondary'), size: 12 } },
        xaxis: { visible: false }, yaxis: { visible: false, scaleanchor: 'x' } }), PLOT_CONFIG);
    });
    if (ring) {
      const providers = Object.entries(ring.providers).map(([k, v]) => `${label(k)} ${v.value} (${pct(v.share)})`).join(', ');
      $('#ring-facts').replaceChildren(el('div', { class: 'ring-facts' },
        el('h3', { text: (TITLES[ring.id] && TITLES[ring.id].title) || ring.id, style: 'margin-bottom:10px' }),
        el('dl', {}, [
          ['Claims', String(ring.claims)], ['Active', `${ring.months_active} months (${ring.first_month} to ${ring.last_month})`],
          ['Claimed', moneyFull(ring.total_claimed)], ['Shared', `${ring.shared_phones} phone numbers, ${ring.shared_addresses} addresses`],
          ['Injured per claim', num(ring.avg_injured)], ['Cities', Object.keys(ring.cities).join(', ')],
          ['Recurring providers', providers || '—'], ['Avg. fraud probability', pct(ring.avg_probability)],
          [`In ${monthName(DATA.month)}`, `${ring.claims_current_month} new claim(s)`],
          ['Hindsight check', `${pct(ring.hidden_truth_fraud_rate)} fraudulent${ring.hidden_truth_ring ? ` (generator ${ring.hidden_truth_ring.value})` : ''}`],
        ].flatMap(([k, v]) => [el('dt', { text: k }), el('dd', { text: v })])),
        el('p', { class: 'muted', style: 'margin-top:12px;font-size:12.5px', text: 'The model already ranks these claims high: rings reuse providers and identifiers, which the network features capture. The graph view is for the investigator who has to map the group.' })));
    }
  }

  /* ---------- Monitoring --------------------------------------------------------------------------- */
  function renderMonitoring() {
    const drift = DATA.drift;
    const alerts = drift.features.filter((f) => f.status === 'alert');
    const rag = DATA.rag_eval.metrics || {};
    $('#health-kpis').replaceChildren(
      tile('Drift alerts this month', String(alerts.length), alerts.length ? [alerts.slice(0, 3).map((a) => featureLabel(a.feature)).join(', ')] : ['none'], null),
      tile('Score drift (PSI)', num(drift.score.psi, 3), [drift.score.psi < drift.thresholds.warn ? 'stable: the model is not reacting' : 'shifted'], null),
      tile('Serving model', `v${M.champion.version}`, [`${CANDIDATE_NAMES[M.champion.candidate] || M.champion.candidate}, ${M.decision.replace('_', ' ')}`], null),
      tile('Copilot retrieval hit@3', rag.hit_at_3 !== undefined ? pct(rag.hit_at_3) : '—', [`${DATA.rag_eval.rows.length} evaluation questions`], null),
    );

    chart('chart-tail', (node) => {
      const rows = drift.features.filter((f) => f.tail_ratio !== null && f.tail_ratio !== undefined).sort((a, b) => b.tail_ratio - a.tail_ratio).slice(0, 8).reverse();
      Plotly.react(node, [{
        x: rows.map((r) => r.tail_ratio), y: rows.map((r) => featureLabel(r.feature)), type: 'bar', orientation: 'h',
        marker: { color: cssVar('--series-1'), cornerradius: 4 },
        text: rows.map((r) => `${num(r.tail_ratio)}×`), textposition: 'outside', cliponaxis: false,
        textfont: { color: cssVar('--text-secondary'), size: 11.5 },
        customdata: rows.map((r) => [r.tail_claims, r.psi]),
        hovertemplate: '%{y}: <b>%{x:.1f}×</b> the usual rate<br>%{customdata[0]} claims above last year\'s 99th percentile, PSI %{customdata[1]}<extra></extra>',
      }], layout({ bargap: 0.45, margin: { l: 8, r: 40, t: 24, b: 36 }, xaxis: { title: { text: 'times the usual rate' }, rangemode: 'tozero' },
        yaxis: { showgrid: false },
        shapes: [1, 3].map((x) => ({ type: 'line', x0: x, x1: x, yref: 'paper', y0: 0, y1: 1, line: { color: cssVar('--axis'), width: 1 } })),
        annotations: [{ x: 3, y: 1, yref: 'paper', text: 'alert at 3×', showarrow: false, xanchor: 'left', xshift: 4, yanchor: 'bottom', font: { size: 11, color: cssVar('--text-secondary') } }],
      }), PLOT_CONFIG);
    });

    chart('chart-scores', (node) => {
      const bins = drift.score.bins;
      const names = bins.slice(0, -1).map((b, i) => `${Math.round(b * 100)}–${Math.round(bins[i + 1] * 100)}%`);
      const test = M.data.test.months;
      Plotly.react(node, [
        { x: names, y: drift.score.reference, name: `Test months (${monthName(test[0])}–${monthName(test[1])})`, type: 'bar',
          marker: { color: cssVar('--series-1'), cornerradius: 4 }, hovertemplate: '%{x}: <b>%{y:.1%}</b><extra>test months</extra>' },
        { x: names, y: drift.score.current, name: monthName(DATA.month), type: 'bar',
          marker: { color: cssVar('--series-2'), cornerradius: 4 }, hovertemplate: '%{x}: <b>%{y:.1%}</b><extra>this month</extra>' },
      ], layout({ showlegend: true, barmode: 'group', bargap: 0.35, bargroupgap: 0.12, margin: { l: 48, r: 8, t: 34, b: 44 },
        legend: { orientation: 'h', x: 0, y: 1.16, font: { color: cssVar('--text-secondary'), size: 12 } },
        xaxis: { title: { text: 'Fraud probability' }, showgrid: false }, yaxis: { tickformat: '.0%' } }), PLOT_CONFIG);
    });

    const shown = drift.features.slice(0, 12);
    $('#drift-table').replaceChildren(table([
      { label: 'Feature', render: (r) => featureLabel(r.feature) },
      { label: 'Status', render: (r) => statusBadge(r.status) },
      { label: 'PSI', num: true, render: (r) => num(r.psi, 3) },
      { label: 'Mean shift (sd)', num: true, render: (r) => (r.mean_shift_sd === null ? '—' : num(r.mean_shift_sd, 2)) },
      { label: 'Extreme-value rate', num: true, render: (r) => (r.tail_ratio === null ? '—' : `${num(r.tail_ratio)}× (${r.tail_claims})`) },
      { label: 'Last 12 months', num: true, render: (r) => (r.reference_mean === null ? '—' : num(r.reference_mean, 2)) },
      { label: 'This month', num: true, render: (r) => (r.current_mean === null ? '—' : num(r.current_mean, 2)) },
    ], shown));

    chart('chart-pr', (node) => {
      const curves = M.pr_curve;
      const names = { lightgbm: 'LightGBM (champion)', logreg: 'Logistic regression' };
      const colors = { lightgbm: cssVar('--series-1'), logreg: cssVar('--series-3') };
      const traces = Object.keys(curves).map((key) => ({
        x: curves[key].recall, y: curves[key].precision, name: names[key] || key, type: 'scatter', mode: 'lines',
        line: { color: colors[key] || cssVar('--series-4'), width: 2 }, hovertemplate: `recall %{x:.0%} · precision %{y:.0%}<extra>${names[key] || key}</extra>`,
      }));
      traces.push({
        x: [RULES.frauds_caught / RULES.frauds_total], y: [RULES.hit_rate], name: 'Scorecard at 20 files / month', type: 'scatter', mode: 'markers',
        marker: { size: 11, color: cssVar('--series-2'), line: { color: cssVar('--surface-1'), width: 2 } },
        hovertemplate: 'Scorecard: recall %{x:.0%} · precision %{y:.0%}<extra></extra>',
      });
      traces.push({
        x: [CHAMP.frauds_caught / CHAMP.frauds_total], y: [CHAMP.hit_rate], name: 'Model at 20 files / month', type: 'scatter', mode: 'markers',
        marker: { size: 11, color: cssVar('--series-1'), line: { color: cssVar('--surface-1'), width: 2 } },
        hovertemplate: 'Model queue: recall %{x:.0%} · precision %{y:.0%}<extra></extra>',
      });
      Plotly.react(node, traces, layout({ showlegend: true, hovermode: 'closest', margin: { l: 52, r: 12, t: 50, b: 44 },
        legend: { orientation: 'h', x: 0, y: 1.24, font: { color: cssVar('--text-secondary'), size: 12 } },
        xaxis: { title: { text: 'Recall (share of frauds found)' }, tickformat: '.0%', range: [0, 1] },
        yaxis: { title: { text: 'Precision' }, tickformat: '.0%', range: [0, 1.02] } }), PLOT_CONFIG);
    });

    chart('chart-importance', (node) => {
      const entries = Object.entries(M.importance).slice(0, 10).reverse();
      Plotly.react(node, [{
        x: entries.map((e) => e[1]), y: entries.map((e) => featureLabel(e[0])), type: 'bar', orientation: 'h',
        marker: { color: cssVar('--series-1'), cornerradius: 4 }, hovertemplate: '%{y}: <b>%{x:.3f}</b><extra></extra>',
      }], layout({ bargap: 0.42, margin: { l: 8, r: 16, t: 8, b: 36 }, xaxis: { title: { text: 'mean |SHAP| (log-odds)' } }, yaxis: { showgrid: false } }), PLOT_CONFIG);
    });

    const cands = M.candidates;
    const names = CANDIDATE_NAMES;
    $('#candidates-table').replaceChildren(table([
      { label: 'Candidate', render: (r) => el('span', {}, names[r.key] || r.key, r.key === M.champion.candidate ? el('span', { class: 'chip', style: 'margin-left:6px', text: 'champion' }) : null) },
      { label: 'ROC-AUC', num: true, render: (r) => num(r.test.roc_auc, 3) },
      { label: 'PR-AUC', num: true, render: (r) => num(r.test.pr_auc, 3) },
      { label: 'Hit rate', num: true, render: (r) => pct(r.test.hit_rate) },
      { label: 'Net / month (test)', num: true, render: (r) => money(r.test.net_savings) },
      { label: 'Net / month (valid)', num: true, render: (r) => money(r.valid.net_savings) },
    ], Object.keys(cands).map((key) => ({ key, ...cands[key] }))));
    const prev = M.previous_champion;
    const kb = DATA.kb || {};
    $('#lineage').replaceChildren(...[
      ['Registered model', `${M.champion.model_name} v${M.champion.version} (alias: champion)`],
      ['MLflow run', M.champion.mlflow_run_id],
      ['Trained', M.champion.trained_at.replace('T', ' ').replace('+00:00', ' UTC')],
      ['Training months', `${M.champion.train_months[0]} to ${M.champion.train_months[1]}`],
      ['Selection', 'highest validation net savings at capacity'],
      ['Promotion gate', prev ? `${M.decision.replace('_', ' ')} (previous v${prev.version}: ${money(prev.net_savings)}/month)` : `${M.decision.replace('_', ' ')} (first champion)`],
      ['Knowledge base', `${kb.documents || '?'} documents, ${kb.chunks || '?'} chunks, ${kb.embed_model || 'keyword only'}`],
    ].flatMap(([k, v]) => [el('dt', { text: k }), el('dd', { text: v })]));
  }

  /* ---------- Copilot ------------------------------------------------------------------------------ */
  function renderCopilot() {
    const rag = DATA.rag_eval.metrics || {};
    $('#rag-kpis').replaceChildren(
      tile('Retrieval hit@3', rag.hit_at_3 !== undefined ? pct(rag.hit_at_3) : '—', ['right document in the top 3']),
      tile('Out-of-scope declined', rag.out_of_scope_declined !== undefined ? pct(rag.out_of_scope_declined) : '—', ['instead of a made-up answer']),
      tile('Expected facts in answers', rag.term_recall !== undefined ? pct(rag.term_recall) : '—', ['term recall, local 3B model']),
      tile('Median answer time', rag.latency_p50_ms !== undefined ? `${(rag.latency_p50_ms / 1000).toFixed(1)} s` : '—', ['on a laptop CPU']),
    );
    const examples = DATA.copilot_examples || [];
    $('#recorded-note').textContent = examples.length
      ? `Recorded answers from the local model (${examples[0].model || 'local model'}), captured by the evaluation run. Serve the page with the API to ask your own questions.`
      : 'No recorded answers: the last evaluation ran without a local model.';
    $('#recorded').replaceChildren(...examples.map((ex) => el('div', { class: 'card', style: 'margin-top:12px' },
      el('h3', { text: ex.claim_id ? `${ex.question} (${ex.claim_id})` : ex.question, style: 'margin-bottom:10px' }), renderAnswer(ex, true))));
  }

  function setupLiveCopilot() {
    const select = $('#ask-claim');
    for (const claim of DATA.queue.slice(0, 20)) select.append(el('option', { value: claim.claim_id, text: `${claim.claim_id} · #${claim.rank} ${label(claim.loss_type)}` }));
    const questions = ['What are the red flags of a staged collision?', 'A storage yard is demanding a release fee. What should the adjuster do?',
      'What is the emerging pattern this month?', 'Can we deny a claim because the model score is high?', 'Does the model use the claimant\'s age?'];
    $('#ask-examples').replaceChildren(...questions.map((q) => {
      const b = el('button', { type: 'button', text: q });
      b.addEventListener('click', () => { $('#ask-input').value = q; $('#ask-form').requestSubmit(); });
      return b;
    }));
    $('#ask-form').addEventListener('submit', async (event) => {
      event.preventDefault();
      const button = $('#ask-form button');
      const out = $('#ask-answer');
      button.disabled = true;
      out.replaceChildren(el('p', { class: 'muted', style: 'margin-top:12px', text: 'Retrieving and asking the local model…' }));
      try {
        out.replaceChildren(el('div', { style: 'margin-top:14px' }, renderAnswer(await askCopilot($('#ask-input').value, select.value))));
      } catch (error) {
        out.replaceChildren(el('p', { class: 'muted', text: `Copilot error: ${error.message}` }));
      } finally {
        button.disabled = false;
      }
    });
  }

  /* ---------- Live mode ---------------------------------------------------------------------------- */
  const SCENARIOS = {
    towing: { label: 'Long tow to a distant yard', claim: { loss_type: 'rear_end', city: 'Mississauga', province: 'ON', loss_hour: 17,
      report_delay_days: 0, policy_tenure_days: 1460, vehicle_value: 31000, vehicle_age: 5, n_occupants: 1, n_injured: 0,
      soft_tissue_only: false, injury_reserve: 0, police_report: true, witnesses: 1, prior_claims_3y: 0, body_shop_id: 'SHOP-006',
      tow_company_id: 'TOW-25', tow_distance_km: 72, storage_days: 31, tow_fee: 5300, repair_estimate: 4200,
      narrative: 'Rear-end collision on Highway 403. Tow truck arrived at the scene before police were called. Vehicle towed by TOW-25 to a storage yard in Barrie. Storage charges are accruing and a release fee was demanded.' } },
    staged: { label: 'Multi-injury rear-end', claim: { loss_type: 'rear_end', city: 'Calgary', province: 'AB', loss_hour: 21,
      report_delay_days: 1, policy_tenure_days: 60, vehicle_value: 18000, vehicle_age: 9, n_occupants: 4, n_injured: 4,
      soft_tissue_only: true, injury_reserve: 64000, police_report: false, witnesses: 0, prior_claims_3y: 1, body_shop_id: 'SHOP-021',
      clinic_id: 'CLINIC-13', rep_id: 'REP-07', tow_distance_km: 0, storage_days: 0, tow_fee: 0, repair_estimate: 3900,
      narrative: 'Insured states the vehicle ahead stopped suddenly on Deerfoot Trail. All 4 occupants reported neck and back pain after the collision. Occupants were referred to CLINIC-13 and treatment plans were submitted within days. Claimants retained REP-07 shortly after the loss. No police attendance; reported at a collision reporting centre.' } },
    glass: { label: 'Windshield chip', claim: { loss_type: 'glass', city: 'Ottawa', province: 'ON', loss_hour: 9, report_delay_days: 1,
      policy_tenure_days: 2900, vehicle_value: 27000, vehicle_age: 4, n_occupants: 1, n_injured: 0, soft_tissue_only: false,
      injury_reserve: 0, police_report: false, witnesses: 0, prior_claims_3y: 0, tow_distance_km: 0, storage_days: 0, tow_fee: 0,
      repair_estimate: 640, narrative: 'Windshield cracked by a stone on Highway 417; replacement requested. Photos received.' } },
  };
  const EDITABLE = [['policy_tenure_days', 'Policy age (days)'], ['n_injured', 'People injured'], ['injury_reserve', 'Injury reserve ($)'],
    ['tow_fee', 'Towing & storage ($)'], ['storage_days', 'Storage days'], ['repair_estimate', 'Repair estimate ($)']];

  function setupScoreForm() {
    const form = $('#score-form');
    const scenario = el('select', { id: 'scenario', name: 'scenario' }, Object.entries(SCENARIOS).map(([k, s]) => el('option', { value: k, text: s.label })));
    const inputs = {};
    form.replaceChildren(el('label', { class: 'full' }, 'Scenario', scenario),
      ...EDITABLE.map(([key, text]) => { inputs[key] = el('input', { type: 'number', min: '0', step: 'any', name: key, required: true }); return el('label', {}, text, inputs[key]); }),
      el('label', { class: 'full' }, 'Adjuster notes', inputs.narrative = el('textarea', { name: 'narrative', maxlength: '4000', required: true })),
      el('div', { class: 'full' }, el('button', { class: 'primary', type: 'submit', text: 'Score this claim' })));
    const fill = () => {
      const c = SCENARIOS[scenario.value].claim;
      for (const [key] of EDITABLE) inputs[key].value = c[key];
      inputs.narrative.value = c.narrative;
    };
    scenario.addEventListener('change', fill);
    fill();
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const base = { ...SCENARIOS[scenario.value].claim };
      for (const [key] of EDITABLE) base[key] = Number(inputs[key].value);
      base.narrative = inputs.narrative.value;
      base.n_occupants = Math.max(base.n_occupants, base.n_injured);
      const claim = { ...base, report_ts: new Date().toISOString().slice(0, 19), phone_id: `PH-${Date.now() % 10000000}`, address_id: `AD-${Date.now() % 1000000}` };
      const out = $('#score-result');
      try {
        const response = await fetch('api/score', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(claim) });
        const body = await response.json();
        if (!response.ok) throw new Error(JSON.stringify(body.detail));
        out.replaceChildren(el('div', { class: 'card', style: 'margin-top:12px;background:var(--surface-2)' },
          el('div', { class: 'detail-head' }, el('h3', { text: body.claim_id }), bandChip(body.band)),
          el('dl', { class: 'mini-kpis', style: 'margin-top:10px' },
            el('div', {}, el('dt', { text: 'Fraud probability' }), el('dd', { text: pct(body.probability, 1) })),
            el('div', {}, el('dt', { text: 'Expected saving' }), el('dd', { text: money(body.expected_value) })),
            el('div', {}, el('dt', { text: 'Queue cut-off' }), el('dd', { text: money(body.queue_cutoff) })),
            el('div', {}, el('dt', { text: 'Latency' }), el('dd', { text: `${Math.round(body.latency_ms)} ms` }))),
          el('div', { class: 'reasons', style: 'margin-top:12px' }, reasonRows(body.reasons_for, 'up', Math.max(...body.reasons_for.map((r) => r.weight), 0.01)))));
      } catch (error) {
        out.replaceChildren(el('p', { class: 'muted', text: `Scoring failed: ${error.message}` }));
      }
    });
  }

  async function pollLive() {
    try {
      const response = await fetch('api/live?limit=40');
      const body = await response.json();
      const feed = $('#live-feed');
      if (body.events.length) {
        feed.replaceChildren(...body.events.map((e) => el('div', { class: 'feed-row' },
          el('span', { class: 'claim-id', text: e.claim_id }),
          el('span', {}, `${label(e.loss_type)} · ${e.city}`, el('span', { class: 'sub', text: `${money(e.total_claimed)} · ${pct(e.probability, 1)} · ${e.received_at.slice(11, 19)} UTC` })),
          bandChip(e.band))));
      }
    } catch (_) { /* server went away: keep the last frame */ }
    setTimeout(pollLive, 3000);
  }

  async function detectLive() {
    // Static hosting has no API behind it: skip the probe rather than log a 404 for every visitor.
    if (location.protocol === 'file:' || location.hostname.endsWith('.github.io')) return;
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 2500);
      const response = await fetch('api/health', { signal: controller.signal });
      clearTimeout(timer);
      if (!response.ok) return;
      state.health = await response.json();
      state.live = true;
    } catch (_) {
      return;
    }
    const chip = $('#mode-chip');
    const ready = state.health.copilot && state.health.copilot.ready;
    chip.textContent = `Live · model v${state.health.model.version}${ready ? ' · local LLM ready' : ' · LLM offline'}`;
    chip.classList.add('live');
    for (const node of $$('.live-only')) node.hidden = false;
    setupLiveCopilot();
    setupScoreForm();
    pollLive();
    if (state.selected) selectClaim(state.selected);
  }

  /* ---------- How it works ------------------------------------------------------------------------ */
  function renderStack() {
    const steps = [
      ['Data', 'Python generator → Parquet', '16k synthetic claims, seven fraud mechanisms, one scheme that only appears in the newest month.'],
      ['Features', 'DuckDB SQL', 'One query file, used by batch training and by the API for each new claim. Look-back windows only.'],
      ['Training', 'LightGBM + TF-IDF', 'Narrative model stacked into a tabular model; compared with logistic regression and the rules.'],
      ['Tracking', 'MLflow', 'Every run, metric and model logged; the winner is registered and only becomes champion if it beats the last one.'],
      ['Serving', 'FastAPI + Docker', 'Scoring with reason codes, read-only non-root container, Prometheus metrics, this dashboard.'],
      ['Monitoring', 'PSI + tail checks, Prometheus/Grafana', 'Feature and score drift each month; service metrics scraped live.'],
      ['Patterns', 'KMeans, density ratio, graph', 'Segments, kinds of claims suddenly more common, and groups linked by shared phones or addresses.'],
      ['Copilot', 'LM Studio + DuckDB vectors', 'Hybrid retrieval over policies and this run\'s briefs; local 3B model; evaluated in MLflow.'],
    ];
    $('#flow').replaceChildren(...steps.map(([title, tool, text], i) => el('div', { class: 'flow-step' },
      el('span', { class: 'num', text: String(i + 1).padStart(2, '0') }), el('h3', { text: title }), el('div', { class: 'tool', text: tool }), el('p', { text: text }))));
    const rows = [
      ['Data', 'Kafka, Spark, Flink, Redis', 'Parquet + DuckDB; API replay instead of a broker; DuckDB table as online store', 'Single node handles millions of rows; same SQL ports to Spark', 'Event Hubs / Kafka → Databricks (Delta) → Redis or feature store'],
      ['Experiment tracking', 'MLflow', 'MLflow (SQLite backend) with model registry and aliases', 'Same tool, zero infrastructure', 'Databricks managed MLflow or Azure ML'],
      ['Orchestration', 'Airflow, Kubeflow', 'fraudops all (one CLI command) + GitHub Actions schedule', 'Eight steps do not need a scheduler cluster', 'Databricks Workflows, ADF or Airflow'],
      ['Deployment', 'FastAPI, Docker, Kubernetes', 'FastAPI + Docker Compose (hardened container)', 'Kubernetes adds nothing for one service', 'AKS or Azure Container Apps'],
      ['Monitoring', 'Prometheus, Grafana', 'Prometheus + Grafana (compose profile), PSI and tail-ratio drift', 'Same tools; drift checks in the pipeline', 'Azure Monitor / Managed Grafana + scheduled drift jobs'],
      ['Cloud platform', 'SageMaker, Azure ML, Vertex AI, Databricks', 'None: runs on a laptop, $0', 'A PoC should prove value before it costs money', 'Azure ML or Databricks'],
      ['CI/CD', 'Git, GitHub Actions', 'GitHub Actions: lint, tests, pipeline smoke run, image build, Pages deploy', 'Free for public repositories', 'Same, plus environments and approvals'],
      ['LLMOps', 'vLLM, vector database, LangSmith', 'LM Studio (OpenAI-compatible), DuckDB vector search + BM25, evaluation in MLflow', 'Local model keeps claim data on the machine', 'vLLM or Azure OpenAI, Azure AI Search or pgvector, MLflow tracing / LangSmith'],
    ];
    $('#stack-table').replaceChildren(table([
      { label: 'Stage', key: 0 }, { label: 'Reference stack', key: 1 }, { label: 'This PoC', key: 2 }, { label: 'Why', key: 3 }, { label: 'Production path', key: 4 },
    ], rows.map((r) => ({ 0: r[0], 1: r[1], 2: r[2], 3: r[3], 4: r[4] }))));
  }

  /* ---------- boot ----------------------------------------------------------------------------------- */
  renderOverview();
  renderQueue();
  renderEmerging();
  renderMap();
  renderPatternView();
  renderSegments();
  renderRings();
  renderMonitoring();
  renderCopilot();
  renderStack();
  showTab((location.hash || '#overview').slice(1), false);
  window.addEventListener('hashchange', () => showTab(location.hash.slice(1), false));
  detectLive();
})();
