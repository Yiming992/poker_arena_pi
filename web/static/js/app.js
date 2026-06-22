// Poker Arena web client.
(() => {
  const SUIT = { h: "\u2665", d: "\u2666", c: "\u2663", s: "\u2660" };
  const RED = new Set(["h", "d"]);

  let ws = null;
  let state = null;
  let mode = "observer"; // observer | queued | player
  let lastReasoning = {}; // player_id -> {name, action, reasoning}
  let timerInterval = null;

  const $ = (id) => document.getElementById(id);

  // ---------------- WebSocket ----------------
  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.onopen = () => setConn(true);
    ws.onclose = () => { setConn(false); setTimeout(connect, 1500); };
    ws.onmessage = (e) => handle(JSON.parse(e.data));
  }

  function send(obj) {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
  }

  function setConn(ok) {
    $("conn-dot").className = "dot " + (ok ? "online" : "offline");
    $("conn-text").textContent = ok ? "connected" : "reconnecting…";
  }

  function handle(msg) {
    switch (msg.type) {
      case "game_state": state = msg; render(); break;
      case "notice": notice(msg.text); break;
      case "join_queued": mode = "queued"; notice("You're seated for the next hand."); updateControls(); break;
      case "join_rejected": notice("Join rejected: " + (msg.reason || "")); mode = "observer"; updateControls(); break;
      case "join_cancelled": mode = "observer"; notice("Join cancelled."); updateControls(); break;
      case "left": mode = "observer"; notice("You left the table."); updateControls(); break;
      case "session_complete": showStandings(msg.standings); break;
    }
  }

  function showStandings(standings) {
    const body = $("standings-body");
    body.innerHTML = "";
    (standings || []).forEach((s, i) => {
      const row = document.createElement("div");
      row.className = "standings-row";
      row.innerHTML =
        `<span class='rank'>#${i + 1} ${escapeHtml(s.name)}</span>` +
        `<span>${s.stack} chips · ${s.hands_won} hands won</span>`;
      body.appendChild(row);
    });
    $("standings-modal").classList.remove("hidden");
  }

  // ---------------- Rendering ----------------
  const SEAT_POS = {
    // seat index (0-based among seated players) -> {x%, y%} around the oval
  };
  function seatLayout(n) {
    // Evenly distribute n seats around the oval; seat 0 at bottom center.
    const pos = [];
    for (let i = 0; i < n; i++) {
      const angle = Math.PI / 2 + (2 * Math.PI * i) / n; // start bottom
      const x = 50 + 46 * Math.cos(angle);
      const y = 50 + 44 * Math.sin(angle);
      pos.push({ x, y });
    }
    return pos;
  }

  function cardEl(card, small) {
    const d = document.createElement("div");
    d.className = "card" + (small ? " small" : "");
    if (!card) { d.classList.add("back"); d.innerHTML = "<div class='r'>?</div>"; return d; }
    if (RED.has(card.suit)) d.classList.add("red");
    d.innerHTML = `<div class='r'>${card.rank}</div><div class='s'>${SUIT[card.suit]}</div>`;
    return d;
  }

  function render() {
    if (!state) return;
    // The server's view is authoritative: a "player" view means we're seated.
    if (state.view === "player") mode = "player";
    else if (mode !== "queued") mode = "observer";
    $("view-mode").textContent = state.view;
    $("hand-info").textContent = state.hand_in_progress
      ? `Hand #${state.hand_number} · ${state.stage}`
      : (state.hand_number ? `Hand #${state.hand_number} done` : "waiting");
    $("pot").textContent = "Pot: " + state.pot;
    $("stage-badge").textContent = state.stage || "—";

    renderCommunity();
    renderSeats();
    renderReasoning();
    renderHistory();
    renderResult();
    updateControls();
    if (state.view === "player" && state.your_turn) showPlayerControls();
    else hidePlayerControls();
  }

  function renderCommunity() {
    const c = $("community");
    c.innerHTML = "";
    (state.community_cards || []).forEach((card) => c.appendChild(cardEl(card, false)));
  }

  function holeCardsFor(p) {
    if (state.view === "observer") {
      const hc = (state.all_hole_cards || {})[p.player_id];
      return hc ? hc : (p.is_active && !p.has_folded ? [null, null] : null);
    }
    // player view
    if (p.player_id && state.players) {
      const me = state.players.find((x) => x.is_human);
      if (me && p.player_id === me.player_id) return state.your_hole_cards || null;
    }
    const opp = (state.opponent_hole_cards || {})[p.player_id];
    if (opp) return opp;
    return (p.is_active && !p.has_folded) ? [null, null] : null;
  }

  function renderSeats() {
    const table = $("table");
    // remove old seats
    [...table.querySelectorAll(".seat")].forEach((s) => s.remove());
    const players = state.players || [];
    const pos = seatLayout(players.length);
    players.forEach((p, i) => {
      const seat = document.createElement("div");
      seat.className = "seat";
      if (p.player_id === state.current_player_id) seat.classList.add("active");
      if (p.has_folded) seat.classList.add("folded");
      seat.style.left = pos[i].x + "%";
      seat.style.top = pos[i].y + "%";

      const hole = holeCardsFor(p);
      const holeHtml = document.createElement("div");
      holeHtml.className = "hole";
      if (hole) hole.forEach((c) => holeHtml.appendChild(cardEl(c, true)));

      const badges = [];
      if (p.seat === state.dealer_seat) badges.push("<span class='badge dealer'>D</span>");
      if (p.is_human) badges.push("<span class='badge human'>YOU</span>");
      if (p.benched) badges.push("<span class='badge bench'>BENCH</span>");
      if (p.is_all_in) badges.push("<span class='badge allin'>ALL-IN</span>");

      const plate = document.createElement("div");
      plate.className = "nameplate";
      plate.innerHTML =
        `<div class='pname'>${escapeHtml(p.name)} ${badges.join(" ")}</div>` +
        (p.model ? `<div class='pmodel'>${escapeHtml(p.model)}</div>` : `<div class='pmodel'>${p.agent_type}</div>`) +
        `<div class='pstack'>${p.stack}</div>`;

      seat.appendChild(plate);
      seat.appendChild(holeHtml);
      if (p.current_bet > 0) {
        const chip = document.createElement("div");
        chip.className = "betchip";
        chip.textContent = p.current_bet;
        seat.appendChild(chip);
      }
      table.appendChild(seat);
    });
  }

  function renderReasoning() {
    const hist = state.action_history || [];
    // Build latest reasoning per player from history (current hand).
    const latest = {};
    hist.forEach((r) => {
      if (r.reasoning) latest[r.player] = { name: r.player, action: r.action, amount: r.amount, reasoning: r.reasoning };
    });
    const list = $("reasoning-list");
    list.innerHTML = "";
    const players = state.players || [];
    players.forEach((p) => {
      if (p.is_human) return;
      const r = latest[p.name];
      const card = document.createElement("div");
      card.className = "reason-card" + (r ? "" : " empty");
      const actLabel = r ? `${r.action}${r.amount ? " " + r.amount : ""}` : "";
      card.innerHTML =
        `<div class='rc-head'><span class='rc-name'>${escapeHtml(p.name)}</span>` +
        `<span class='rc-act'>${actLabel}</span></div>` +
        `<div class='rc-body'>${r ? escapeHtml(r.reasoning) : "waiting to act…"}</div>`;
      list.appendChild(card);
    });
  }

  function renderHistory() {
    const list = $("history-list");
    const hist = state.action_history || [];
    list.innerHTML = "";
    hist.slice().reverse().forEach((r) => {
      const row = document.createElement("div");
      row.className = "hist-row";
      const amt = r.amount ? ` ${r.amount}` : "";
      row.innerHTML = `<span class='hh'>#${r.hand_number}</span> <span class='hp'>${escapeHtml(r.player)}</span> <span class='ha'>${r.action}${amt}</span> <span>(${r.stage})</span>`;
      list.appendChild(row);
    });
  }

  function renderResult() {
    const banner = $("result-banner");
    const res = state.last_result;
    if (res && !state.hand_in_progress) {
      const names = {};
      (state.players || []).forEach((p) => (names[p.player_id] = p.name));
      const winners = res.winners.map((w) => names[w] || w).join(", ");
      const total = Object.values(res.pot_awards).reduce((a, b) => a + b, 0);
      banner.textContent = `${winners} wins ${total}` + (res.showdown ? " at showdown" : "");
      banner.classList.remove("hidden");
    } else {
      banner.classList.add("hidden");
    }
  }

  // ---------------- Controls ----------------
  function updateControls() {
    const join = $("join-btn");
    const leave = $("leave-btn");
    if (mode === "observer") {
      join.classList.remove("hidden"); join.textContent = "Join Next Hand"; join.disabled = false;
      leave.classList.add("hidden");
    } else if (mode === "queued") {
      join.classList.remove("hidden"); join.textContent = "Cancel Join"; join.disabled = false;
      leave.classList.add("hidden");
    } else if (mode === "player") {
      join.classList.add("hidden");
      leave.classList.remove("hidden");
    }
  }

  function showPlayerControls() {
    const pc = $("player-controls");
    pc.classList.remove("hidden");
    const valid = state.valid_actions || [];
    pc.querySelectorAll(".act").forEach((b) => {
      const act = b.dataset.act;
      const ok = valid.includes(act);
      b.disabled = !ok;
      if (act === "call") b.textContent = ok ? `Call ${state.call_amount}` : "Call";
    });
    const slider = $("raise-slider");
    const amount = $("raise-amount");
    const canRaise = valid.includes("raise");
    if (canRaise) {
      slider.min = state.min_raise; slider.max = state.max_raise;
      slider.value = state.min_raise; amount.value = state.min_raise;
      $("raise-hint").textContent = `min ${state.min_raise} · max ${state.max_raise}`;
    } else {
      $("raise-hint").textContent = "";
    }
    startTimer();
  }

  function hidePlayerControls() {
    $("player-controls").classList.add("hidden");
    stopTimer();
  }

  function startTimer() {
    stopTimer();
    let t = 60;
    $("turn-timer").textContent = `Your turn — ${t}s`;
    timerInterval = setInterval(() => {
      t--;
      $("turn-timer").textContent = t > 0 ? `Your turn — ${t}s` : "Time's up…";
      if (t <= 0) stopTimer();
    }, 1000);
  }
  function stopTimer() { if (timerInterval) { clearInterval(timerInterval); timerInterval = null; } }

  // ---------------- Notices ----------------
  function notice(text) {
    const stack = $("notice-stack");
    const n = document.createElement("div");
    n.className = "notice";
    n.textContent = text;
    stack.appendChild(n);
    setTimeout(() => n.remove(), 6000);
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  // ---------------- Events ----------------
  $("join-btn").addEventListener("click", () => {
    if (mode === "observer") send({ type: "join_request" });
    else if (mode === "queued") { send({ type: "cancel_join" }); }
  });
  $("leave-btn").addEventListener("click", () => send({ type: "leave_request" }));
  $("restart-btn").addEventListener("click", () => {
    if (confirm("Restart the session with current config?")) send({ type: "restart" });
  });
  $("speed-select").addEventListener("change", (e) => send({ type: "set_speed", speed: e.target.value }));

  document.querySelectorAll(".act").forEach((b) => {
    b.addEventListener("click", () => {
      const act = b.dataset.act;
      let amount = 0;
      if (act === "raise") amount = parseInt($("raise-amount").value || "0", 10);
      send({ type: "player_action", action: act, amount });
      hidePlayerControls();
    });
  });
  $("raise-slider").addEventListener("input", (e) => { $("raise-amount").value = e.target.value; });
  $("raise-amount").addEventListener("input", (e) => { $("raise-slider").value = e.target.value; });
  $("standings-close").addEventListener("click", () => $("standings-modal").classList.add("hidden"));

  connect();
})();
