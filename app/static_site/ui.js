// The page: reads the form, asks predictor.js for an estimate on every edit,
// and draws the location map (listing density from the training data, since
// the published page cannot load map tiles from other sites).
"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const form = $("form");
  const azn = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
  const money = (v) => `${azn.format(v)} AZN`;
  const pct = (v) => `${(100 * v).toFixed(1)}%`;
  const HOUSE = "heyet evi/bag evi";
  const DEFAULT_LOCATION = "28 may m.";
  const PAKO = "https://cdnjs.cloudflare.com/ajax/libs/pako/2.1.0/pako_inflate.min.js";

  let est = null;
  let pin = null;                // {lat, lng} the user tapped, or null
  let edited = false;            // false while the form still holds the example listing
  let timer = 0;
  let explainFailed = false;

  // ------------------------------------------------------------------ form
  function syncCategory() {
    const house = form.category.value === HOUSE;
    $("land").hidden = !house;
    $("floors").hidden = house;
  }

  function readForm() {
    const num = (id) => ($(id).value.trim() === "" ? null : Number($(id).value));
    const house = form.category.value === HOUSE;
    return {
      category: form.category.value,
      area_m2: num("area_m2"),
      rooms: num("rooms"),
      floor: house ? null : num("floor"),
      total_floors: house ? null : num("total_floors"),
      land_area_sot: house ? num("land_area_sot") : null,
      location: $("location").value || null,
      lat: pin ? pin.lat : null,
      lng: pin ? pin.lng : null,
      repair: $("repair").checked,
      mortgage: $("mortgage").checked,
      bill_of_sale: $("bill_of_sale").checked,
      description: $("description").value.trim(),
    };
  }

  function fieldsValid() {
    let ok = true;
    for (const el of form.querySelectorAll("input[type=number]")) {
      if (el.closest("[hidden]")) continue;
      const bad = !el.checkValidity();
      el.setAttribute("aria-invalid", String(bad));
      ok = ok && !bad;
    }
    return ok;
  }

  function showError(msg) {
    $("error").textContent = msg;
    $("error").hidden = !msg;
  }

  function estimate() {
    if (!est) return;
    if (!fieldsValid()) {
      showError("Check the highlighted fields: area 10–3000 m², rooms 1–20, floors −2 to 60.");
      return;
    }
    const item = readForm();
    const errs = est.validate(item);
    if (errs.length) { showError(errs.map(friendly).join(" ")); return; }
    showError("");
    renderResult(est.predict(item));
  }

  function friendly(e) {
    if (e.startsWith("area_m2: required")) return "Enter the area in m².";
    if (e.startsWith("floor cannot")) return "The floor can't be above the number of floors in the building.";
    return e.replace(/^(\w+):/, (_, k) => `${k.replace(/_/g, " ")}:`) + ".";
  }

  function schedule() {
    clearTimeout(timer);
    timer = setTimeout(estimate, 120);
  }

  // ------------------------------------------------------------------ result
  function renderResult(r) {
    $("loading").hidden = true;
    $("out").hidden = false;
    $("example").hidden = edited;
    $("price").textContent = money(r.price_azn);
    $("range").textContent = `80% range: ${money(r.price_low_azn)} – ${money(r.price_high_azn)}`;
    const span = Math.log(r.price_high_azn) - Math.log(r.price_low_azn);
    const at = span > 0 ? (100 * (Math.log(r.price_azn) - Math.log(r.price_low_azn))) / span : 50;
    $("bar-mark").style.left = `${Math.min(100, Math.max(0, at))}%`;
    const card = est.card();
    $("trees").textContent = `The middle 80% of the forest's ${card.forest.n_estimators} trees say ` +
      `${money(r.trees_low_azn)} – ${money(r.trees_high_azn)}. A wide spread means the trees disagree about this listing.`;
    $("ppm").textContent = money(r.price_per_m2_azn);
    const tier = $("tier");
    tier.textContent = r.premium ? "Premium" : "Standard";
    tier.className = `badge ${r.premium ? "premium" : "standard"}`;
    const sure = r.inside_margin
      ? `This listing is inside the SVM's margin, where it was right ${pct(card.acc_inside_margin ?? 0)} of the time on held-out listings.`
      : `This listing is outside the SVM's margin, where it was right ${pct(card.acc_outside_margin ?? 0)} of the time on held-out listings.`;
    $("tier-note").textContent = `Premium means above ${money(r.tier_threshold_azn)}, the median training price. ${sure}` +
      (r.models_agree ? "" : " The price model and the tier model disagree here, so this one is borderline.");
    renderFactors(r.factors);
    const u = r.location_used;
    const coords = { "you": "the spot you marked", "typical for location": "typical for this location",
                     "not known": "not known" }[u.coordinates_from] || u.coordinates_from;
    $("place-note").textContent = u.location
      ? `Location: ${[u.district, u.city].filter(Boolean).map(title).join(", ") || "as given"}; coordinates: ${coords}.`
      : u.lat != null ? "Location: the spot you marked on the map." : "";
    const ul = $("warnings");
    ul.replaceChildren(...r.warnings.map((w) => Object.assign(document.createElement("li"), { textContent: w })));
  }

  const title = (s) => s.replace(/\b\p{L}/gu, (c) => c.toUpperCase());

  function renderFactors(factors) {
    if (!factors) {
      $("factors-base").textContent = explainFailed
        ? "The breakdown could not be downloaded. Reload the page to try again."
        : `Downloading the breakdown (${(est.m.binary.explain.text_bytes / 2 ** 20).toFixed(0)} MB more)…`;
      $("factors").replaceChildren();
      return;
    }
    $("factors-base").textContent =
      `Compared with a typical listing (${money(est.options().typical_price_azn)}), each factor moved the estimate by:`;
    const max = Math.max(...factors.map((f) => Math.abs(f.log_effect)), 1e-9);
    $("factors").replaceChildren(...factors.map((f) => {
      const li = document.createElement("li");
      const name = Object.assign(document.createElement("span"), { className: "name", textContent: f.factor });
      const track = Object.assign(document.createElement("span"), { className: "track" });
      const fill = Object.assign(document.createElement("span"), { className: `fill ${f.log_effect > 0 ? "up" : "down"}` });
      fill.style.width = `${(50 * Math.abs(f.log_effect)) / max}%`;
      track.appendChild(fill);
      const val = Object.assign(document.createElement("span"), {
        className: "val", textContent: `${f.pct_effect > 0 ? "+" : ""}${f.pct_effect}%` });
      li.append(name, track, val);
      return li;
    }));
  }

  function renderCard(c) {
    const rows = [
      ["Price model", c.price_model],
      ["Tier model", c.tier_model],
      ["Test RMSE of log(price)", c.test_rmse_log.toFixed(3)],
      ["Test R² of log(price)", c.test_r2_log.toFixed(3)],
      ["Test mean abs. % error", pct(c.test_mape > 1 ? c.test_mape / 100 : c.test_mape)],
      ["80% range: share of test prices inside", pct(c.interval_coverage)],
      ["Tier F1 / ROC-AUC on test", `${c.test_f1.toFixed(3)} / ${c.test_roc_auc.toFixed(3)}`],
      ["Training / test listings", `${azn.format(c.n_dev)} / ${azn.format(c.n_test)}`],
      ["Trees in the forest", String(c.forest.n_estimators)],
      ["Trained", `${c.trained_at}${c.fast ? " (fast mode, demo only)" : ""}`],
    ];
    const table = document.createElement("table");
    for (const [k, v] of rows) {
      const tr = table.insertRow();
      tr.insertCell().textContent = k;
      tr.insertCell().textContent = v;
    }
    const wrap = Object.assign(document.createElement("div"), { className: "table-wrap" });
    wrap.appendChild(table);
    const note = document.createElement("p");
    note.textContent = "The test listings were held out from training and scored once. The 80% range comes from " +
      "the forest's out-of-bag errors, so it was calibrated without looking at the test set. This page runs " +
      "the same trained models as the project's Python app, ported to JavaScript and checked to give the same answers.";
    $("card").replaceChildren(wrap, note);
    $("sub").textContent = `Describe a flat or house and get a price from decision trees and an SVM we wrote from ` +
      `scratch in NumPy, trained on ${azn.format(c.n_dev)} bina.az listings. The models run on your device; ` +
      `nothing you type is sent anywhere.`;
  }

  // ------------------------------------------------------------------ map
  const map = (() => {
    const canvas = $("map");
    const ctx = canvas.getContext("2d");
    const COS = Math.cos((40.4 * Math.PI) / 180);
    let meta = null, cells = [], maxCount = 1, metros = [];
    let w = 0, h = 0, dpr = 1;
    let center = { lat: 40.405, lng: 49.87 };
    let k = 1500;                       // pixels per degree of latitude
    let kMin = 300;
    const K_MAX = 40000;
    let focus = null;                   // the chosen location's typical spot

    const toPx = (lat, lng) => [w / 2 + (lng - center.lng) * k * COS, h / 2 - (lat - center.lat) * k];
    const toGeo = (x, y) => ({ lat: center.lat - (y - h / 2) / k, lng: center.lng + (x - w / 2) / (k * COS) });
    const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

    function clampView() {
      if (!meta) return;
      const box = meta.box;
      k = Math.min(K_MAX, Math.max(kMin, k));
      center.lat = Math.min(box.lat[1], Math.max(box.lat[0], center.lat));
      center.lng = Math.min(box.lng[1], Math.max(box.lng[0], center.lng));
    }

    function resize() {
      dpr = window.devicePixelRatio || 1;
      w = canvas.clientWidth;
      h = canvas.clientHeight;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      if (meta) kMin = Math.min(w / ((meta.box.lng[1] - meta.box.lng[0]) * COS), h / (meta.box.lat[1] - meta.box.lat[0]));
      draw();
    }

    function draw() {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = css("--map-bg");
      ctx.fillRect(0, 0, w, h);
      if (meta && meta.rows) {
        const accent = css("--accent");
        if (!heat || heatColor !== accent) paintHeat(accent);
        const lat0 = meta.box.lat[0], lng0 = meta.box.lng[0], d = meta.cell_deg;
        const [x0, y0] = toPx(lat0 + meta.rows * d, lng0);
        const [x1, y1] = toPx(lat0, lng0 + meta.cols * d);
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = "high";
        ctx.drawImage(heat, x0, y0, x1 - x0, y1 - y0);
      }
      // metro stations, labelled once zoomed in
      const ink = css("--map-dot");
      ctx.font = `11px ${css("--mono")}`;
      for (const m of metros) {
        const [x, y] = toPx(m.lat, m.lng);
        if (x < -40 || y < -20 || x > w + 40 || y > h + 20) continue;
        ctx.fillStyle = ink;
        ctx.globalAlpha = 0.75;
        ctx.beginPath();
        ctx.arc(x, y, k > 2500 ? 3 : 2, 0, 2 * Math.PI);
        ctx.fill();
        if (k > 4500) { ctx.globalAlpha = 0.8; ctx.fillText(m.name, x + 5, y - 4); }
      }
      ctx.globalAlpha = 1;
      if (focus) {
        const [x, y] = toPx(focus.lat, focus.lng);
        ctx.strokeStyle = ink;
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.arc(x, y, 11, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      if (pin) {
        const [x, y] = toPx(pin.lat, pin.lng);
        ctx.fillStyle = css("--accent");
        ctx.strokeStyle = css("--surface");
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
      }
      drawScale(ink);
    }

    // one pixel per grid cell, north up; the browser smooths it when scaled
    let heat = null, heatColor = "";
    function paintHeat(color) {
      heat = document.createElement("canvas");
      heat.width = meta.cols;
      heat.height = meta.rows;
      const hc = heat.getContext("2d");
      hc.fillStyle = color;
      for (const [r, c, n] of cells) {
        hc.globalAlpha = 0.1 + 0.8 * (Math.log(n) / Math.log(maxCount));
        hc.fillRect(c, meta.rows - 1 - r, 1, 1);
      }
      heatColor = color;
    }

    function drawScale(ink) {
      const kmPerPx = 111.32 / k;
      const target = 80 * kmPerPx;
      const nice = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50].find((v) => v >= target * 0.6) || 50;
      const len = nice / kmPerPx;
      const x1 = w - 12, x0 = x1 - len, y = h - 10;
      ctx.strokeStyle = ink;
      ctx.fillStyle = ink;
      ctx.globalAlpha = 0.8;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x0, y - 4); ctx.lineTo(x0, y); ctx.lineTo(x1, y); ctx.lineTo(x1, y - 4);
      ctx.stroke();
      ctx.textAlign = "right";
      ctx.fillText(nice < 1 ? `${nice * 1000} m` : `${nice} km`, x1, y - 6);
      ctx.textAlign = "left";
      ctx.globalAlpha = 1;
    }

    function zoomAt(factor, x = w / 2, y = h / 2) {
      const before = toGeo(x, y);
      k *= factor;
      clampView();
      const after = toGeo(x, y);
      center.lat += before.lat - after.lat;
      center.lng += before.lng - after.lng;
      clampView();
      draw();
    }

    function goTo(lat, lng, zoom) {
      center = { lat, lng };
      if (zoom) k = Math.max(k, zoom);
      clampView();
      draw();
    }

    // pointer gestures: drag to pan, pinch to zoom, tap to drop the pin
    const pointers = new Map();
    let moved = false, pinch = 0;
    canvas.addEventListener("pointerdown", (e) => {
      canvas.setPointerCapture(e.pointerId);
      pointers.set(e.pointerId, { x: e.offsetX, y: e.offsetY });
      if (pointers.size === 1) moved = false;
      if (pointers.size === 2) { const [a, b] = [...pointers.values()]; pinch = Math.hypot(a.x - b.x, a.y - b.y); moved = true; }
    });
    canvas.addEventListener("pointermove", (e) => {
      const p = pointers.get(e.pointerId);
      if (!p) return;
      const dx = e.offsetX - p.x, dy = e.offsetY - p.y;
      if (pointers.size === 1) {
        if (!moved && Math.hypot(dx, dy) < 6) return;
        moved = true;
        center.lat += dy / k;
        center.lng -= dx / (k * COS);
        clampView();
        p.x = e.offsetX; p.y = e.offsetY;
        draw();
      } else if (pointers.size === 2) {
        p.x = e.offsetX; p.y = e.offsetY;
        const [a, b] = [...pointers.values()];
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        if (pinch > 0) zoomAt(d / pinch, (a.x + b.x) / 2, (a.y + b.y) / 2);
        pinch = d;
      }
    });
    const end = (e) => {
      const tap = pointers.size === 1 && !moved && e.type === "pointerup";
      pointers.delete(e.pointerId);
      if (pointers.size < 2) pinch = 0;
      if (tap) setPin(toGeo(e.offsetX, e.offsetY));
    };
    canvas.addEventListener("pointerup", end);
    canvas.addEventListener("pointercancel", end);
    canvas.addEventListener("keydown", (e) => {
      const step = 60;
      const moves = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] };
      if (moves[e.key]) {
        const [dx, dy] = moves[e.key];
        center.lat -= dy / k;
        center.lng += dx / (k * COS);
        clampView();
        draw();
      } else if (e.key === "+" || e.key === "=") zoomAt(1.6);
      else if (e.key === "-") zoomAt(1 / 1.6);
      else if (e.key === "Enter") setPin({ ...center });
      else return;
      e.preventDefault();
    });
    $("zoom-in").addEventListener("click", () => zoomAt(1.6));
    $("zoom-out").addEventListener("click", () => zoomAt(1 / 1.6));
    new ResizeObserver(resize).observe(canvas);
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
    new MutationObserver(draw).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

    function init(density, gazetteer) {
      meta = density;
      if (density) {
        cells = density.cells;
        maxCount = Math.max(2, ...cells.map((c) => c[2]));
      } else {
        $("map-key").textContent = "Dots are metro stations.";
        meta = { box: { lat: [40.25, 40.65], lng: [49.45, 50.4] } };
      }
      metros = Object.entries(gazetteer)
        .filter(([key, g]) => key.endsWith(" m.") && g.lat != null)
        .map(([, g]) => ({ lat: g.lat, lng: g.lng, name: g.label.replace(/ \(metro\)$/, "") }));
      resize();
    }

    return { init, draw, goTo, toGeo, setFocus: (f) => { focus = f; draw(); }, inBox: (lat, lng) =>
      meta && lat >= meta.box.lat[0] && lat <= meta.box.lat[1] && lng >= meta.box.lng[0] && lng <= meta.box.lng[1] };
  })();

  function nearestLocation(lat, lng) {
    let best = null, bestKm = Infinity;
    for (const [key, g] of Object.entries(est.options().locations.reduce((o, l) => (o[l.value] = l, o), {}))) {
      if (g.lat == null) continue;
      const dy = (g.lat - lat) * 111.32, dx = (g.lng - lng) * 111.32 * Math.cos((lat * Math.PI) / 180);
      const d = Math.hypot(dx, dy);
      if (d < bestKm) { bestKm = d; best = key; }
    }
    return bestKm <= 2 ? best : null;
  }

  function setPin(p) {
    if (!est) return;
    pin = { lat: Math.round(p.lat * 1e5) / 1e5, lng: Math.round(p.lng * 1e5) / 1e5 };
    let msg = `Marked spot: ${pin.lat.toFixed(4)}, ${pin.lng.toFixed(4)}.`;
    if (!$("location").value) {
      const near = nearestLocation(pin.lat, pin.lng);
      if (near) {
        $("location").value = near;
        map.setFocus(null);
        msg += ` Neighbourhood set to the nearest one, ${est.m.gazetteer[near].label}; change it if that's wrong.`;
      }
    }
    $("coords").textContent = `${msg} Tap again to move it.`;
    $("map-clear").hidden = false;
    edited = true;
    map.draw();
    schedule();
  }

  function clearPin(silent) {
    pin = null;
    $("map-clear").hidden = true;
    $("coords").textContent = "Tap the map to mark the exact spot (optional). Drag to move, pinch or use +/− to zoom.";
    map.draw();
    if (!silent) schedule();
  }

  function focusLocation() {
    const g = est && est.m.gazetteer[$("location").value];
    if (g && g.lat != null) {
      map.setFocus({ lat: g.lat, lng: g.lng });
      if (map.inBox(g.lat, g.lng)) map.goTo(g.lat, g.lng, 3500);
      else $("coords").textContent = `${g.label} is outside the map area; its typical coordinates are used.`;
    } else map.setFocus(null);
  }

  // ------------------------------------------------------------------ start
  function fillOptions() {
    const opts = est.options();
    const allowed = new Set(opts.categories.map((c) => c.value));
    for (const input of form.querySelectorAll("input[name=category]")) input.closest("label").hidden = !allowed.has(input.value);
    const sel = $("location");
    for (const loc of opts.locations) sel.add(new Option(loc.label, loc.value));
    if (est.m.gazetteer[DEFAULT_LOCATION]) sel.value = DEFAULT_LOCATION;
  }

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const s = Object.assign(document.createElement("script"), { src, onload: resolve, onerror: reject });
      document.head.appendChild(s);
    });
  }

  form.addEventListener("submit", (e) => { e.preventDefault(); estimate(); });
  form.addEventListener("input", (e) => {
    if (e.target.name === "category") syncCategory();
    edited = true;
    schedule();
  });
  $("location").addEventListener("change", () => { clearPin(true); focusLocation(); });
  $("map-clear").addEventListener("click", () => clearPin(false));
  syncCategory();

  (async () => {
    const mb = (b) => `${(b / 2 ** 20).toFixed(0)} MB`;
    try {
      if (typeof DecompressionStream !== "function") await loadScript(PAKO);
      est = await Estimator.load("", (done, total) => {
        $("dl-size").textContent = mb(total);
        $("progress").style.width = `${(100 * done) / total}%`;
        $("loading-text").textContent = done < total
          ? `Downloading the models… ${mb(done)} of ${mb(total)}`
          : "Unpacking the models…";
      });
      fillOptions();
      renderCard(est.card());
      map.init(est.m.density, est.m.gazetteer);
      focusLocation();
      estimate();
      est.explainReady.then(estimate, () => { explainFailed = true; estimate(); });
    } catch (err) {
      $("loading-text").textContent = `The models could not be loaded: ${err.message}. Reload the page to try again.`;
    }
  })();
})();
