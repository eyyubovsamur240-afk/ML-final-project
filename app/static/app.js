// Front end for the price estimator: fills the form from /api/options,
// posts the listing to /api/predict and renders the answer. No build step.
"use strict";

const $ = (id) => document.getElementById(id);
const form = $("form");
const azn = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const money = (v) => `${azn.format(v)} AZN`;
const pct = (v) => `${(100 * v).toFixed(1)}%`;
let options = null;
let map = null;
let pin = null;

async function getJSON(url, init) {
  const res = await fetch(url, init);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = body.detail;
    const msg = Array.isArray(d) ? d.map((e) => `${e.loc.slice(-1)[0]}: ${e.msg}`).join("; ") : d;
    throw new Error(msg || `Request failed (${res.status})`);
  }
  return body;
}

function renderOptions(opts) {
  const seg = $("category-group");
  opts.categories.forEach((c, i) => {
    const label = document.createElement("label");
    label.innerHTML = `<input type="radio" name="category" value="${c.value}" ${i === 0 ? "checked" : ""}><span>${c.label}</span>`;
    seg.appendChild(label);
  });
  const sel = $("location");
  for (const loc of opts.locations) {
    const o = document.createElement("option");
    o.value = loc.value;
    o.textContent = loc.label;
    sel.appendChild(o);
  }
  seg.addEventListener("change", syncCategory);
  syncCategory();
}

function syncCategory() {
  const house = form.category.value === "heyet evi/bag evi";
  $("land").hidden = !house;
  $("floors").hidden = house;
}

// The map is optional: Leaflet loads from a CDN after the form works, and if
// it can't be reached the map box simply disappears.
const LEAFLET = "https://unpkg.com/leaflet@1.9.4/dist/leaflet";
function loadLeaflet() {
  const css = Object.assign(document.createElement("link"), {
    rel: "stylesheet", href: `${LEAFLET}.css`, crossOrigin: "",
    integrity: "sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" });
  document.head.appendChild(css);
  return new Promise((resolve, reject) => {
    const js = Object.assign(document.createElement("script"), {
      src: `${LEAFLET}.js`, crossOrigin: "", onload: resolve, onerror: reject,
      integrity: "sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" });
    document.head.appendChild(js);
    setTimeout(() => reject(new Error("timeout")), 8000);
  });
}

function hideMap() { $("map").hidden = true; $("coords").hidden = true; }

function initMap(center) {
  if (!window.L) { hideMap(); return; }
  map = L.map("map", { scrollWheelZoom: false }).setView(center, 11);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    { maxZoom: 18, attribution: "© OpenStreetMap" }).addTo(map);
  map.on("click", (e) => setPin(e.latlng.lat, e.latlng.lng, true));
  $("location").addEventListener("change", () => {
    const loc = options.locations.find((l) => l.value === $("location").value);
    clearPin();
    if (loc && loc.lat != null) map.setView([loc.lat, loc.lng], 13);
  });
}

function setPin(lat, lng, fromUser) {
  if (pin) pin.setLatLng([lat, lng]); else pin = L.marker([lat, lng]).addTo(map);
  form.lat.value = lat.toFixed(5);
  form.lng.value = lng.toFixed(5);
  $("coords").textContent = `Exact spot: ${lat.toFixed(4)}, ${lng.toFixed(4)}` + (fromUser ? " (click again to move)" : "");
}

function clearPin() {
  if (pin) { pin.remove(); pin = null; }
  form.lat.value = form.lng.value = "";
  $("coords").textContent = "Click the map to set the exact spot (optional).";
}

function readForm() {
  const num = (name) => (form[name].value === "" ? null : Number(form[name].value));
  const house = form.category.value === "heyet evi/bag evi";
  const body = {
    category: form.category.value,
    area_m2: num("area_m2"),
    rooms: num("rooms"),
    floor: house ? null : num("floor"),
    total_floors: house ? null : num("total_floors"),
    land_area_sot: house ? num("land_area_sot") : null,
    location: form.location.value || null,
    lat: num("lat"),
    lng: num("lng"),
    repair: form.repair.checked,
    mortgage: form.mortgage.checked,
    bill_of_sale: form.bill_of_sale.checked,
    description: form.description.value.trim(),
  };
  return body;
}

function validate() {
  let ok = true;
  for (const el of form.querySelectorAll("input[type=number]")) {
    if (el.closest("[hidden]")) continue;
    const bad = !el.checkValidity();
    el.setAttribute("aria-invalid", bad);
    ok = ok && !bad;
  }
  return ok;
}

function renderResult(r) {
  $("empty").hidden = true;
  $("out").hidden = false;
  $("price").textContent = money(r.price_azn);
  $("range").textContent = `80% range: ${money(r.price_low_azn)} – ${money(r.price_high_azn)}`;
  const span = Math.log(r.price_high_azn) - Math.log(r.price_low_azn);
  $("bar-mark").style.left = `${(100 * (Math.log(r.price_azn) - Math.log(r.price_low_azn))) / span}%`;
  $("trees").textContent = `The forest's ${window.modelCard?.forest?.n_estimators ?? ""} trees mostly agree on ${money(r.trees_low_azn)} – ${money(r.trees_high_azn)} ` +
    "(middle 80% of individual trees; a wide spread means the models are unsure about this listing).";
  $("ppm").textContent = money(r.price_per_m2_azn);
  renderFactors(r.factors);
  const tier = $("tier");
  tier.textContent = r.premium ? "Premium" : "Standard";
  tier.className = `badge ${r.premium ? "premium" : "standard"}`;
  const card = window.modelCard || {};
  const sure = r.inside_margin
    ? `This listing is inside the SVM's margin, where it was right ${pct(card.acc_inside_margin ?? 0)} of the time on held-out listings.`
    : `This listing is outside the SVM's margin, where it was right ${pct(card.acc_outside_margin ?? 0)} of the time on held-out listings.`;
  $("tier-note").textContent = `Premium means above ${money(r.tier_threshold_azn)}, the median training price. ${sure}` +
    (r.models_agree ? "" : " The price model and the tier model disagree here, so this one is borderline.");
  const u = r.location_used;
  $("place-note").textContent = u.location
    ? `Location: ${[u.district, u.city].filter(Boolean).join(", ") || "as given"}; coordinates ${u.coordinates_from}.`
    : u.lat != null ? "Location: coordinates you picked on the map." : "";
  $("warnings").innerHTML = "";
  for (const w of r.warnings) {
    const li = document.createElement("li");
    li.textContent = w;
    $("warnings").appendChild(li);
  }
}

function renderFactors(factors) {
  $("factors-base").textContent =
    `Compared with a typical listing (${money(options.typical_price_azn)}), each factor moved the estimate by:`;
  const max = Math.max(...factors.map((f) => Math.abs(f.log_effect)), 1e-9);
  $("factors").innerHTML = "";
  for (const f of factors) {
    const li = document.createElement("li");
    const w = (50 * Math.abs(f.log_effect)) / max;
    const sign = f.pct_effect > 0 ? "+" : "";
    li.innerHTML = `<span></span><span class="track"><span class="fill ${f.log_effect > 0 ? "up" : "down"}" style="width:${w}%"></span></span><span class="val">${sign}${f.pct_effect}%</span>`;
    li.firstChild.textContent = f.factor;
    $("factors").appendChild(li);
  }
}

function renderCard(c) {
  window.modelCard = c;
  const rows = [
    ["Price model", c.price_model],
    ["Tier model", c.tier_model],
    ["Test RMSE of log(price)", c.test_rmse_log.toFixed(3)],
    ["Test R² of log(price)", c.test_r2_log.toFixed(3)],
    ["Test mean abs. % error", pct(c.test_mape > 1 ? c.test_mape / 100 : c.test_mape)],
    ["80% range coverage on test", pct(c.interval_coverage)],
    ["Tier F1 / ROC-AUC on test", `${c.test_f1.toFixed(3)} / ${c.test_roc_auc.toFixed(3)}`],
    ["Training / test listings", `${azn.format(c.n_dev)} / ${azn.format(c.n_test)}`],
    ["Trained", `${c.trained_at}${c.fast ? " (fast mode, demo only)" : ""}`],
  ];
  const tbody = rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
  $("card").innerHTML = `<table>${tbody}</table>`;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  $("error").hidden = true;
  if (!validate()) {
    $("error").textContent = "Please fix the highlighted fields.";
    $("error").hidden = false;
    return;
  }
  $("submit").disabled = true;
  $("submit").textContent = "Estimating…";
  try {
    const r = await getJSON("/api/predict", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(readForm()),
    });
    renderResult(r);
  } catch (err) {
    $("error").textContent = err.message;
    $("error").hidden = false;
  } finally {
    $("submit").disabled = false;
    $("submit").textContent = "Estimate price";
  }
});

(async () => {
  try {
    options = await getJSON("/api/options");
    renderOptions(options);
    renderCard(await getJSON("/api/model"));
    loadLeaflet().then(() => initMap(options.map_center), hideMap);
  } catch (err) {
    $("error").textContent = `The model is not available: ${err.message}`;
    $("error").hidden = false;
    $("submit").disabled = true;
  }
})();
