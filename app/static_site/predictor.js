/*
 * predictor.js — the web interface's serving model, ported from app/predictor.py
 * so the estimator runs entirely in the browser (no Python server needed).
 *
 * It reproduces PricePredictor.predict step by step on the exported bundle:
 *   listing -> one-row feature frame (data_prep.make_features)
 *           -> design row (Preprocessor.transform: median imputation,
 *              missing flags, one-hot levels, location-tag columns)
 *           -> random forest of our CART trees (per-tree walk, NumPy's
 *              pairwise mean, linear-interpolated 10/90% quantiles)
 *           -> RFF Pegasos SVM on the standardised row
 *           -> Saabas path decomposition grouped into plain-language factors.
 * tests/test_static_site.py runs this file under Node and checks it against
 * the Python predictor on many listings, so the two cannot drift apart.
 *
 * Works in the browser (window.Estimator) and in Node (module.exports).
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.Estimator = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // ---------------------------------------------------------------- numerics
  /** NumPy's pairwise summation (np.add.reduce on a contiguous float64 array). */
  function pairwiseSum(a, lo, n) {
    if (n < 8) {
      let r = 0.0;
      for (let i = 0; i < n; i++) r += a[lo + i];
      return r;
    }
    if (n <= 128) {
      const r = [a[lo], a[lo + 1], a[lo + 2], a[lo + 3], a[lo + 4], a[lo + 5], a[lo + 6], a[lo + 7]];
      let i = 8;
      for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] += a[lo + i + j];
      let res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
      for (; i < n; i++) res += a[lo + i];
      return res;
    }
    let n2 = Math.floor(n / 2);
    n2 -= n2 % 8;
    return pairwiseSum(a, lo, n2) + pairwiseSum(a, lo + n2, n - n2);
  }
  const npMean = (a) => pairwiseSum(a, 0, a.length) / a.length;

  /** np.quantile(a, q) with the default 'linear' method (NumPy's two-sided lerp). */
  function npQuantile(a, q) {
    const s = Float64Array.from(a).sort();
    const n = s.length;
    const vi = (n - 1) * q;
    const lo = Math.floor(vi);
    const g = vi - lo;
    const x0 = s[lo], x1 = s[Math.min(lo + 1, n - 1)];
    const d = x1 - x0;
    return g >= 0.5 ? x1 - d * (1 - g) : x0 + d * g;
  }

  /** Python's round(x, ndigits): round half to even. */
  function pyRound(x, ndigits) {
    const f = Math.pow(10, ndigits);
    const y = ndigits >= 0 ? x * f : x / Math.pow(10, -ndigits);
    let r = Math.round(y);
    if (Math.abs(y % 1) === 0.5) r = 2 * Math.round(y / 2);   // tie -> even
    const out = ndigits >= 0 ? r / f : r * Math.pow(10, -ndigits);
    return out === 0 ? 0 : out;                                 // no -0
  }

  /** Python's format(x, 'g') for the moderate values used in warnings. */
  function fmtG(x) {
    return String(parseFloat(Number(x).toPrecision(6)));
  }

  // ---------------------------------------------------------------- text
  const AZ = { "ə": "e", "Ə": "e", "ı": "i", "İ": "i", "ö": "o", "Ö": "o", "ü": "u", "Ü": "u",
               "ç": "c", "Ç": "c", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g" };
  /** data_prep.transliterate: Azerbaijani letters to ASCII, strip accents, lower-case. */
  function transliterate(text) {
    let t = "";
    for (const ch of String(text)) t += AZ[ch] !== undefined ? AZ[ch] : ch;
    return t.normalize("NFKD").replace(/\p{Mn}/gu, "").toLowerCase();
  }
  const codePoints = (s) => Array.from(s).length;   // Python len() counts code points

  function haversineKm(lat1, lng1, lat2, lng2) {
    const r = Math.PI / 180;
    const [p1, l1, p2, l2] = [lat1 * r, lng1 * r, lat2 * r, lng2 * r];
    const a = Math.sin((p2 - p1) / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin((l2 - l1) / 2) ** 2;
    return 2 * 6371.0 * Math.asin(Math.sqrt(a));
  }

  const isNum = (v) => typeof v === "number" && !Number.isNaN(v);
  const nan = NaN;

  // ---------------------------------------------------------------- model
  class Estimator {
    /**
     * @param {object} meta  model.json written by app/export_static.py
     * @param {Uint8Array} core  the gunzipped model-*.bin stream (meta.binary.core)
     */
    constructor(meta, core) {
      this.m = meta;
      const p = meta.preprocess;
      this.cont = p.numeric.concat(p.binary);
      this.nFeat = p.feature_names.length;
      const sec = unpack(core, meta.binary.core.sections);

      // Forest: every tree's nodes in pre-order, so a split's left child is the
      // next node. feat = split feature (-1 = leaf). Thresholds come from one
      // sorted dictionary per feature, leaf values from a dictionary too (both
      // exact float64). Right children are rebuilt: after a leaf, the next node
      // is the right child of the deepest split still waiting for one.
      const f = meta.forest;
      const N = f.total_nodes;
      this.feat = sec.feat;
      this.thr = new Float64Array(N);
      this.val = new Float64Array(N).fill(NaN);   // internal means arrive with attachExplain
      this.right = new Int32Array(N).fill(-1);
      this.treeStart = f.tree_offsets;
      this.nTrees = f.n_trees;
      const open = [];
      let s = 0, l = 0;
      for (let t = 0; t < this.nTrees; t++) {
        const lo = this.treeStart[t], hi = t + 1 < this.nTrees ? this.treeStart[t + 1] : N;
        for (let i = lo; i < hi; i++) {
          if (i > lo && this.feat[i - 1] < 0) this.right[open.pop()] = i;
          const ft = this.feat[i];
          if (ft >= 0) {
            this.thr[i] = sec.thr_dict[sec.thr_off[ft] + sec.thr_idx[s++]];
            open.push(i);
          } else this.val[i] = sec.leaf_dict[sec.leaf_idx[l++]];
        }
        if (open.length) throw new Error(`tree ${t} is not a complete pre-order tree`);
      }
      if (s !== sec.thr_idx.length || l !== sec.leaf_idx.length) throw new Error("the model data does not match the trees");
      this.hasExplain = false;

      // RFF SVM: W f64[d*D] (row-major), phase c f64[D], weights coef f64[D].
      this.D = meta.svm.n_components;
      if (meta.svm.n_features !== this.nFeat) throw new Error("SVM and preprocessor disagree on features");
      this.W = sec.W;
      this.c = sec.c;
      this.coef = sec.coef;
    }

    /** Add the internal node means (explain-*.bin), needed only by explain(). */
    attachExplain(raw) {
      const nodeVal = unpack(raw, this.m.binary.explain.sections).node_val;
      let s = 0;
      for (let i = 0; i < this.feat.length; i++) if (this.feat[i] >= 0) this.val[i] = nodeVal[s++];
      if (s !== nodeVal.length) throw new Error("the explanation data does not match the trees");
      this.hasExplain = true;
    }

    // ---------------------------------------------------------- input checks
    /** The API's validation (schemas.ListingIn + the unknown-location check). */
    validate(item) {
      const r = this.m.limits;
      const errs = [];
      const inRange = (name, v, lo, hi, int) => {
        if (v === null || v === undefined) return;
        if (!isNum(v) || v < lo || v > hi || (int && !Number.isInteger(v)))
          errs.push(`${name}: must be ${int ? "a whole number " : ""}between ${lo} and ${hi}`);
      };
      if (!this.m.category_labels[item.category]) errs.push("category: choose new building, old building or house");
      if (item.area_m2 === null || item.area_m2 === undefined) errs.push("area_m2: required");
      inRange("area_m2", item.area_m2, r.area[0], r.area[1], false);
      inRange("rooms", item.rooms, r.rooms[0], r.rooms[1], true);
      inRange("floor", item.floor, -2, r.max_floors, true);
      inRange("total_floors", item.total_floors, 1, r.max_floors, true);
      inRange("land_area_sot", item.land_area_sot, r.land[0], r.land[1], false);
      inRange("lat", item.lat, r.lat[0], r.lat[1], false);
      inRange("lng", item.lng, r.lng[0], r.lng[1], false);
      if (isNum(item.floor) && isNum(item.total_floors) && item.floor > item.total_floors)
        errs.push("floor cannot be above the number of floors in the building");
      if ((item.lat == null) !== (item.lng == null)) errs.push("give both lat and lng, or neither");
      if (codePoints(item.description || "") > r.description) errs.push(`description: at most ${r.description} characters`);
      if (item.location != null && !(item.location in this.m.gazetteer))
        errs.push(`Unknown location '${item.location}'`);
      return errs;
    }

    // ---------------------------------------------------------- features
    /** PricePredictor._frame: one cleaned-looking row + what was used for location. */
    frame(item) {
      const place = item.location ? this.m.gazetteer[item.location] || null : null;
      const lat = item.lat != null ? item.lat : place && place.lat != null ? place.lat : null;
      const lng = item.lng != null ? item.lng : place && place.lng != null ? place.lng : null;
      const house = item.category === "heyet evi/bag evi";
      const row = {
        area_m2: Number(item.area_m2),
        rooms: item.rooms != null ? item.rooms : nan,
        floor: item.floor != null ? item.floor : nan,
        total_floors: item.total_floors != null ? item.total_floors : nan,
        land_area_sot: house && item.land_area_sot != null ? item.land_area_sot : nan,
        lat: lat != null ? lat : nan,
        lng: lng != null ? lng : nan,
        repair: item.repair == null ? nan : item.repair ? 1.0 : 0.0,
        mortgage: item.mortgage ? 1.0 : 0.0,
        bill_of_sale: item.bill_of_sale ? 1.0 : 0.0,
        category: item.category,
        building_type: null,
        city: place ? place.city : null,
        district: place ? place.district : null,
        location: item.location || null,
        tags: (place && place.tags) || "",
        description: item.description || "",
      };
      const text = (v) => (typeof v === "string" ? v : null);
      const used = {
        location: text(row.location), district: text(row.district), city: text(row.city),
        lat, lng,
        coordinates_from: item.lat != null ? "you" : lat != null ? "typical for location" : "not known",
      };
      return { row, used };
    }

    /** data_prep.make_features for one row. */
    features(row) {
      const f = {};
      const a = row.area_m2, rooms = row.rooms, fl = row.floor, tot = row.total_floors;
      f.area_m2 = a;
      f.log_area = Math.log(a);
      f.rooms = rooms;
      f.area_per_room = a / rooms;
      f.floor = fl;
      f.total_floors = tot;
      f.floor_ratio = fl / tot;
      f.is_top_floor = isNum(tot) && isNum(fl) ? (fl === tot ? 1.0 : 0.0) : nan;
      f.is_first_floor = isNum(fl) ? (fl <= 1 ? 1.0 : 0.0) : nan;
      f.land_area_sot = isNum(row.land_area_sot) ? row.land_area_sot : 0.0;
      f.has_land = f.land_area_sot > 0 ? 1.0 : 0.0;
      f.lat = row.lat;
      f.lng = row.lng;
      const c = this.m.constants.baku_center;
      f.dist_center_km = isNum(row.lat) && isNum(row.lng) ? haversineKm(row.lat, row.lng, c[0], c[1]) : nan;
      f.desc_log_len = Math.log1p(codePoints(row.description));
      f.repair = row.repair;
      f.mortgage = row.mortgage;
      f.bill_of_sale = row.bill_of_sale;
      const desc = transliterate(row.description);
      for (const [name, words] of Object.entries(this.m.constants.description_keywords))
        f[name] = words.some((w) => desc.includes(w)) ? 1.0 : 0.0;
      for (const k of this.m.preprocess.categorical) f[k] = row[k];
      for (const k of this.m.preprocess.multilabel) f[k] = row[k] || "";
      return f;
    }

    /** Preprocessor.transform for one row -> Float64Array of the model's columns. */
    transform(f) {
      const p = this.m.preprocess;
      const x = new Float64Array(this.nFeat);
      let j = 0;
      for (const k of this.cont) {
        const v = Number(f[k]);
        x[j++] = Number.isNaN(v) ? p.medians[k] : v;
      }
      for (const k of p.indicator_cols) x[j++] = Number.isNaN(Number(f[k])) ? 1.0 : 0.0;
      const hits = {};
      for (const k of p.categorical) {
        hits[k] = false;
        for (const lvl of p.levels[k]) {
          const on = f[k] === lvl;
          if (on) hits[k] = true;
          x[j++] = on ? 1.0 : 0.0;
        }
      }
      for (const k of p.categorical) x[j++] = hits[k] ? 0.0 : 1.0;
      for (const k of p.multilabel) {
        const have = new Set(String(f[k] || "").split("|").filter(Boolean));
        for (const lab of p.labels[k]) x[j++] = have.has(lab) ? 1.0 : 0.0;
      }
      if (j !== this.nFeat) throw new Error(`built ${j} columns, model expects ${this.nFeat}`);
      return x;
    }

    // ---------------------------------------------------------- models
    leaf(t, x) {
      let node = this.treeStart[t];
      while (this.feat[node] >= 0)
        node = x[this.feat[node]] <= this.thr[node] ? node + 1 : this.right[node];
      return node;
    }

    svmScore(x) {
      const s = this.m.scaler, d = this.nFeat, D = this.D;
      const z = new Float64Array(d);
      for (let i = 0; i < d; i++) z[i] = (x[i] - s.mean[i]) / s.scale[i];
      const amp = Math.sqrt(2.0 / D);
      let score = 0.0;
      for (let k = 0; k < D; k++) {
        let dot = 0.0;
        for (let i = 0; i < d; i++) dot += z[i] * this.W[i * D + k];
        score += amp * Math.cos(dot + this.c[k]) * this.coef[k];
      }
      return score + this.m.svm.intercept;
    }

    /** PricePredictor.explain: Saabas path decomposition, grouped into factors. */
    explain(x, top = 6) {
      const contrib = new Float64Array(this.nFeat);
      for (let t = 0; t < this.nTrees; t++) {
        let node = this.treeStart[t];
        while (this.feat[node] >= 0) {
          const f = this.feat[node];
          const child = x[f] <= this.thr[node] ? node + 1 : this.right[node];
          contrib[f] += this.val[child] - this.val[node];
          node = child;
        }
      }
      for (let i = 0; i < this.nFeat; i++) contrib[i] /= this.nTrees;
      const k = this.m.constants;
      if (k.per_m2) contrib[k.log_area_col] += x[k.log_area_col] - k.log_area_ref;
      const grouped = new Map();
      this.m.preprocess.feature_names.forEach((name, i) => {
        let base = name.split("=")[0];
        if (base.endsWith("_missing")) base = base.slice(0, -"_missing".length);
        const label = k.factor_labels[base] || base;
        grouped.set(label, (grouped.get(label) || 0.0) + contrib[i]);
      });
      return [...grouped.entries()]
        .sort((p, q) => Math.abs(q[1]) - Math.abs(p[1]))
        .slice(0, top)
        .filter(([, v]) => Math.abs(v) > 1e-4)
        .map(([factor, v]) => ({ factor, log_effect: pyRound(v, 4), pct_effect: pyRound(100 * Math.expm1(v), 1) }));
    }

    warnings(item) {
      const out = [];
      const R = this.m.ranges;
      for (const [name, value, label] of [["area_m2", item.area_m2, "Area"], ["rooms", item.rooms, "Rooms"],
                                         ["total_floors", item.total_floors, "Floors in the building"]]) {
        const [lo, hi] = R[name];
        if (value != null && !(lo <= value && value <= hi))
          out.push(`${label} ${fmtG(value)} is outside what the models saw in training ` +
                   `(${fmtG(lo)}–${fmtG(hi)}); treat the estimate with care.`);
      }
      if (item.location && !(item.location in this.m.gazetteer)) out.push("Unknown location; it was treated as 'other'.");
      if (item.location == null && item.lat == null)
        out.push("No location given; location is the strongest price driver, so the estimate is rough.");
      return out;
    }

    // ---------------------------------------------------------- predict
    /** PricePredictor.predict -> the same fields as the API's PredictionOut. */
    /** Each tree's log-price for design row x (forest output + the log-area offset). */
    perTree(x) {
      const k = this.m.constants;
      const offset = k.per_m2 ? x[k.log_area_col] : 0.0;
      const out = new Float64Array(this.nTrees);
      for (let t = 0; t < this.nTrees; t++) out[t] = this.val[this.leaf(t, x)] + offset;
      return out;
    }

    /** The price estimate alone (same number as predict().price_azn, no SVM or
     *  explanation): cheap enough to run for many variants of a listing. */
    quote(item) {
      if (this.validate(item).length) return null;
      const x = this.transform(this.features(this.frame(item).row));
      return pyRound(Math.exp(npMean(this.perTree(x))), -2);
    }

    predict(item) {
      const errs = this.validate(item);
      if (errs.length) { const e = new Error(errs.join("; ")); e.details = errs; throw e; }
      const { row, used } = this.frame(item);
      const x = this.transform(this.features(row));
      const perTree = this.perTree(x);
      const logPrice = npMean(perTree);
      const tLo = Math.exp(npQuantile(perTree, 0.10)), tHi = Math.exp(npQuantile(perTree, 0.90));
      const score = this.svmScore(x);
      const price = Math.exp(logPrice);
      const [lo, hi] = this.m.interval;
      const thr = this.m.threshold;
      return {
        price_azn: pyRound(price, -2),
        price_low_azn: pyRound(price * Math.exp(lo), -2),
        price_high_azn: pyRound(price * Math.exp(hi), -2),
        price_per_m2_azn: pyRound(price / item.area_m2, 0),
        premium: score >= 0,
        svm_score: pyRound(score, 4),
        inside_margin: Math.abs(score) < 1.0,
        tier_threshold_azn: pyRound(thr, -2),
        models_agree: (price > thr) === (score >= 0),
        trees_low_azn: pyRound(tLo, -2),
        trees_high_azn: pyRound(tHi, -2),
        factors: this.hasExplain ? this.explain(x) : null,   // null until attachExplain
        location_used: used,
        warnings: this.warnings(item),
      };
    }

    options() { return this.m.options; }
    card() { return this.m.card; }
  }

  const TYPES = { f64: Float64Array, i32: Int32Array, i16: Int16Array, u32: Uint32Array, u16: Uint16Array };
  /** Split the model data into typed arrays, undoing the byte shuffle. */
  function unpack(raw, sections) {
    const out = {};
    for (const s of sections) {
      const T = TYPES[s.dtype], w = T.BYTES_PER_ELEMENT, n = s.count;
      const bytes = new Uint8Array(n * w);
      if (s.shuffled) {
        for (let b = 0; b < w; b++) {
          const src = s.offset + b * n;
          for (let i = 0; i < n; i++) bytes[i * w + b] = raw[src + i];
        }
      } else bytes.set(raw.subarray(s.offset, s.offset + n * w));
      out[s.name] = new T(bytes.buffer);
    }
    return out;
  }

  async function gunzip(bytes) {
    if (typeof DecompressionStream === "function") {
      const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
      return new Uint8Array(await new Response(stream).arrayBuffer());
    }
    if (typeof pako !== "undefined") return pako.ungzip(bytes);   // older browsers
    throw new Error("this browser cannot unpack the model (no DecompressionStream)");
  }

  /** base64 text -> bytes (Uint8Array.fromBase64 where the browser has it). */
  function fromBase64(text) {
    if (typeof Uint8Array.fromBase64 === "function") return Uint8Array.fromBase64(text.trim());
    const bin = atob(text.trim());
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  /** Download one stream's files (gzip, base64 text) and join them. */
  async function fetchPacked(base, stream, onProgress) {
    let done = 0;
    onProgress(0, stream.text_bytes, false);
    const chunks = await Promise.all(stream.files.map(async (name) => {
      const res = await fetch(`${base}${name}`);
      if (!res.ok) throw new Error(`could not download ${name} (${res.status})`);
      const text = await res.text();
      done += text.length;
      onProgress(done, stream.text_bytes, false);
      return fromBase64(text);
    }));
    const packed = new Uint8Array(stream.bytes);
    let off = 0;
    for (const c of chunks) { packed.set(c, off); off += c.byteLength; }
    if (off !== stream.bytes) throw new Error("the model data is incomplete");
    return packed;
  }

  async function gunzipChecked(packed, stream) {
    const raw = await gunzip(packed);
    if (raw.byteLength !== stream.raw_bytes) throw new Error("the model data is incomplete");
    return raw;
  }

  /**
   * Browser loader: fetch model.json and the model files published next to the
   * page. Resolves as soon as estimates work; est.explainReady resolves once the
   * explanation data has arrived too. onProgress(bytesDone, bytesTotal, fromCache)
   * follows the first download.
   */
  async function load(base = "", onProgress = () => {}) {
    const res = await fetch(`${base}model.json`);
    if (!res.ok) throw new Error(`could not download model.json (${res.status})`);
    const meta = await res.json();
    // The model files are cached on the device after the first visit, keyed by
    // the exact files, so a retrained model is downloaded again.
    const model = `${meta.card.trained_at}|${meta.card.data_sha256}|`;
    const key = (stream) => `${model}${stream.files.join(",")}|${stream.bytes}`;
    const get = async (stream, progress) => {
      const cached = await cacheGet(key(stream));
      if (cached && cached.byteLength === stream.bytes) {
        progress(stream.text_bytes, stream.text_bytes, true);
        return gunzipChecked(cached, stream);
      }
      const packed = await fetchPacked(base, stream, progress);
      cachePut(key(stream), packed, model);    // fire and forget
      return gunzipChecked(packed, stream);
    };
    const est = new Estimator(meta, await get(meta.binary.core, onProgress));
    est.explainReady = get(meta.binary.explain, () => {}).then((raw) => est.attachExplain(raw));
    return est;
  }

  // ------------------------------------------------------------ device cache
  // IndexedDB can be missing, full, blocked (private mode) or slow to answer:
  // every failure falls back to the network, and no call waits more than 3 s.
  const DB = "estimator-cache", STORE = "streams";
  function idb(mode, work) {
    return new Promise((resolve) => {
      const done = (v) => { clearTimeout(timer); resolve(v); };
      const timer = setTimeout(() => resolve(null), 3000);
      try {
        const open = indexedDB.open(DB, 1);
        open.onupgradeneeded = () => open.result.createObjectStore(STORE);
        open.onerror = () => done(null);
        open.onsuccess = () => {
          try {
            const tx = open.result.transaction(STORE, mode);
            const req = work(tx.objectStore(STORE));
            tx.oncomplete = () => { done(req ? req.result : true); open.result.close(); };
            tx.onerror = tx.onabort = () => { done(null); open.result.close(); };
          } catch (e) { done(null); }
        };
      } catch (e) { done(null); }
    });
  }
  async function cacheGet(key) {
    if (typeof indexedDB === "undefined") return null;
    const v = await idb("readonly", (store) => store.get(key));
    return v instanceof ArrayBuffer ? new Uint8Array(v) : v instanceof Uint8Array ? v : null;
  }
  /** Store one stream; files of other (older) models are removed. */
  async function cachePut(key, bytes, model) {
    if (typeof indexedDB === "undefined") return;
    const buf = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    return idb("readwrite", (store) => {
      const keys = store.getAllKeys();
      keys.onsuccess = () => {
        for (const k of keys.result) if (!String(k).startsWith(model)) store.delete(k);
        store.put(buf, key);
      };
      return null;
    });
  }

  return { Estimator, load, _internals: { unpack, fromBase64, pairwiseSum, npMean, npQuantile, pyRound, transliterate, fmtG } };
});
