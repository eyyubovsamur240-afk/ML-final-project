// Compare predictor.js with the Python predictor on golden cases.
//   node app/static_site/check.js <site dir> <golden.json>
// golden.json = [{input: listing, expected: PricePredictor.predict(...) as a dict}, ...]
// (written by app/export_static.py --check or tests/test_static_site.py).
// Prints a JSON summary; exit code 1 if any case differs beyond the tolerances.
"use strict";

const fs = require("fs");
const path = require("path");
const zlib = require("zlib");
const { Estimator } = require("./predictor.js");

const [site, goldenPath] = process.argv.slice(2);
const meta = JSON.parse(fs.readFileSync(path.join(site, "model.json"), "utf8"));
const stream = (s) => new Uint8Array(zlib.gunzipSync(Buffer.concat(
  s.files.map((f) => Buffer.from(fs.readFileSync(path.join(site, f), "utf8"), "base64")))));
const est = new Estimator(meta, stream(meta.binary.core));
est.attachExplain(stream(meta.binary.explain));
const cases = JSON.parse(fs.readFileSync(goldenPath, "utf8"));

// Prices are rounded to 100 AZN, scores to 1e-4. The JS walk does the same
// float64 operations as NumPy, so results are normally identical; libm
// (exp, log, cos) and BLAS summation order may differ in the last bit, which
// can move a value across a rounding boundary. Allow exactly one rounding unit.
const UNIT = {
  price_azn: 100, price_low_azn: 100, price_high_azn: 100, trees_low_azn: 100, trees_high_azn: 100,
  tier_threshold_azn: 100, price_per_m2_azn: 1, svm_score: 1e-4,
};
const EPS = 1e-9;

const failures = [];
let exact = 0;
const maxDiff = {};
for (const [i, { input, expected }] of cases.entries()) {
  let got;
  try {
    got = est.predict(input);
  } catch (err) {
    failures.push({ case: i, error: String(err.message || err) });
    continue;
  }
  const bad = [];
  let same = true;
  for (const [key, unit] of Object.entries(UNIT)) {
    const d = Math.abs(got[key] - expected[key]);
    maxDiff[key] = Math.max(maxDiff[key] || 0, d);
    if (d > 0) same = false;
    if (d > unit + EPS * Math.abs(expected[key])) bad.push(`${key}: js ${got[key]} vs py ${expected[key]}`);
  }
  // Booleans follow from the SVM score / price; only a score within 1e-9 of a
  // decision boundary could legitimately flip them.
  const nearBoundary = Math.abs(expected.svm_score) < 1e-4 || Math.abs(Math.abs(expected.svm_score) - 1) < 1e-4;
  for (const key of ["premium", "inside_margin", "models_agree"]) {
    if (got[key] !== expected[key]) {
      same = false;
      if (!nearBoundary) bad.push(`${key}: js ${got[key]} vs py ${expected[key]}`);
    }
  }
  if (JSON.stringify(got.location_used) !== JSON.stringify(expected.location_used)) {
    same = false;
    bad.push(`location_used: js ${JSON.stringify(got.location_used)} vs py ${JSON.stringify(expected.location_used)}`);
  }
  if (JSON.stringify(got.warnings) !== JSON.stringify(expected.warnings)) {
    same = false;
    bad.push(`warnings: js ${JSON.stringify(got.warnings)} vs py ${JSON.stringify(expected.warnings)}`);
  }
  const gf = got.factors, ef = expected.factors;
  if (gf.length !== ef.length || gf.some((f, j) => f.factor !== ef[j].factor)) {
    same = false;
    bad.push(`factors: js ${gf.map((f) => f.factor)} vs py ${ef.map((f) => f.factor)}`);
  } else {
    gf.forEach((f, j) => {
      const dl = Math.abs(f.log_effect - ef[j].log_effect), dp = Math.abs(f.pct_effect - ef[j].pct_effect);
      maxDiff.log_effect = Math.max(maxDiff.log_effect || 0, dl);
      maxDiff.pct_effect = Math.max(maxDiff.pct_effect || 0, dp);
      if (dl > 0 || dp > 0) same = false;
      if (dl > 1e-4 + EPS || dp > 0.1 + EPS) bad.push(`factor ${f.factor}: js ${f.log_effect}/${f.pct_effect} vs py ${ef[j].log_effect}/${ef[j].pct_effect}`);
    });
  }
  if (same) exact += 1;
  if (bad.length) failures.push({ case: i, input, diffs: bad });
}

console.log(JSON.stringify({
  cases: cases.length, identical: exact, failures: failures.length,
  max_abs_diff: maxDiff, first_failures: failures.slice(0, 5),
}));
process.exit(failures.length ? 1 : 0);
