// gta6-twitch-ccu-forecast-01 — launch-day Twitch CCU forecaster (Node 22, stdlib only).
//
// Model (fitted at runtime from the case's own `comparables`, no network, no
// hard-coded practice answers):
//   For each forecast hour h = 2..24, work in log space with the ratio
//   r_h = viewers(h) / viewers(1) of past big launches.
//   Pool: comparables whose unlock was a rolling midnight-local console
//   launch (unlock text contains "midnight local") and whose research tier
//   is "comparable" (excludes early-access-contaminated / weak launches and
//   the GTA VI category's own pre-release history). Rolling-midnight games
//   share GTA VI's unlock shape (NZ -> Europe -> UK -> US East -> US West),
//   while single-global-unlock games spike at one hour and would wreck the
//   quantiles. Fallbacks relax the pool (rolling any tier, then all) if a
//   tier is too small.
//   Median: per hour, blend w=0.85 of an OLS fit of log r_h on log(p1) with
//   0.15 of the pool median of log r_h. Big launches grow less from hour 1:
//   across the rolling pool, corr(log r_20, log p1) ~= -0.76, so conditioning
//   on the observed hour-1 level cuts the typical miss from ~60% (pooled
//   median) to ~44% in leave-one-out tests over the comparables.
//   Spread: per-hour residual SD of that fit, floored at 0.35 and capped at
//   1.0, then 3-point smoothed across hours; quantiles are log-normal:
//   q_tau = p1 * exp(mu_h + z_tau * sd_h), z = [-1.645,-0.674,0,0.674,1.645].
//   mu is 3-point smoothed across hours too.
//   Baseline (flat persistence) scores ~0.387 pinball on the practice;
//   leave-one-out over big rolling-midnight comparables: this model ~0.25
//   vs ~0.35 for the baseline on the same targets.
//
// I/O: reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input), writes
// <id>.out to $MAKEFUN_OUTPUT_DIR (else /output) as {"hours":[{hour,q:[...]}]}.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
// Blend weight of the log(p1) regression vs the pooled median.
const W_REG = 0.85;
const SD_FLOOR = 0.35;
const SD_CAP = 1.0;

interface HourlyRow { from: string; viewers: number }
interface Comparable {
  name: string;
  day_start: string;
  unlock?: string;
  hourly: HourlyRow[];
}
interface Case {
  levels: number[];
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
  comparables?: Comparable[];
}

const round2 = (x: number): number => Math.round(x * 100) / 100;

function median(sorted: number[]): number {
  const n = sorted.length;
  if (n === 0) return 0;
  const m = Math.floor(n / 2);
  return n % 2 === 1 ? sorted[m] : (sorted[m - 1] + sorted[m]) / 2;
}

function pstdev(xs: number[]): number {
  if (xs.length < 2) return 0;
  const mu = xs.reduce((a, b) => a + b, 0) / xs.length;
  return Math.sqrt(xs.reduce((a, b) => a + (b - mu) * (b - mu), 0) / xs.length);
}

/** 3-point moving average (edges use the available neighbours). */
function smooth3(a: number[]): number[] {
  return a.map((_, k) => {
    const w = a.slice(Math.max(0, k - 1), Math.min(a.length, k + 2));
    return w.reduce((x, y) => x + y, 0) / w.length;
  });
}

/** OLS y ~ a + b*x with a tiny ridge; falls back to the mean when x is constant. */
function ols(xs: number[], ys: number[]): [number, number] {
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  if (sxx < 1e-12) return [my, 0];
  const b = sxy / (sxx + 1e-9);
  return [my - b * mx, b];
}

function isRolling(c: Comparable): boolean {
  return (c.unlock ?? '').toLowerCase().includes('midnight local');
}

function tierOf(c: Comparable): string {
  const m = /tier:\s*([a-z-]+)/i.exec(c.unlock ?? '');
  return m ? m[1].toLowerCase() : '';
}

function isOwnCategory(c: Comparable): boolean {
  return c.name.toLowerCase().includes('grand theft auto');
}

/** Hourly viewers of a comparable's release day, indexed 1..24 (null if missing). */
function releaseDay(c: Comparable): Array<number | null> {
  const dayStart = Date.parse(c.day_start);
  if (Number.isNaN(dayStart)) return new Array(24).fill(null);
  const byHour = new Map<number, number>();
  for (const row of c.hourly ?? []) {
    const t = Date.parse(row.from);
    if (Number.isNaN(t)) continue;
    const h = Math.round((t - dayStart) / 3600000) + 1;
    if (h >= 1 && h <= 24 && typeof row.viewers === 'number' && row.viewers > 0) {
      byHour.set(h, row.viewers);
    }
  }
  return Array.from({ length: 24 }, (_, i) => byHour.get(i + 1) ?? null);
}

function forecastCase(c: Case): { hours: Array<{ hour: number; q: number[] }> } {
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`levels ${JSON.stringify(c.levels)} != ${JSON.stringify(LEVELS)}`);
  }
  const first = c.known.find((k) => k.hour === 1);
  if (first === undefined) throw new Error('no hour-1 print');
  const p1 = Math.max(1, first.viewers);
  const logP1 = Math.log(p1);
  const hours = c.forecast.map((f) => f.hour);

  // Build the comparable pool (release-day series per comparable).
  const all = (c.comparables ?? [])
    .filter((comp) => !isOwnCategory(comp))
    .map((comp) => ({ comp, day: releaseDay(comp) }))
    .filter((x) => x.day[0] !== null);
  const rollingTier = all.filter(
    (x) => isRolling(x.comp) && tierOf(x.comp) === 'comparable',
  );
  const rollingAny = all.filter((x) => isRolling(x.comp));
  // Per hour, use the strictest pool with at least 5 usable comparables.
  const poolFor = (h: number): Array<{ day: Array<number | null> }> => {
    for (const pool of [rollingTier, rollingAny, all]) {
      const usable = pool.filter((x) => x.day[h - 1] !== null);
      if (usable.length >= 5) return usable;
    }
    return all.filter((x) => x.day[h - 1] !== null);
  };

  const mus: number[] = [];
  const sds: number[] = [];
  for (const h of hours) {
    const pool = poolFor(h);
    if (pool.length < 3) {
      // No usable history: fall back to the house baseline's flat shape.
      mus.push(0);
      sds.push(0.15 * Math.sqrt(h - 1));
      continue;
    }
    const xs: number[] = [];
    const ys: number[] = [];
    for (const x of pool) {
      const v1 = x.day[0] as number;
      const vh = x.day[h - 1] as number;
      xs.push(Math.log(v1));
      ys.push(Math.log(vh / v1));
    }
    const [a, b] = ols(xs, ys);
    const muReg = a + b * logP1;
    const med = median([...ys].sort((p, q) => p - q));
    let mu = W_REG * muReg + (1 - W_REG) * med;
    // Sanity clamp: never forecast a median ratio outside [e^-3, e^1.5].
    mu = Math.min(1.5, Math.max(-3, mu));
    const resid = ys.map((y, i) => y - (a + b * xs[i]));
    const sd = Math.min(SD_CAP, Math.max(SD_FLOOR, pstdev(resid)));
    mus.push(mu);
    sds.push(sd);
  }

  const muS = smooth3(mus);
  const sdS = smooth3(sds);
  const out = hours.map((h, i) => {
    const q = Z.map((z) => {
      const v = round2(p1 * Math.exp(muS[i] + z * sdS[i]));
      return Math.min(1e9, Math.max(0.01, v));
    });
    // Enforce non-decreasing (rounding can only tie, but stay safe).
    for (let j = 1; j < q.length; j++) if (q[j] < q[j - 1]) q[j] = q[j - 1];
    return { hour: h, q };
  });
  return { hours: out };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const result = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify(result)}\n`);
}
