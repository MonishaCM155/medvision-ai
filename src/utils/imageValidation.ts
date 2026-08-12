/**
 * imageValidation.ts
 * ------------------------------------------------
 * Client-side medical-image gate for the inference pipeline.
 *
 * Before any disease prediction is allowed, the uploaded image must be
 * validated as a usable FRONTAL CHEST X-RAY. This module performs:
 *
 *   1. Format / corruption check     — decodable, supported type
 *   2. Resolution check              — large enough for a radiograph
 *   3. Grayscale check               — rejects color photographs/pets/landscapes
 *   4. Orientation / aspect check    — rejects documents & wide landscape shots
 *   5. Brightness check              — rejects too-dark / too-bright inputs
 *   6. Contrast check                — rejects flat, washed-out inputs
 *   7. Sharpness check               — rejects blurry images (Laplacian variance)
 *   8. Thoracic-anatomy check        — structural heuristic: bright mediastinal
 *                                      spine band flanked by dark lung fields
 *                                      (rejects CT slices, limb X-rays, ultrasound,
 *                                      documents, and unrelated content)
 *
 * The checks are intentionally conservative heuristics (no pretrained detector
 * is bundled with the demo). Every rejection returns a human-readable reason.
 */

export type ValidationStatus = 'pass' | 'warn' | 'fail';

export interface ValidationCheck {
  key: string;
  label: string;
  status: ValidationStatus;
  detail: string;
}

export interface ImageMetrics {
  width: number;
  height: number;
  aspectRatio: number; // width / height
  orientation: 'portrait' | 'landscape' | 'square';
  meanIntensity: number; // 0-255 mean luma
  stdIntensity: number; // luma standard deviation
  contrastRatio: number; // dynamic range (p98 - p2)
  colorDeviation: number; // max mean channel gap (0 = pure grayscale)
  sharpness: number; // Laplacian variance
  structureScore: number; // 0-100 thoracic-anatomy likelihood
}

export interface ValidationReport {
  passed: boolean;
  score: number; // 0-100 aggregate suitability score
  checks: ValidationCheck[];
  metrics: ImageMetrics;
  message: string; // human-readable outcome (reason when rejected)
}

/** Analysis resolution cap — keeps computation fast and resolution-agnostic. */
const ANALYSIS_MAX_DIM = 512;
/** Minimum short edge for a usable chest radiograph (pixels). */
const MIN_SHORT_EDGE = 256;

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Could not decode image data'));
    img.src = src;
  });
}

interface GrayStats {
  mean: number;
  std: number;
  p2: number;
  p98: number;
  contrastRatio: number; // dynamic range (p98 - p2)
}

/** Downscale to a grayscale Uint8ClampedArray + luma histogram stats. */
function toGrayStats(img: HTMLImageElement): { w: number; h: number; data: Uint8ClampedArray; stats: GrayStats } {
  const scale = Math.min(1, ANALYSIS_MAX_DIM / Math.max(img.naturalWidth, img.naturalHeight));
  const w = Math.max(1, Math.round(img.naturalWidth * scale));
  const h = Math.max(1, Math.round(img.naturalHeight * scale));

  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('Canvas 2D context unavailable');
  ctx.drawImage(img, 0, 0, w, h);

  const imageData = ctx.getImageData(0, 0, w, h);
  const px = imageData.data;
  const gray = new Uint8ClampedArray(w * h);

  let sum = 0;
  let sumSq = 0;
  const hist = new Float64Array(256);

  for (let i = 0; i < gray.length; i++) {
    const r = px[i * 4];
    const g = px[i * 4 + 1];
    const b = px[i * 4 + 2];
    const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    gray[i] = luma;
    sum += luma;
    sumSq += luma * luma;
    hist[Math.min(255, Math.round(luma))] += 1;
  }

  const n = gray.length;
  const mean = sum / n;
  const variance = Math.max(0, sumSq / n - mean * mean);
  const std = Math.sqrt(variance);

  // Percentiles for dynamic range
  let acc = 0;
  let p2 = 0;
  let p98 = 255;
  for (let v = 0; v < 256; v++) {
    acc += hist[v];
    if (acc >= n * 0.02 && p2 === 0) p2 = v;
    if (acc >= n * 0.98) {
      p98 = v;
      break;
    }
  }

  return {
    w,
    h,
    data: gray,
    stats: {
      mean,
      std,
      p2,
      p98,
      contrastRatio: Math.max(0.5, p98 - p2),
    },
  };
}

/** Laplacian variance — classic no-reference sharpness metric. */
function laplacianVariance(gray: Uint8ClampedArray, w: number, h: number): number {
  let acc = 0;
  let count = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const lap = 4 * gray[i] - gray[i - 1] - gray[i + 1] - gray[i - w] - gray[i + w];
      acc += lap * lap;
      count++;
    }
  }
  return count ? acc / count : 0;
}

/** Mean luma of a rectangular region (fractions of width/height). */
function regionMean(gray: Uint8ClampedArray, w: number, h: number, x0: number, y0: number, x1: number, y1: number): number {
  const xa = Math.max(0, Math.round(x0 * w));
  const xb = Math.min(w, Math.round(x1 * w));
  const ya = Math.max(0, Math.round(y0 * h));
  const yb = Math.min(h, Math.round(y1 * h));
  if (xb <= xa || yb <= ya) return 0;
  let sum = 0;
  let count = 0;
  for (let y = ya; y < yb; y++) {
    for (let x = xa; x < xb; x++) {
      sum += gray[y * w + x];
      count++;
    }
  }
  return count ? sum / count : 0;
}

/**
 * Thoracic-anatomy heuristic.
 * Frontal CXRs share a strong signature: a bright vertical mediastinal/spine
 * band down the middle, flanked by darker lung fields, with bright clavicle
 * density in the upper third. Limb films, CT slices, documents and photos do
 * not exhibit this pattern.
 */
function thoracicStructureScore(gray: Uint8ClampedArray, w: number, h: number): number {
  const overall = regionMean(gray, w, h, 0, 0, 1, 1) || 1;

  const center = regionMean(gray, w, h, 0.4, 0.08, 0.6, 0.92); // spine/mediastinum band
  const leftLung = regionMean(gray, w, h, 0.16, 0.25, 0.38, 0.85);
  const rightLung = regionMean(gray, w, h, 0.62, 0.25, 0.84, 0.85);
  const upper = regionMean(gray, w, h, 0.25, 0.05, 0.75, 0.32); // clavicle/shoulder zone

  const centerRatio = center / overall;
  const lungRatio = (leftLung + rightLung) / 2 / overall;
  const upperRatio = upper / overall;

  // Bright spine vs dark lungs is the strongest signal.
  const spineScore = Math.min(100, Math.max(0, (centerRatio - 1.0) * 130));
  // Lungs darker than the global mean (dark negative-space on both sides).
  const lungScore = Math.min(40, Math.max(0, (1.05 - lungRatio) * 120));
  // Upper-third brightness (clavicles / shoulders).
  const upperScore = Math.min(25, Math.max(0, (upperRatio - 0.8) * 60));

  let score = spineScore + lungScore + upperScore;

  if (lungRatio >= 1.15) {
    // Both flanks brighter than the global mean — no dark lung fields at all.
    // This is the signature of limb films / ultrasound frames / bright CT
    // windows, not a chest radiograph.
    score = Math.min(35, spineScore * 0.4);
  } else if (lungRatio >= 1.0) {
    // Flanks not meaningfully darker than average — borderline (e.g. severe
    // bilateral consolidation). Cap so it can only WARN, never pass.
    score = Math.min(55, score);
  }

  return Math.min(100, Math.round(score));
}

/* ------------------------------------------------------------------ */
/* Main analysis                                                       */
/* ------------------------------------------------------------------ */

export function analyzeImage(dataUrl: string): Promise<ValidationReport> {
  return loadImage(dataUrl)
    .then((img) => {
      const { w, h, data: gray, stats } = toGrayStats(img);
      const metrics = analyzeMetrics(img, gray, w, h, stats);
      return buildReport(metrics);
    })
    .catch((err) => {
      const reason = err instanceof Error ? err.message : 'Unknown decode error';
      return buildReport(emptyMetrics(), [corruptCheck(reason)]);
    });
}

function corruptCheck(reason: string): ValidationCheck {
  return {
    key: 'format',
    label: 'File integrity & format',
    status: 'fail',
    detail: `The image file is corrupted, truncated, or in an unsupported format — ${reason}. Re-export the image as PNG/JPEG/WebP and retry.`,
  };
}

function analyzeMetrics(
  img: HTMLImageElement,
  gray: Uint8ClampedArray,
  w: number,
  h: number,
  stats: GrayStats
): ImageMetrics {
  const aspectRatio = h > 0 ? img.naturalWidth / img.naturalHeight : 0;
  const orientation: ImageMetrics['orientation'] = aspectRatio > 1.08 ? 'landscape' : aspectRatio < 0.92 ? 'portrait' : 'square';
  return {
    width: img.naturalWidth,
    height: img.naturalHeight,
    aspectRatio,
    orientation,
    meanIntensity: Math.round(stats.mean * 10) / 10,
    stdIntensity: Math.round(stats.std * 10) / 10,
    contrastRatio: Math.round(stats.contrastRatio * 10) / 10,
    colorDeviation: Math.round(maxChannelDeviation(img, w, h) * 10) / 10,
    sharpness: Math.round(laplacianVariance(gray, w, h)),
    structureScore: thoracicStructureScore(gray, w, h),
  };
}

/** Max mean |channel gap| across pixels — 0 for pure grayscale. */
function maxChannelDeviation(img: HTMLImageElement, w: number, h: number): number {
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return 255;
  ctx.drawImage(img, 0, 0, w, h);
  const px = ctx.getImageData(0, 0, w, h).data;
  let sum = 0;
  for (let i = 0; i < px.length; i += 4) {
    sum += Math.max(Math.abs(px[i] - px[i + 1]), Math.abs(px[i + 1] - px[i + 2]), Math.abs(px[i] - px[i + 2]));
  }
  return sum / Math.max(1, px.length / 4);
}

/** Builds a full-frame stats structure for the corrupt path (all-zero metrics). */
function emptyMetrics(): ImageMetrics {
  return {
    width: 0,
    height: 0,
    aspectRatio: 0,
    orientation: 'square',
    meanIntensity: 0,
    stdIntensity: 0,
    contrastRatio: 0,
    colorDeviation: 255,
    sharpness: 0,
    structureScore: 0,
  };
}

function buildReport(metrics: ImageMetrics, forcedChecks?: ValidationCheck[]): ValidationReport {
  const checks: ValidationCheck[] = forcedChecks || [];
  if (!forcedChecks) {
    const fail = (key: string, label: string, detail: string): ValidationCheck => ({ key, label, status: 'fail', detail });
    const pass = (key: string, label: string, detail: string): ValidationCheck => ({ key, label, status: 'pass', detail });
    const warn = (key: string, label: string, detail: string): ValidationCheck => ({ key, label, status: 'warn', detail });

    const m = metrics;
    const minEdge = Math.min(m.width, m.height);
    const tooSmall = minEdge < MIN_SHORT_EDGE;

    checks.push(
      pass('format', 'File integrity & format', `${m.width}×${m.height} px · decodable image`),
      tooSmall
        ? fail('resolution', 'Image resolution', `Short edge is ${minEdge}px — a diagnostic radiograph needs at least ${MIN_SHORT_EDGE}px. The upload is too small/low-resolution.`)
        : pass('resolution', 'Image resolution', `${m.width}×${m.height} px meets minimum radiograph resolution`),
      m.colorDeviation > 18
        ? fail('grayscale', 'Grayscale characteristics', `Strong color detected (mean channel gap ${m.colorDeviation.toFixed(1)}). Chest radiographs are grayscale — this looks like a color photograph, pet photo, landscape, or other non-radiographic content.`)
        : pass('grayscale', 'Grayscale characteristics', `Grayscale (mean channel gap ${m.colorDeviation.toFixed(1)}) — consistent with a radiograph`),
      m.aspectRatio <= 0 || m.aspectRatio > 1.45
        ? fail('orientation', 'Orientation & framing', `${m.orientation} framing (aspect ${m.aspectRatio.toFixed(2)}). Frontal chest X-rays are near-square or portrait. This framing is consistent with a document, landscape photo, or incorrectly oriented scan — rotate to portrait and retry.`)
        : pass('orientation', 'Orientation & framing', `${m.orientation} framing (aspect ${m.aspectRatio.toFixed(2)}) matches chest radiograph geometry`),
      m.meanIntensity < 20
        ? fail('brightness', 'Exposure / brightness', `Image is too dark (mean intensity ${m.meanIntensity}). The radiograph may be under-exposed, over-collimated, or corrupted.`)
        : m.meanIntensity > 235
          ? fail('brightness', 'Exposure / brightness', `Image is too bright (mean intensity ${m.meanIntensity}). Over-exposed or blank-white input is not diagnostically usable.`)
          : pass('brightness', 'Exposure / brightness', `Exposure acceptable (mean intensity ${m.meanIntensity}/255)`),
      m.stdIntensity < 8
        ? fail('contrast', 'Contrast / dynamic range', `Low contrast (σ ${m.stdIntensity.toFixed(1)}, range ${m.contrastRatio}). The image is flat or washed out — no useful anatomy contrast.`)
        : pass('contrast', 'Contrast / dynamic range', `Adequate contrast (σ ${m.stdIntensity.toFixed(1)}, dynamic range ${m.contrastRatio})`),
      m.sharpness < 15
        ? fail('sharpness', 'Sharpness / blur', `Image appears blurry (focus metric ${m.sharpness}). Blurred or motion-affected inputs are unsuitable for diagnosis — recapture with the patient holding still.`)
        : m.sharpness < 30
          ? warn('sharpness', 'Sharpness / blur', `Focus is borderline (metric ${m.sharpness}). Acceptable, but a sharper image is preferred.`)
          : pass('sharpness', 'Sharpness / blur', `Sharp focus (metric ${m.sharpness})`),
      m.structureScore < 45
        ? fail('anatomy', 'Frontal chest X-ray detection', `No thoracic anatomy signature detected (structure score ${m.structureScore}/100). The image does not match a frontal chest radiograph — it may be a CT/MRI/ultrasound slice, a limb X-ray, a document, or unrelated content. Only frontal chest X-rays can be analyzed.`)
        : m.structureScore < 60
          ? warn('anatomy', 'Frontal chest X-ray detection', `Weak thoracic anatomy signature (structure score ${m.structureScore}/100). Analysis will proceed with reduced confidence.`)
          : pass('anatomy', 'Frontal chest X-ray detection', `Thoracic anatomy detected (structure score ${m.structureScore}/100) — bright mediastinal band flanked by dark lung fields`)
    );
  }

  const failed = checks.filter((c) => c.status === 'fail');
  const warned = checks.filter((c) => c.status === 'warn');
  const passedCount = checks.filter((c) => c.status === 'pass').length;

  const passed = failed.length === 0;
  const score = Math.max(0, Math.min(100, Math.round((passedCount / checks.length) * 100 - warned.length * 5)));

  const message = passed
    ? `Valid frontal chest X-ray (suitability ${score}/100). Ready for AI analysis.`
    : `Image rejected — ${failed[0].label.toLowerCase()}. ${failed[0].detail}`;

  return { passed, score, checks, metrics, message };
}
