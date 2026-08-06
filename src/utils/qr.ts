import { hashCode } from './format';

/**
 * Generates a deterministic, QR-code-looking SVG data URL seeded from a string.
 * Includes the three finder patterns for visual authenticity. Not a scannable
 * standard QR code — used for report verification mock-ups.
 */
export function generateQrSvg(seed: string, modules = 21, size = 96): string {
  const h = hashCode(seed);
  let state = h;

  const rnd = () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0xffffffff;
  };

  const cell = size / modules;
  const cells: { x: number; y: number }[] = [];
  const inFinder = (x: number, y: number) =>
    (x < 7 && y < 7) || (x >= modules - 7 && y < 7) || (x < 7 && y >= modules - 7);

  for (let y = 0; y < modules; y++) {
    for (let x = 0; x < modules; x++) {
      if (inFinder(x, y)) continue;
      if (rnd() < 0.45) cells.push({ x, y });
    }
  }

  const finder = (fx: number, fy: number) => {
    const rects = [
      `<rect x="${fx * cell}" y="${fy * cell}" width="${7 * cell}" height="${7 * cell}" fill="#0f172a"/>`,
      `<rect x="${(fx + 1) * cell}" y="${(fy + 1) * cell}" width="${5 * cell}" height="${5 * cell}" fill="white"/>`,
      `<rect x="${(fx + 2) * cell}" y="${(fy + 2) * cell}" width="${3 * cell}" height="${3 * cell}" fill="#0f172a"/>`,
    ];
    return rects.join('');
  };

  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}" shape-rendering="crispEdges">
      <rect width="${size}" height="${size}" fill="white"/>
      ${cells.map((c) => `<rect x="${c.x * cell}" y="${c.y * cell}" width="${cell}" height="${cell}" fill="#0f172a"/>`).join('')}
      ${finder(0, 0)}
      ${finder(modules - 7, 0)}
      ${finder(0, modules - 7)}
    </svg>
  `;

  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

/** Suggested follow-up tests derived from the top diagnosis. */
export function suggestedTests(topDiagnosis: string): string[] {
  const t = topDiagnosis.toLowerCase();
  if (t.includes('covid') || t.includes('viral')) {
    return ['RT-PCR SARS-CoV-2', 'CBC + CRP + D-Dimer', 'Arterial Blood Gas (ABG)', 'High-Resolution CT Chest'];
  }
  if (t.includes('cardio') || t.includes('heart')) {
    return ['Echocardiogram (LVEF)', 'ECG 12-lead', 'BNP / NT-proBNP', 'Serial CXR'];
  }
  if (t.includes('effusion')) {
    return ['Diagnostic Thoracentesis (Light\'s Criteria)', 'Thoracic Ultrasound', 'Chest CT', 'Pleural Fluid Culture'];
  }
  if (t.includes('pneumothorax')) {
    return ['Chest CT (quantification)', 'Serial upright CXR', 'Oxygen saturation monitoring'];
  }
  if (t.includes('no finding') || t.includes('normal')) {
    return ['None — routine screening complete'];
  }
  return ['CBC with differential', 'Sputum Gram stain & Culture', 'CRP / ESR', 'Follow-up CXR in 4–6 weeks'];
}
