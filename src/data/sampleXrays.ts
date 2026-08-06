import { PredictionResult, ModelMetadata } from '../types';

export interface SampleXray {
  id: string;
  title: string;
  subtitle: string;
  category: string;
  description: string;
  svgDataUrl: string;
  heatmapSvgUrl: string;
  sampleResult: Omit<PredictionResult, 'id' | 'timestamp'>;
}

// Generate realistic SVG Chest X-Ray Data URLs
function createChestXraySvg(type: 'normal' | 'pneumonia' | 'cardiomegaly' | 'effusion' | 'covid'): string {
  let pathologyOverlay = '';
  if (type === 'pneumonia') {
    // Opacity in right lower lobe
    pathologyOverlay = `
      <ellipse cx="230" cy="270" rx="35" ry="25" fill="rgba(255, 255, 255, 0.45)" filter="blur(8px)" />
      <ellipse cx="240" cy="280" rx="20" ry="18" fill="rgba(255, 255, 255, 0.65)" filter="blur(4px)" />
    `;
  } else if (type === 'cardiomegaly') {
    // Enlarged cardiac silhouette
    pathologyOverlay = `
      <!-- Enlarged heart shadow extending into left chest -->
      <path d="M 180 220 Q 250 250 280 320 Q 230 350 160 320 Z" fill="rgba(220, 220, 220, 0.85)" filter="blur(6px)" />
    `;
  } else if (type === 'effusion') {
    // Costophrenic angle blunting right lung base
    pathologyOverlay = `
      <path d="M 210 330 Q 270 340 280 370 L 200 370 Z" fill="rgba(240, 240, 240, 0.9)" filter="blur(3px)" />
    `;
  } else if (type === 'covid') {
    // Bilateral peripheral ground glass opacities
    pathologyOverlay = `
      <ellipse cx="120" cy="240" rx="30" ry="40" fill="rgba(255, 255, 255, 0.5)" filter="blur(7px)" />
      <ellipse cx="260" cy="260" rx="35" ry="35" fill="rgba(255, 255, 255, 0.55)" filter="blur(7px)" />
      <ellipse cx="110" cy="300" rx="25" ry="20" fill="rgba(255, 255, 255, 0.4)" filter="blur(6px)" />
    `;
  }

  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 450" width="400" height="450" style="background:#0a0c10;">
      <defs>
        <radialGradient id="lungField" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="#18202c"/>
          <stop offset="100%" stop-color="#080a0e"/>
        </radialGradient>
        <linearGradient id="ribGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="rgba(255,255,255,0.08)"/>
          <stop offset="50%" stop-color="rgba(255,255,255,0.4)"/>
          <stop offset="100%" stop-color="rgba(255,255,255,0.08)"/>
        </linearGradient>
        <filter id="blurFilter">
          <feGaussianBlur stdDeviation="3" />
        </filter>
      </defs>

      <!-- Main Thoracic Background -->
      <rect width="400" height="450" fill="#08090d"/>

      <!-- Thoracic Cavity Outline -->
      <path d="M 80 80 Q 200 40 320 80 L 340 380 Q 200 420 60 380 Z" fill="#0d1117" stroke="rgba(255,255,255,0.15)" stroke-width="2"/>

      <!-- Spine Column -->
      <rect x="190" y="50" width="20" height="340" fill="rgba(220, 220, 220, 0.6)" filter="blur(2px)"/>
      ${Array.from({ length: 14 }).map((_, i) => `<line x1="185" y1="${60 + i * 22}" x2="215" y2="${60 + i * 22}" stroke="rgba(255,255,255,0.8)" stroke-width="3"/>`).join('')}

      <!-- Clavicles -->
      <path d="M 80 85 Q 140 105 190 95" stroke="rgba(240,240,240,0.85)" stroke-width="8" fill="none" stroke-linecap="round"/>
      <path d="M 320 85 Q 260 105 210 95" stroke="rgba(240,240,240,0.85)" stroke-width="8" fill="none" stroke-linecap="round"/>

      <!-- Right Lung Field -->
      <path d="M 90 110 Q 180 110 180 340 Q 90 350 80 200 Z" fill="url(#lungField)"/>

      <!-- Left Lung Field -->
      <path d="M 310 110 Q 220 110 220 340 Q 310 350 320 200 Z" fill="url(#lungField)"/>

      <!-- Cardiac Silhouette (Normal) -->
      ${type !== 'cardiomegaly' ? `
        <path d="M 185 220 Q 230 250 240 310 Q 200 330 180 300 Z" fill="rgba(210, 210, 210, 0.75)" filter="blur(4px)"/>
      ` : ''}

      <!-- Diaphragm -->
      <path d="M 70 360 Q 130 330 190 355" stroke="rgba(255,255,255,0.7)" stroke-width="5" fill="none"/>
      <path d="M 210 355 Q 270 335 330 365" stroke="rgba(255,255,255,0.7)" stroke-width="5" fill="none"/>

      <!-- Rib Cage Bands (Anterior/Posterior Ribs) -->
      ${[120, 150, 180, 210, 240, 270, 300, 330].map(y => `
        <path d="M 75 ${y} Q 130 ${y - 15} 185 ${y + 5}" stroke="url(#ribGrad)" stroke-width="4" fill="none"/>
        <path d="M 325 ${y} Q 270 ${y - 15} 215 ${y + 5}" stroke="url(#ribGrad)" stroke-width="4" fill="none"/>
      `).join('')}

      <!-- Vascular Markings / Bronchovascular Tree -->
      <path d="M 170 200 Q 130 180 110 150 M 170 200 Q 140 230 110 250 M 170 200 Q 150 270 130 310" stroke="rgba(255,255,255,0.25)" stroke-width="2" fill="none" filter="blur(1px)"/>
      <path d="M 230 200 Q 270 180 290 150 M 230 200 Q 260 230 290 250 M 230 200 Q 250 270 270 310" stroke="rgba(255,255,255,0.25)" stroke-width="2" fill="none" filter="blur(1px)"/>

      <!-- Pathology Overlays -->
      ${pathologyOverlay}

      <!-- Lead Markers / DICOM overlay text -->
      <text x="25" y="40" fill="#a0aec0" font-family="monospace" font-size="12" font-weight="bold">R (PORTABLE PA)</text>
      <text x="25" y="420" fill="#718096" font-family="monospace" font-size="10">DICOM 16-BIT 1024x1024</text>
      <text x="280" y="420" fill="#718096" font-family="monospace" font-size="10">MEDVISION AI LAB</text>
    </svg>
  `;

  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

// Generate corresponding Grad-CAM SVG heatmap
function createGradCamSvg(type: 'normal' | 'pneumonia' | 'cardiomegaly' | 'effusion' | 'covid'): string {
  let heatmapContent = '';

  if (type === 'pneumonia') {
    // Focal intense red activation over right lower lung opacity
    heatmapContent = `
      <radialGradient id="gradPneumonia" cx="58%" cy="62%" r="22%">
        <stop offset="0%" stop-color="rgba(255,0,0,0.85)"/>
        <stop offset="40%" stop-color="rgba(255,165,0,0.7)"/>
        <stop offset="70%" stop-color="rgba(255,255,0,0.5)"/>
        <stop offset="100%" stop-color="rgba(0,0,255,0)"/>
      </radialGradient>
      <circle cx="232" cy="275" r="75" fill="url(#gradPneumonia)"/>
      <rect x="180" y="220" width="105" height="110" fill="none" stroke="#ef4444" stroke-width="2" stroke-dasharray="4 2"/>
      <text x="185" y="215" fill="#ef4444" font-family="sans-serif" font-size="11" font-weight="bold">ROI: Right Lobe Infiltrate (0.92)</text>
    `;
  } else if (type === 'cardiomegaly') {
    heatmapContent = `
      <radialGradient id="gradCardio" cx="55%" cy="65%" r="30%">
        <stop offset="0%" stop-color="rgba(255,0,0,0.85)"/>
        <stop offset="50%" stop-color="rgba(255,140,0,0.65)"/>
        <stop offset="80%" stop-color="rgba(0,255,128,0.3)"/>
        <stop offset="100%" stop-color="rgba(0,0,255,0)"/>
      </radialGradient>
      <ellipse cx="220" cy="290" rx="90" ry="70" fill="url(#gradCardio)"/>
      <rect x="140" y="220" width="150" height="130" fill="none" stroke="#f59e0b" stroke-width="2" stroke-dasharray="4 2"/>
      <text x="145" y="215" fill="#f59e0b" font-family="sans-serif" font-size="11" font-weight="bold">ROI: Enlarged Cardiac Shadow (CTR > 0.55)</text>
    `;
  } else if (type === 'effusion') {
    heatmapContent = `
      <radialGradient id="gradEffusion" cx="62%" cy="78%" r="20%">
        <stop offset="0%" stop-color="rgba(255,0,0,0.9)"/>
        <stop offset="50%" stop-color="rgba(255,120,0,0.7)"/>
        <stop offset="85%" stop-color="rgba(0,180,255,0.3)"/>
        <stop offset="100%" stop-color="rgba(0,0,255,0)"/>
      </radialGradient>
      <ellipse cx="245" cy="345" rx="60" ry="40" fill="url(#gradEffusion)"/>
      <rect x="190" y="300" width="110" height="75" fill="none" stroke="#ef4444" stroke-width="2" stroke-dasharray="4 2"/>
      <text x="195" y="295" fill="#ef4444" font-family="sans-serif" font-size="11" font-weight="bold">ROI: Costophrenic Angle Blunting</text>
    `;
  } else if (type === 'covid') {
    heatmapContent = `
      <radialGradient id="gradCovid1" cx="30%" cy="58%" r="20%">
        <stop offset="0%" stop-color="rgba(255,0,0,0.85)"/>
        <stop offset="50%" stop-color="rgba(255,160,0,0.6)"/>
        <stop offset="100%" stop-color="rgba(0,0,255,0)"/>
      </radialGradient>
      <radialGradient id="gradCovid2" cx="68%" cy="60%" r="22%">
        <stop offset="0%" stop-color="rgba(255,0,0,0.88)"/>
        <stop offset="50%" stop-color="rgba(255,160,0,0.6)"/>
        <stop offset="100%" stop-color="rgba(0,0,255,0)"/>
      </radialGradient>
      <circle cx="120" cy="250" r="65" fill="url(#gradCovid1)"/>
      <circle cx="265" cy="260" r="70" fill="url(#gradCovid2)"/>
      <rect x="70" y="195" width="225" height="145" fill="none" stroke="#ef4444" stroke-width="2" stroke-dasharray="4 2"/>
      <text x="75" y="190" fill="#ef4444" font-family="sans-serif" font-size="11" font-weight="bold">ROI: Bilateral Peripheral Opacities</text>
    `;
  } else {
    // Normal: Diffuse minimal activation centered on anatomical landmarks
    heatmapContent = `
      <radialGradient id="gradNormal" cx="50%" cy="50%" r="40%">
        <stop offset="0%" stop-color="rgba(0,120,255,0.2)"/>
        <stop offset="100%" stop-color="rgba(0,0,0,0)"/>
      </radialGradient>
      <circle cx="200" cy="220" r="140" fill="url(#gradNormal)"/>
    `;
  }

  const baseSvg = createChestXraySvg(type);
  // Decode base svg string, inject heatmap before final text
  const decoded = decodeURIComponent(baseSvg.replace('data:image/svg+xml;utf8,', ''));
  const injected = decoded.replace('<!-- Pathology Overlays -->', `<!-- Pathology Overlays -->\n${heatmapContent}`);

  return `data:image/svg+xml;utf8,${encodeURIComponent(injected)}`;
}

export const SAMPLE_XRAYS: SampleXray[] = [
  {
    id: 'sample-pneumonia-01',
    title: 'Acute Bacterial Pneumonia',
    subtitle: 'Right Lower Lobe Consolidation',
    category: 'Pneumonia',
    description: 'Patient presented with high fever (38.9°C), productive cough, and shortness of breath. X-ray shows right mid-to-lower lung field opacity.',
    svgDataUrl: createChestXraySvg('pneumonia'),
    heatmapSvgUrl: createGradCamSvg('pneumonia'),
    sampleResult: {
      imageName: 'chest_xray_pneumonia_rll.dcm',
      originalImageUrl: createChestXraySvg('pneumonia'),
      heatmapOverlayUrl: createGradCamSvg('pneumonia'),
      claheApplied: true,
      noiseRemovalApplied: true,
      modelUsed: 'DenseNet121-CheXNet (PyTorch)',
      inferenceTimeMs: 142,
      diseases: [
        { disease: 'Pneumonia', probability: 0.94, severityContribution: 0.45, category: 'infection', description: 'Consolidation and inflammatory exudate in lung parenchyma.' },
        { disease: 'Lung Opacity', probability: 0.88, severityContribution: 0.30, category: 'lung_opacity', description: 'Increased radiodensity consistent with focal air-space filling.' },
        { disease: 'Atelectasis', probability: 0.32, severityContribution: 0.10, category: 'structural', description: 'Mild volume loss secondary to bronchial mucus plugging.' },
        { disease: 'Pleural Effusion', probability: 0.28, severityContribution: 0.10, category: 'pleural', description: 'Small reactive parapneumonic fluid accumulation.' },
        { disease: 'Edema', probability: 0.08, severityContribution: 0.05, category: 'lung_opacity', description: 'No signs of diffuse pulmonary congestion.' },
        { disease: 'Cardiomegaly', probability: 0.05, severityContribution: 0.0, category: 'structural', description: 'Cardiac silhouette size within normal limits.' },
        { disease: 'COVID-19', probability: 0.12, severityContribution: 0.0, category: 'infection', description: 'Low likelihood of viral ground-glass distribution.' },
        { disease: 'Tuberculosis', probability: 0.06, severityContribution: 0.0, category: 'infection', description: 'No apical cavitary lesions or miliary pattern.' },
        { disease: 'Pneumothorax', probability: 0.02, severityContribution: 0.0, category: 'pleural', description: 'No visible visceral pleural edge or free intrapleural air.' },
        { disease: 'No Finding', probability: 0.03, severityContribution: 0.0, category: 'normal', description: 'Pathological findings detected.' },
      ],
      topDiagnosis: 'Pneumonia',
      topConfidence: 0.94,
      severity: 'High',
      severityScore: 82,
      gradCamRegions: [
        {
          name: 'Right Lower Zone Infiltrate',
          intensity: 94,
          bbox: { x: 180, y: 220, width: 105, height: 110, label: 'Consolidation', confidence: 0.94 },
          interpretation: 'High Grad-CAM density focused over right paracardiac and lower lobe region corresponding to dense airspace consolidation.',
        },
      ],
      report: {
        patientId: 'PAT-883921',
        patientAge: 54,
        patientSex: 'M',
        studyDate: '2026-08-01',
        indication: 'Fever, acute cough, purulent sputum, and right pleuritic chest pain.',
        technique: 'Single PA view upright chest radiograph.',
        findings: [
          'LUNGS: Dense focal air-space opacity demonstrated in the right lower lung zone with air bronchograms, highly suspicious for bacterial lobar pneumonia.',
          'PLEURA: Minimal blunting at the right costophrenic sulcus consistent with a small reactive parapneumonic pleural effusion. Left pleural space clear.',
          'CARDIOMEDIASTINAL: Heart size is within normal limits (Cardiothoracic Ratio < 0.50). Mediastinal contours and hila are unremarkable.',
          'BONES & SOFT TISSUES: Intact osseous thoracic cage. No acute rib fractures.',
        ],
        impression: '1. Dense right lower lobe pneumonia with focal consolidation.\n2. Trace reactive right pleural effusion.',
        recommendations: [
          'Correlate clinically with sputum cultures, CBC, and inflammatory markers (CRP/ESR).',
          'Initiate empirically targeted antibiotic therapy as clinically indicated.',
          'Recommend follow-up chest radiograph in 4-6 weeks post-treatment to verify complete radiographic resolution.',
        ],
        disclaimer: 'Not for clinical diagnosis. Educational and research demonstration purposes only.',
      },
      keyMetrics: {
        snr: 28.4,
        resolution: '1024x1024',
        meanIntensity: 112.4,
        contrastRatio: 4.8,
      },
    },
  },
  {
    id: 'sample-covid19-02',
    title: 'Bilateral Viral Pneumonia (COVID-19 Pattern)',
    subtitle: 'Peripheral Ground-Glass Opacities',
    category: 'COVID-19',
    description: 'Patient presented with dyspnea, hypoxia (SpO2 89%), and loss of taste/smell. Chest X-ray demonstrates classic bilateral multifocal peripheral GGOs.',
    svgDataUrl: createChestXraySvg('covid'),
    heatmapSvgUrl: createGradCamSvg('covid'),
    sampleResult: {
      imageName: 'chest_xray_covid_bilateral.dcm',
      originalImageUrl: createChestXraySvg('covid'),
      heatmapOverlayUrl: createGradCamSvg('covid'),
      claheApplied: true,
      noiseRemovalApplied: false,
      modelUsed: 'EfficientNet-B3 (PyTorch + ViT Ensemble)',
      inferenceTimeMs: 168,
      diseases: [
        { disease: 'COVID-19', probability: 0.91, severityContribution: 0.40, category: 'infection', description: 'Multifocal bilateral ground-glass opacities with peripheral predominance.' },
        { disease: 'Pneumonia', probability: 0.85, severityContribution: 0.30, category: 'infection', description: 'Bilateral viral pulmonary infiltrates.' },
        { disease: 'Lung Opacity', probability: 0.89, severityContribution: 0.20, category: 'lung_opacity', description: 'Widespread bilateral parenchymal haziness.' },
        { disease: 'Edema', probability: 0.24, severityContribution: 0.05, category: 'lung_opacity', description: 'Intercellular exudate vs early capillary leak.' },
        { disease: 'Atelectasis', probability: 0.18, severityContribution: 0.05, category: 'structural', description: 'Subsegmental basilar linear bands.' },
        { disease: 'Pleural Effusion', probability: 0.10, severityContribution: 0.0, category: 'pleural', description: 'No gross pleural effusion.' },
        { disease: 'Cardiomegaly', probability: 0.06, severityContribution: 0.0, category: 'structural', description: 'Normal heart size.' },
        { disease: 'Tuberculosis', probability: 0.04, severityContribution: 0.0, category: 'infection', description: 'Unlikely distribution.' },
        { disease: 'Pneumothorax', probability: 0.01, severityContribution: 0.0, category: 'pleural', description: 'No pneumothorax.' },
        { disease: 'No Finding', probability: 0.02, severityContribution: 0.0, category: 'normal', description: 'Pathological findings detected.' },
      ],
      topDiagnosis: 'COVID-19 / Viral Pneumonia',
      topConfidence: 0.91,
      severity: 'Critical',
      severityScore: 88,
      gradCamRegions: [
        {
          name: 'Bilateral Peripheral Lung Fields',
          intensity: 91,
          bbox: { x: 70, y: 195, width: 225, height: 145, label: 'Bilateral GGO', confidence: 0.91 },
          interpretation: 'Diffuse multi-focal heatmap activations in peripheral and basilar distribution characteristic of viral pneumonia.',
        },
      ],
      report: {
        patientId: 'PAT-772109',
        patientAge: 62,
        patientSex: 'F',
        studyDate: '2026-08-01',
        indication: 'Shortness of breath, persistent fever, SpO2 89% on room air.',
        technique: 'Upright Portable AP View.',
        findings: [
          'LUNGS: Multifocal, peripheral, and basal ground-glass opacities noted in both left and right lung zones.',
          'CARDIOMEDIASTINAL: Normal cardiac silhouette and thoracic aortic contour.',
          'PLEURA: No pleural effusions or pneumothorax.',
        ],
        impression: '1. Multifocal bilateral lung opacities with lower zone and peripheral distribution, strongly suggestive of COVID-19 pneumonitis.\n2. High risk of progressive respiratory distress.',
        recommendations: [
          'Urgent arterial blood gas analysis and clinical oxygen saturation monitoring.',
          'Correlate with RT-PCR viral testing and baseline inflammatory laboratory panel.',
        ],
        disclaimer: 'Not for clinical diagnosis. Educational and research demonstration purposes only.',
      },
      keyMetrics: {
        snr: 26.1,
        resolution: '1024x1024',
        meanIntensity: 104.2,
        contrastRatio: 4.2,
      },
    },
  },
  {
    id: 'sample-cardiomegaly-03',
    title: 'Cardiomegaly with Mild Congestion',
    subtitle: 'Enlarged Cardiac Silhouette',
    category: 'Cardiomegaly',
    description: 'Patient with chronic hypertension and dyspnea on exertion. X-ray demonstrates cardiothoracic ratio exceeding 0.58.',
    svgDataUrl: createChestXraySvg('cardiomegaly'),
    heatmapSvgUrl: createGradCamSvg('cardiomegaly'),
    sampleResult: {
      imageName: 'chest_xray_cardiomegaly_pa.dcm',
      originalImageUrl: createChestXraySvg('cardiomegaly'),
      heatmapOverlayUrl: createGradCamSvg('cardiomegaly'),
      claheApplied: false,
      noiseRemovalApplied: false,
      modelUsed: 'ConvNeXt-Base (PyTorch)',
      inferenceTimeMs: 155,
      diseases: [
        { disease: 'Cardiomegaly', probability: 0.96, severityContribution: 0.50, category: 'structural', description: 'Marked enlargement of the cardiac silhouette with transverse diameter > 55%.' },
        { disease: 'Edema', probability: 0.42, severityContribution: 0.25, category: 'lung_opacity', description: 'Mild Kerley B lines and peribronchial cuffing.' },
        { disease: 'Pleural Effusion', probability: 0.35, severityContribution: 0.15, category: 'pleural', description: 'Small bilateral blunting at costophrenic angles.' },
        { disease: 'Lung Opacity', probability: 0.28, severityContribution: 0.10, category: 'lung_opacity', description: 'Vascular prominence in upper lobes (cephalization).' },
        { disease: 'Atelectasis', probability: 0.20, severityContribution: 0.0, category: 'structural', description: 'Compressive basilar subsegmental atelectasis.' },
        { disease: 'Pneumonia', probability: 0.11, severityContribution: 0.0, category: 'infection', description: 'No acute focal consolidation.' },
        { disease: 'COVID-19', probability: 0.03, severityContribution: 0.0, category: 'infection', description: 'No viral features.' },
        { disease: 'Tuberculosis', probability: 0.02, severityContribution: 0.0, category: 'infection', description: 'Negative.' },
        { disease: 'Pneumothorax', probability: 0.01, severityContribution: 0.0, category: 'pleural', description: 'Negative.' },
        { disease: 'No Finding', probability: 0.02, severityContribution: 0.0, category: 'normal', description: 'Pathology detected.' },
      ],
      topDiagnosis: 'Cardiomegaly',
      topConfidence: 0.96,
      severity: 'Moderate',
      severityScore: 68,
      gradCamRegions: [
        {
          name: 'Cardiac Silhouette Boundary',
          intensity: 96,
          bbox: { x: 140, y: 220, width: 150, height: 130, label: 'Enlarged Cardiac Contour', confidence: 0.96 },
          interpretation: 'Grad-CAM strongly activates along the left ventricle and apical cardiac margin confirming enlargement.',
        },
      ],
      report: {
        patientId: 'PAT-334182',
        patientAge: 68,
        patientSex: 'M',
        studyDate: '2026-08-01',
        indication: 'Hypertension, progressive exertional shortness of breath, bilateral lower extremity edema.',
        technique: 'Standard PA and Lateral upright chest radiograph.',
        findings: [
          'CARDIOMEDIASTINAL: Significant enlargement of the cardiac silhouette (CTR = 0.58). Prominent left ventricular apex and upper zone vascular cephalization.',
          'LUNGS: Mild interstitial prominence without frank pulmonary alveolar edema.',
          'PLEURA: Minor blunting of costophrenic angles bilaterally.',
        ],
        impression: '1. Moderate to severe Cardiomegaly.\n2. Mild pulmonary venous congestion.',
        recommendations: [
          'Echocardiogram recommended to assess left ventricular ejection fraction (LVEF) and valvular function.',
          'Optimize antihypertensive and diuretic therapy.',
        ],
        disclaimer: 'Not for clinical diagnosis. Educational and research demonstration purposes only.',
      },
      keyMetrics: {
        snr: 31.2,
        resolution: '1024x1024',
        meanIntensity: 128.0,
        contrastRatio: 5.1,
      },
    },
  },
  {
    id: 'sample-normal-04',
    title: 'Unremarkable Chest Radiograph',
    subtitle: 'No Acute Pathological Findings',
    category: 'No Finding',
    description: 'Routine pre-employment screening chest radiograph. Clear lung fields, normal cardiomediastinal contour.',
    svgDataUrl: createChestXraySvg('normal'),
    heatmapSvgUrl: createGradCamSvg('normal'),
    sampleResult: {
      imageName: 'chest_xray_normal_screening.dcm',
      originalImageUrl: createChestXraySvg('normal'),
      heatmapOverlayUrl: createGradCamSvg('normal'),
      claheApplied: false,
      noiseRemovalApplied: false,
      modelUsed: 'DenseNet121-CheXNet (PyTorch)',
      inferenceTimeMs: 120,
      diseases: [
        { disease: 'No Finding', probability: 0.97, severityContribution: 0.0, category: 'normal', description: 'Lungs are clear without focal consolidation, pneumothorax, or effusion.' },
        { disease: 'Pneumonia', probability: 0.02, severityContribution: 0.0, category: 'infection', description: 'Negative.' },
        { disease: 'Cardiomegaly', probability: 0.03, severityContribution: 0.0, category: 'structural', description: 'Normal heart size.' },
        { disease: 'Lung Opacity', probability: 0.02, severityContribution: 0.0, category: 'lung_opacity', description: 'Negative.' },
        { disease: 'Pleural Effusion', probability: 0.01, severityContribution: 0.0, category: 'pleural', description: 'Negative.' },
        { disease: 'Edema', probability: 0.01, severityContribution: 0.0, category: 'lung_opacity', description: 'Negative.' },
        { disease: 'Atelectasis', probability: 0.02, severityContribution: 0.0, category: 'structural', description: 'Negative.' },
        { disease: 'COVID-19', probability: 0.01, severityContribution: 0.0, category: 'infection', description: 'Negative.' },
        { disease: 'Tuberculosis', probability: 0.01, severityContribution: 0.0, category: 'infection', description: 'Negative.' },
        { disease: 'Pneumothorax', probability: 0.00, severityContribution: 0.0, category: 'pleural', description: 'Negative.' },
      ],
      topDiagnosis: 'No Finding (Normal)',
      topConfidence: 0.97,
      severity: 'Low',
      severityScore: 4,
      gradCamRegions: [],
      report: {
        patientId: 'PAT-100293',
        patientAge: 31,
        patientSex: 'F',
        studyDate: '2026-08-01',
        indication: 'Routine pre-operative / pre-employment clearance.',
        technique: 'PA and Lateral chest views.',
        findings: [
          'LUNGS: Both lung fields are clear and well-expanded. No focal opacity, consolidation, pneumothorax, or pleural effusion.',
          'CARDIOMEDIASTINAL: Heart size and mediastinal contours are normal.',
          'BONES: Thoracic skeleton and soft tissues are unremarkable.',
        ],
        impression: 'Unremarkable chest radiograph. No acute cardiopulmonary disease.',
        recommendations: [
          'No immediate follow-up radiograph required.',
        ],
        disclaimer: 'Not for clinical diagnosis. Educational and research demonstration purposes only.',
      },
      keyMetrics: {
        snr: 34.8,
        resolution: '1024x1024',
        meanIntensity: 105.8,
        contrastRatio: 5.4,
      },
    },
  },
];

export const MODEL_BENCHMARKS: ModelMetadata[] = [
  {
    id: 'densenet121',
    name: 'DenseNet-121 (CheXNet Benchmark)',
    architecture: 'DenseNet121',
    parameters: '7.0M',
    auroc: 0.841,
    f1Score: 0.812,
    accuracy: 89.4,
    latencyFp32Ms: 42,
    latencyFp16Ms: 18,
    latencyOnnxMs: 12,
    flopsGiga: 2.8,
    description: 'Industry benchmark architecture introduced by Stanford CheXNet paper. Excellent feature reuse via dense connectivity.',
    recommendedFor: 'General multi-disease detection & high Grad-CAM spatial resolution.',
  },
  {
    id: 'efficientnet_b3',
    name: 'EfficientNet-B3',
    architecture: 'EfficientNet-B3',
    parameters: '12.2M',
    auroc: 0.865,
    f1Score: 0.835,
    accuracy: 91.2,
    latencyFp32Ms: 58,
    latencyFp16Ms: 24,
    latencyOnnxMs: 16,
    flopsGiga: 1.8,
    description: 'Compound scaling of depth, width, and resolution using depthwise separable convolutions.',
    recommendedFor: 'High accuracy with optimal parameter efficiency and fast mobile/edge inference.',
  },
  {
    id: 'convnext_base',
    name: 'ConvNeXt-Base',
    architecture: 'ConvNeXt-Base',
    parameters: '88.5M',
    auroc: 0.882,
    f1Score: 0.854,
    accuracy: 92.8,
    latencyFp32Ms: 98,
    latencyFp16Ms: 38,
    latencyOnnxMs: 28,
    flopsGiga: 15.4,
    description: 'Pure ConvNet modernized with Vision Transformer design choices (7x7 depthwise convs, LayerNorm, GELU).',
    recommendedFor: 'High precision clinical screening where GPU memory is available.',
  },
  {
    id: 'swin_transformer',
    name: 'Swin Transformer (Swin-B)',
    architecture: 'Swin-B',
    parameters: '88.0M',
    auroc: 0.889,
    f1Score: 0.862,
    accuracy: 93.5,
    latencyFp32Ms: 125,
    latencyFp16Ms: 45,
    latencyOnnxMs: 32,
    flopsGiga: 15.4,
    description: 'Hierarchical Vision Transformer using shifted windows for self-attention. Captures global thoracic spatial dependencies.',
    recommendedFor: 'Complex multi-focal pathologies (e.g. COVID-19 GGOs and miliary TB).',
  },
  {
    id: 'vit_base',
    name: 'Vision Transformer (ViT-B/16)',
    architecture: 'ViT-B/16',
    parameters: '86.6M',
    auroc: 0.878,
    f1Score: 0.849,
    accuracy: 92.1,
    latencyFp32Ms: 110,
    latencyFp16Ms: 42,
    latencyOnnxMs: 30,
    flopsGiga: 17.6,
    description: 'Pure self-attention transformer operating on 16x16 image patches without inductive bias.',
    recommendedFor: 'Attention-rollout visualization & cross-modal LLM integration.',
  },
];
