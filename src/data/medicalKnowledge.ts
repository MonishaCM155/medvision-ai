export interface DiseaseInfo {
  id: string;
  name: string;
  category: string;
  overview: string;
  causes: string[];
  symptoms: string[];
  riskFactors: string[];
  diagnosis: string[];
  treatment: string[];
  prevention: string[];
  complications: string[];
  recovery: string;
  anatomy: string;
  similarDiseases: string[];
  faqs: { question: string; answer: string }[];
  references: string[];
}

export const MEDICAL_KNOWLEDGE_BASE: Record<string, DiseaseInfo> = {
  Pneumonia: {
    id: 'pneumonia',
    name: 'Pneumonia',
    category: 'Infection',
    overview: 'Pneumonia is an inflammatory condition of the lung affecting primarily the small air sacs known as alveoli. It is typically caused by infection with viruses or bacteria and less commonly by other microorganisms.',
    causes: [
      'Bacterial infections (e.g., Streptococcus pneumoniae, Haemophilus influenzae)',
      'Viral infections (e.g., Influenza, RSV, SARS-CoV-2)',
      'Fungal pathogens (Cryptococcus, Pneumocystis jirovecii in immunocompromised individuals)',
      'Aspiration of food, liquid, vomit, or saliva into the lungs',
    ],
    symptoms: [
      'Cough, which may produce greenish, yellow or bloody mucus',
      'Fever, sweating and shaking chills',
      'Shortness of breath (dyspnea) and rapid, shallow breathing',
      'Sharp or stabbing chest pain that worsens when breathing deeply or coughing',
      'Loss of appetite, fatigue, and low energy',
    ],
    riskFactors: [
      'Age (infants under 2 and adults over 65)',
      'Chronic illness (Asthma, COPD, heart disease)',
      'Smoking or exposure to secondhand smoke',
      'Weakened or suppressed immune system',
      'Recent hospitalization or mechanical ventilation',
    ],
    diagnosis: [
      'Chest X-Ray: Shows focal or lobar consolidation/airspace opacities',
      'Pulse Oximetry: Evaluates blood oxygen saturation levels',
      'Blood Tests: Complete blood count (CBC) to check white blood cell count',
      'Sputum Culture: Identifies specific infectious organism',
    ],
    treatment: [
      'Antibiotic therapy for bacterial pneumonia (e.g., Macrolides, Beta-lactams, Fluoroquinolones)',
      'Antiviral medications for viral causes when indicated',
      'Supportive care: Oxygen therapy, hydration, rest, and antipyretics',
      'Hospitalization for severe cases with respiratory compromise',
    ],
    prevention: [
      'Pneumococcal vaccination (PCV15/PCV20 or PPSV23)',
      'Annual Influenza and COVID-19 vaccination',
      'Frequent hand hygiene and respiratory etiquette',
      'Smoking cessation',
    ],
    complications: [
      'Pleural effusion or empyema (pus in pleural space)',
      'Lung abscess formation',
      'Bacteremia and septic shock',
      'Acute Respiratory Distress Syndrome (ARDS)',
    ],
    recovery: 'Most young, healthy individuals recover within 1 to 3 weeks. Elderly or immunocompromised patients may take 6 to 8 weeks or longer to regain full strength.',
    anatomy: 'Involves terminal bronchioles and alveoli, where exudate fills alveolar spaces, blocking gas exchange across the alveolar-capillary membrane.',
    similarDiseases: ['COVID-19', 'Atelectasis', 'Pulmonary Edema', 'Tuberculosis', 'Lung Opacity'],
    faqs: [
      {
        question: 'Is pneumonia contagious?',
        answer: 'Bacterial and viral pneumonia can spread from person to person through respiratory droplets from coughing or sneezing.',
      },
      {
        question: 'How does an X-ray detect pneumonia?',
        answer: 'Infected fluid and pus fill the air sacs, appearing as dense white or hazy areas (consolidations) on chest radiograms compared to normal air-filled black lung tissue.',
      },
    ],
    references: [
      'American Thoracic Society / IDSA Guidelines on Community-Acquired Pneumonia (2019)',
      'WHO Global Health Estimates on Lower Respiratory Infections',
    ],
  },
  'COVID-19': {
    id: 'covid19',
    name: 'COVID-19 Pneumonia',
    category: 'Viral Infection',
    overview: 'SARS-CoV-2 infection leading to bilateral ground-glass opacities, interstitial inflammation, and progressive hypoxemic respiratory insufficiency.',
    causes: [
      'Infection with Severe Acute Respiratory Syndrome Coronavirus 2 (SARS-CoV-2)',
      'Binding to ACE2 receptors expressed on Type II pneumocytes in alveolar epithelium',
    ],
    symptoms: [
      'Fever, dry cough, severe fatigue',
      'Shortness of breath and desaturation',
      'Anosmia (loss of smell) and ageusia (loss of taste)',
      'Myalgias and systemic inflammatory symptoms',
    ],
    riskFactors: [
      'Advanced age and male sex',
      'Comorbidities: Diabetes, hypertension, obesity, cardiovascular disease',
      'Immunosuppressive conditions',
    ],
    diagnosis: [
      'Chest Radiograph: Peripheral, bilateral, predominantly lower zone ground-glass opacities',
      'RT-PCR or Rapid Antigen SARS-CoV-2 testing',
      'Chest CT: Subpleural ground-glass opacities with crazy-paving patterns',
    ],
    treatment: [
      'Antiviral agents (Remdesivir, Nirmatrelvir/Ritonavir)',
      'Corticosteroids (Dexamethasone) for oxygen-dependent patients',
      'Immunomodulators (Tocilizumab, Baricitinib) for severe hyperinflammation',
      'Supplemental oxygen or high-flow nasal cannula',
    ],
    prevention: ['mRNA and protein subunit vaccines', 'Well-fitting respirators (N95/KN95)', 'Indoor ventilation improvement'],
    complications: ['ARDS', 'Thromboembolism and pulmonary embolism', 'Cytokine storm syndrome', 'Long COVID post-acute sequelae'],
    recovery: 'Varies from 2 weeks in mild cases to several months in severe ARDS survivors with residual pulmonary fibrosis.',
    anatomy: 'Affects peripheral, subpleural parenchymal regions bilaterally, progressing to diffuse alveolar damage.',
    similarDiseases: ['Pneumonia', 'Lung Opacity', 'Pulmonary Edema', 'ARDS'],
    faqs: [
      {
        question: 'How does COVID-19 differ from bacterial pneumonia on X-ray?',
        answer: 'COVID-19 typically presents with bilateral, peripheral, subpleural ground-glass opacities rather than focal dense lobar consolidation seen in classic bacterial pneumonia.',
      },
    ],
    references: ['NIH COVID-19 Treatment Guidelines', 'RSNA Consensus Statement on Chest Imaging in COVID-19'],
  },
  Cardiomegaly: {
    id: 'cardiomegaly',
    name: 'Cardiomegaly',
    category: 'Structural / Cardiovascular',
    overview: 'Enlargement of the cardiac silhouette where the Cardiothoracic Ratio (CTR) exceeds 0.50 on a standard posteroanterior (PA) chest radiograph.',
    causes: [
      'Chronic hypertension leading to left ventricular hypertrophy',
      'Ischemic heart disease and prior myocardial infarction',
      'Valvular heart disease (aortic stenosis, mitral regurgitation)',
      'Dilated or hypertrophic cardiomyopathy',
      'Pericardial effusion (pericardial sac fluid accumulation)',
    ],
    symptoms: [
      'Dyspnea on exertion or orthopnea (difficulty breathing when lying flat)',
      'Lower extremity edema (swelling in feet and ankles)',
      'Fatigue and reduced exercise tolerance',
      'Palpitations and irregular heartbeat',
    ],
    riskFactors: ['Uncontrolled high blood pressure', 'Coronary artery disease', 'Diabetes mellitus', 'Obesity and sleep apnea'],
    diagnosis: [
      'Chest X-ray: Cardiothoracic ratio > 50%',
      'Echocardiogram: Measures ejection fraction, chamber sizes, and valvular function',
      'Electrocardiogram (ECG): Checks for hypertrophy or arrhythmia patterns',
    ],
    treatment: [
      'ACE inhibitors / ARBs / ARNIs for ventricular remodeling',
      'Beta-blockers and aldosterone antagonists',
      'Diuretics (Furosemide) for fluid congestion relief',
      'Surgical valve replacement or revascularization if structural',
    ],
    prevention: ['Strict blood pressure control (<130/80 mmHg)', 'Low-sodium heart-healthy diet', 'Regular physical activity'],
    complications: ['Congestive heart failure (CHF)', 'Cardiac arrhythmias', 'Sudden cardiac arrest'],
    recovery: 'Reversible in certain early stages with medication; chronic structural enlargement requires lifelong cardiac disease management.',
    anatomy: 'Heart enlargement expanding into the lower left hemithorax on PA chest projection.',
    similarDiseases: ['Pericardial Effusion', 'Pulmonary Edema', 'Pleural Effusion'],
    faqs: [
      {
        question: 'What is cardiothoracic ratio (CTR)?',
        answer: 'CTR is the maximum horizontal width of the heart divided by the inner thoracic diameter. A ratio greater than 0.5 indicates cardiomegaly.',
      },
    ],
    references: ['ACC/AHA Guidelines for the Management of Heart Failure'],
  },
  'Pleural Effusion': {
    id: 'pleural_effusion',
    name: 'Pleural Effusion',
    category: 'Pleural',
    overview: 'Abnormal collection of fluid in the pleural space between the visceral and parietal pleura surrounding the lungs.',
    causes: [
      'Congestive Heart Failure (Transudative)',
      'Pneumonia or empyema (Exudative)',
      'Malignancy (lung or metastatic cancer)',
      'Pulmonary embolism or liver cirrhosis',
    ],
    symptoms: ['Sharp chest pain worsened by inspiration (pleuritic pain)', 'Shortness of breath', 'Dry cough'],
    riskFactors: ['Pre-existing pneumonia or pulmonary infection', 'Heart failure history', 'Malignancy', 'Renal disease'],
    diagnosis: [
      'Chest X-Ray: Blunting of costophrenic angles, meniscus sign, or fluid tracking in fissures',
      'Thoracic Ultrasound: Confirms fluid pocket and guides thoracentesis',
      'Diagnostic Thoracentesis: Fluid analysis using Light’s Criteria',
    ],
    treatment: ['Thoracentesis (fluid drainage)', 'Treatment of underlying cause', 'Chest tube insertion for large/infected effusions (empyema)'],
    prevention: ['Prompt antibiotic therapy for pneumonia', 'Management of heart failure'],
    complications: ['Empyema (infected pleural space)', 'Lung entrapment and fibrous scarring'],
    recovery: 'Resolves rapidly following successful drainage and treatment of underlying etiology.',
    anatomy: 'Pleural space surrounding lung parenchyma, settling dependently at the costophrenic recesses.',
    similarDiseases: ['Atelectasis', 'Pulmonary Edema', 'Pneumoperitoneum'],
    faqs: [
      {
        question: 'What is a meniscus sign on X-ray?',
        answer: 'It is a characteristic curved upper fluid line seen along the lateral chest wall caused by fluid accumulating in the pleural cavity.',
      },
    ],
    references: ['BTS Guidelines for the Management of Pleural Infection'],
  },
  Atelectasis: {
    id: 'atelectasis',
    name: 'Atelectasis',
    category: 'Structural',
    overview: 'Collapse or incomplete expansion of pulmonary parenchyma involving a lobe or segment, reducing gas exchange surface area.',
    causes: ['Mucus plugging in bronchial airways', 'Foreign body aspiration', 'Pleural effusion compression', 'Post-surgical shallow breathing'],
    symptoms: ['Mild shortness of breath', 'Decreased chest wall movement', 'Subtle tachypnea'],
    riskFactors: ['Recent thoracic or abdominal surgery', 'Prolonged bed rest', 'Heavy sedation', 'Airway obstruction'],
    diagnosis: ['Chest X-Ray: Subsegmental linear opacities, volume loss with elevation of hemidiaphragm or fissural displacement'],
    treatment: ['Incentive spirometry and deep breathing exercises', 'Chest physiotherapy and early ambulation', 'Bronchoscopy for mucus removal'],
    prevention: ['Incentive spirometer usage post-operatively', 'Adequate pain management after chest/abdominal procedures'],
    complications: ['Superimposed secondary bacterial pneumonia', 'Hypoxemia'],
    recovery: 'Usually resolves quickly (hours to days) with lung expansion exercises and clearance of bronchial obstruction.',
    anatomy: 'Collapsing of bronchial alveoli causing regional lung tissue opacity and displacement of adjacent structures.',
    similarDiseases: ['Pneumonia', 'Pleural Effusion', 'Lung Opacity'],
    faqs: [
      {
        question: 'Is atelectasis the same as a collapsed lung?',
        answer: 'Atelectasis refers to partial alveolar collapse within lung tissue, whereas a complete collapsed lung usually refers to a pneumothorax (air in pleural space).',
      },
    ],
    references: ['Radiopaedia Pulmonary Atelectasis Review'],
  },
};

export const MEDICAL_GLOSSARY: Record<string, string> = {
  'Grad-CAM': 'Gradient-weighted Class Activation Mapping: An explainable AI algorithm that visualizes spatial regions in an image that influenced a neural network decision.',
  'Consolidation': 'A region of lung tissue that has filled with liquid rather than air, appearing opaque white on X-rays.',
  'Ground-Glass Opacity': 'A hazy area of increased radiodensity in the lung through which underlying bronchial and vascular structures remain visible.',
  'Cardiothoracic Ratio (CTR)': 'The ratio of transverse cardiac diameter to transverse thoracic diameter. Normal is <= 0.50.',
  'Costophrenic Angle': 'The sharp angle where the chest wall meets the diaphragm. Sharp in healthy lungs, blunted when fluid collects.',
  'SNR': 'Signal-to-Noise Ratio: A technical quality metric measuring image clarity and contrast relative to background noise.',
  'Atelectasis': 'Collapse or incomplete expansion of part or all of a lung due to airway obstruction or shallow breathing.',
  'Thoracentesis': 'A procedure to remove fluid or air from around the lungs using a thin needle.',
  'Aspiration': 'Accidental breathing of food, liquid, or saliva into the windpipe and lungs.',
  'Infiltrate': 'A substance denser than air, such as pus, blood, or protein, lingering in lung parenchyma.',
};
