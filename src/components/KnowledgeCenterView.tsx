import React, { useState } from 'react';
import {
  BookOpen, Search, Stethoscope, ChevronRight, Bookmark, Filter,
  ShieldCheck, HelpCircle, ExternalLink, Activity, FileText, Sparkles
} from 'lucide-react';
import { MEDICAL_KNOWLEDGE_BASE, MEDICAL_GLOSSARY, DiseaseInfo } from '../data/medicalKnowledge';

export const KnowledgeCenterView: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'encyclopedia' | 'glossary' | 'faqs'>('encyclopedia');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedDisease, setSelectedDisease] = useState<DiseaseInfo | null>(
    Object.values(MEDICAL_KNOWLEDGE_BASE)[0]
  );

  const diseaseList = Object.values(MEDICAL_KNOWLEDGE_BASE);

  // Filter diseases by search query
  const filteredDiseases = diseaseList.filter(
    (d) =>
      d.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      d.category.toLowerCase().includes(searchQuery.toLowerCase()) ||
      d.overview.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Filter glossary
  const filteredGlossary = Object.entries(MEDICAL_GLOSSARY).filter(
    ([term, def]) =>
      term.toLowerCase().includes(searchQuery.toLowerCase()) ||
      def.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="space-y-6 text-slate-100">
      {/* Top Banner */}
      <div className="bg-slate-900/90 p-5 rounded-xl border border-slate-800 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-indigo-400" />
              <h1 className="text-lg font-bold text-white tracking-tight">
                Medical Knowledge Hub &amp; Pathology Encyclopedia
              </h1>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Comprehensive peer-reviewed radiologic guides, anatomical mechanisms, and disease differentials
            </p>
          </div>

          {/* Tab Switcher */}
          <div className="bg-slate-950 p-1 rounded-lg border border-slate-800 flex text-xs shrink-0">
            <button
              onClick={() => setActiveTab('encyclopedia')}
              className={`px-3 py-1.5 rounded font-medium transition-colors cursor-pointer ${
                activeTab === 'encyclopedia' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'
              }`}
            >
              Encyclopedia ({diseaseList.length})
            </button>
            <button
              onClick={() => setActiveTab('glossary')}
              className={`px-3 py-1.5 rounded font-medium transition-colors cursor-pointer ${
                activeTab === 'glossary' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'
              }`}
            >
              Glossary
            </button>
            <button
              onClick={() => setActiveTab('faqs')}
              className={`px-3 py-1.5 rounded font-medium transition-colors cursor-pointer ${
                activeTab === 'faqs' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'
              }`}
            >
              AI &amp; X-Ray FAQs
            </button>
          </div>
        </div>

        {/* Search Bar */}
        <div className="relative">
          <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search diseases, symptoms, treatments, or medical terminology..."
            className="w-full bg-slate-950 border border-slate-800 rounded-lg pl-10 pr-4 py-2 text-xs text-white placeholder-slate-500 focus:outline-hidden focus:border-indigo-500 transition-colors"
          />
        </div>
      </div>

      {/* ENCYCLOPEDIA TAB */}
      {activeTab === 'encyclopedia' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Disease Selector Sidebar */}
          <div className="space-y-2 lg:col-span-1">
            <h2 className="text-xs font-bold text-slate-400 uppercase tracking-wider px-1">
              Pathology Catalog
            </h2>
            <div className="space-y-1.5 max-h-[620px] overflow-y-auto pr-1">
              {filteredDiseases.map((disease) => {
                const isSelected = selectedDisease?.id === disease.id;
                return (
                  <button
                    key={disease.id}
                    onClick={() => setSelectedDisease(disease)}
                    className={`w-full text-left p-3 rounded-xl border transition-all cursor-pointer ${
                      isSelected
                        ? 'bg-indigo-950/80 border-indigo-600/80 text-white shadow-sm'
                        : 'bg-slate-900/80 hover:bg-slate-800/80 border-slate-800 text-slate-300'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold">{disease.name}</span>
                      <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-slate-800 text-slate-400 border border-slate-700">
                        {disease.category}
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-400 line-clamp-2 mt-1 font-sans">
                      {disease.overview}
                    </p>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Selected Disease Detailed Viewer */}
          {selectedDisease ? (
            <div className="lg:col-span-2 bg-slate-900/90 p-5 rounded-xl border border-slate-800 space-y-5">
              {/* Header */}
              <div className="border-b border-slate-800 pb-4 flex items-start justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono bg-indigo-950 text-indigo-300 px-2 py-0.5 rounded border border-indigo-800">
                      {selectedDisease.category}
                    </span>
                  </div>
                  <h2 className="text-xl font-bold text-white mt-1">{selectedDisease.name}</h2>
                </div>
              </div>

              {/* Overview */}
              <div className="space-y-1">
                <h3 className="text-xs font-bold text-indigo-400 uppercase tracking-wider">Overview</h3>
                <p className="text-xs text-slate-300 leading-relaxed">{selectedDisease.overview}</p>
              </div>

              {/* Pathophysiology & Mechanisms */}
              <div className="space-y-1">
                <h3 className="text-xs font-bold text-indigo-400 uppercase tracking-wider">
                  Anatomy &amp; Mechanism
                </h3>
                <p className="text-xs text-slate-300 leading-relaxed bg-slate-950 p-3 rounded-lg border border-slate-800 font-sans">
                  {selectedDisease.anatomy}
                </p>
              </div>

              {/* Symptoms & Causes Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 space-y-1.5">
                  <h4 className="text-xs font-bold text-emerald-400 flex items-center gap-1.5">
                    <Activity className="w-3.5 h-3.5" />
                    Key Clinical Symptoms
                  </h4>
                  <ul className="space-y-1 text-xs text-slate-300">
                    {selectedDisease.symptoms.map((s, idx) => (
                      <li key={idx} className="flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                        <span>{s}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 space-y-1.5">
                  <h4 className="text-xs font-bold text-amber-400 flex items-center gap-1.5">
                    <ShieldCheck className="w-3.5 h-3.5" />
                    Etiology &amp; Risk Factors
                  </h4>
                  <ul className="space-y-1 text-xs text-slate-300">
                    {selectedDisease.causes.map((c, idx) => (
                      <li key={idx} className="flex items-center gap-1.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                        <span>{c}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              {/* References */}
              <div className="border-t border-slate-800 pt-3 text-[11px] text-slate-400 space-y-1">
                <div className="font-bold text-slate-300">Clinical References:</div>
                <div className="flex flex-wrap gap-2">
                  {selectedDisease.references.map((ref, idx) => (
                    <span key={idx} className="bg-slate-950 px-2 py-0.5 rounded border border-slate-800 text-[10px]">
                      {ref}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="lg:col-span-2 bg-slate-900/90 p-8 rounded-xl border border-slate-800 flex flex-col items-center justify-center text-center text-slate-400 space-y-2">
              <Stethoscope className="w-8 h-8 text-indigo-400" />
              <p className="text-sm font-medium">Select a disease from the pathology catalog to view in-depth details.</p>
            </div>
          )}
        </div>
      )}

      {/* GLOSSARY TAB */}
      {activeTab === 'glossary' && (
        <div className="bg-slate-900/90 p-5 rounded-xl border border-slate-800 space-y-4">
          <h2 className="text-sm font-bold text-white">Radiology &amp; AI Medical Glossary</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {filteredGlossary.map(([term, def]) => (
              <div key={term} className="bg-slate-950 p-3 rounded-lg border border-slate-800 space-y-1">
                <div className="text-xs font-bold text-indigo-400 font-mono">{term}</div>
                <p className="text-xs text-slate-300 leading-relaxed">{def}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* FAQS TAB */}
      {activeTab === 'faqs' && (
        <div className="bg-slate-900/90 p-5 rounded-xl border border-slate-800 space-y-4">
          <h2 className="text-sm font-bold text-white">Frequently Asked Questions: AI in Medical Imaging</h2>
          <div className="space-y-3">
            <div className="bg-slate-950 p-4 rounded-lg border border-slate-800 space-y-1.5">
              <h3 className="text-xs font-bold text-indigo-300">How does Grad-CAM highlight disease locations?</h3>
              <p className="text-xs text-slate-300 leading-relaxed">
                Grad-CAM calculates the gradient of the target disease score with respect to the final convolutional feature maps of the DenseNet-121 architecture. High spatial activation zones highlight regions that contributed most to the prediction.
              </p>
            </div>
            <div className="bg-slate-950 p-4 rounded-lg border border-slate-800 space-y-1.5">
              <h3 className="text-xs font-bold text-indigo-300">Is this software FDA-approved for clinical use?</h3>
              <p className="text-xs text-slate-300 leading-relaxed">
                No. MedVision AI Enterprise is designed purely for academic research, peer education, and algorithm demonstration. All predictions require clinical correlation by a licensed radiologist.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

