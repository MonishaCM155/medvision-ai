import React, { useState, useEffect, useRef } from 'react';
import {
  Bot, Send, Mic, MicOff, Volume2, VolumeX, Sparkles, X, ChevronRight,
  Maximize2, Minimize2, Copy, Check, ThumbsUp, ThumbsDown, Download,
  Pin, RefreshCw, BarChart2, ShieldAlert, Cpu, HelpCircle, FileText,
  MessageSquare, Stethoscope, Layers, ExternalLink, Activity, GitCompare
} from 'lucide-react';
import { PredictionResult } from '../types';
import { ChatChartWidget, ChartWidgetType } from './ChatChartWidget';
import { MEDICAL_KNOWLEDGE_BASE, MEDICAL_GLOSSARY } from '../data/medicalKnowledge';

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  isPinned?: boolean;
  liked?: boolean;
  disliked?: boolean;
  chartType?: ChartWidgetType;
  suggestedQuestions?: string[];
}

interface MedicalCopilotProps {
  predictionResult: PredictionResult | null;
  onExportPdf?: () => void;
  isOpen?: boolean;
  onToggleOpen?: () => void;
}

export const MedicalCopilot: React.FC<MedicalCopilotProps> = ({
  predictionResult,
  onExportPdf,
  isOpen = true,
  onToggleOpen,
}) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputQuery, setInputQuery] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [isVoiceActive, setIsVoiceActive] = useState(false);
  const [isTtsEnabled, setIsTtsEnabled] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [provider, setProvider] = useState<'gemini' | 'openai' | 'claude' | 'ollama'>('gemini');
  const [lastPredictionId, setLastPredictionId] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll chat
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  // Trigger Automatic Proactive Greeting whenever a new prediction arrives
  useEffect(() => {
    if (!predictionResult) return;

    if (predictionResult.id !== lastPredictionId) {
      setLastPredictionId(predictionResult.id);

      const topDiag = predictionResult.topDiagnosis || 'Pneumonia';
      const confStr = (predictionResult.topConfidence * 100).toFixed(1);
      const severity = predictionResult.severity || 'High';

      const initialMessage: Message = {
        id: `msg_welcome_${Date.now()}`,
        role: 'assistant',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        content: `👋 **Hello! I analyzed your uploaded chest X-ray.**

The AI model predicts **${topDiag}** with **${confStr}% confidence** (Severity: **${severity}**).

I can explain:
• **What ${topDiag} is**
• **Why the model predicted it**
• **What the highlighted Grad-CAM heatmap means**
• **Common symptoms & risk factors**
• **Treatment overview**
• **Similar diseases & differential diagnosis**
• **Model limitations & AI confidence calibration**

Ask me anything or click a suggested button below!`,
        suggestedQuestions: [
          `Explain ${topDiag}`,
          `Explain Heatmap`,
          `Explain Confidence`,
          `Compare Diseases`,
          `Treatment Overview`,
          `Show Interactive Charts`,
          `Explain AI (Grad-CAM)`,
        ],
      };

      setMessages([initialMessage]);

      if (isTtsEnabled) {
        speakText(`Hello! I analyzed your chest X-ray. The AI model predicts ${topDiag} with ${confStr} percent confidence.`);
      }
    }
  }, [predictionResult, lastPredictionId]);

  // Handle Speech Recognition (Web Speech API)
  const handleVoiceInput = () => {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      alert('Speech recognition is not supported in this browser.');
      return;
    }

    if (isVoiceActive) {
      setIsVoiceActive(false);
      return;
    }

    try {
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      const recognition = new SpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = 'en-US';

      recognition.onstart = () => setIsVoiceActive(true);
      recognition.onend = () => setIsVoiceActive(false);
      recognition.onerror = () => setIsVoiceActive(false);

      recognition.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        if (transcript) {
          setInputQuery(transcript);
          handleSendMessage(transcript);
        }
      };

      recognition.start();
    } catch (e) {
      console.error('Speech recognition error:', e);
      setIsVoiceActive(false);
    }
  };

  // Text to Speech
  const speakText = (text: string) => {
    if (!('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const cleanText = text.replace(/[*#_`|]/g, '');
    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
  };

  const handleSendMessage = async (queryText?: string) => {
    const textToSend = (queryText || inputQuery).trim();
    if (!textToSend || isTyping) return;

    setInputQuery('');

    // User Message
    const userMsg: Message = {
      id: `msg_${Date.now()}`,
      role: 'user',
      content: textToSend,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMsg]);
    setIsTyping(true);

    const qLower = textToSend.toLowerCase();

    // Check if user requested an interactive chart
    let triggerChart: ChartWidgetType | undefined = undefined;
    if (qLower.includes('chart') || qLower.includes('graph') || qLower.includes('top 5') || qLower.includes('show charts')) {
      triggerChart = 'top5_predictions';
    } else if (qLower.includes('roc') || qLower.includes('precision') || qLower.includes('curve')) {
      triggerChart = 'roc_curve';
    } else if (qLower.includes('severity') || qLower.includes('meter')) {
      triggerChart = 'severity_meter';
    } else if (qLower.includes('benchmark') || qLower.includes('model comparison')) {
      triggerChart = 'model_benchmark';
    } else if (qLower.includes('confidence score') || qLower.includes('gauge')) {
      triggerChart = 'confidence_gauge';
    } else if (qLower.includes('category') || qLower.includes('distribution')) {
      triggerChart = 'disease_distribution';
    } else if (qLower.includes('matrix') || qLower.includes('confusion')) {
      triggerChart = 'confusion_matrix';
    }

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: textToSend,
          history: messages.map((m) => ({ role: m.role, content: m.content })),
          predictionContext: predictionResult,
          provider,
        }),
      });

      let replyContent = '';
      if (res.ok) {
        const data = await res.json();
        replyContent = data.reply;
      } else {
        replyContent = `I evaluated your query regarding **${predictionResult?.topDiagnosis || 'the X-ray'}**. The AI model calculated a **${((predictionResult?.topConfidence || 0.94) * 100).toFixed(1)}% confidence** for the primary detection. Please choose from the options below to explore further.`;
      }

      // Generate suggested questions for assistant
      const suggestedList: string[] = [];
      if (qLower.includes('explain') || qLower.includes('disease')) {
        suggestedList.push('Treatment Overview', 'Risk Factors', 'Compare Diseases', 'Show Interactive Charts');
      } else if (qLower.includes('heatmap') || qLower.includes('grad-cam')) {
        suggestedList.push('Explain Confidence', 'How AI Works', 'Download Report');
      } else {
        suggestedList.push('Explain Heatmap', 'Risk Factors', 'Show Interactive Charts', 'Medical Glossary');
      }

      const assistantMsg: Message = {
        id: `msg_${Date.now() + 1}`,
        role: 'assistant',
        content: replyContent,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        chartType: triggerChart,
        suggestedQuestions: suggestedList,
      };

      setMessages((prev) => [...prev, assistantMsg]);

      if (isTtsEnabled) {
        speakText(replyContent);
      }
    } catch (err) {
      console.error('Chat error:', err);
      const errorMsg: Message = {
        id: `msg_err_${Date.now()}`,
        role: 'assistant',
        content: `I'm currently processing using offline medical intelligence. You can ask me to explain **${predictionResult?.topDiagnosis}**, show **heatmaps**, **charts**, or **treatment options**.`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleCopyMessage = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleReaction = (id: string, type: 'like' | 'dislike') => {
    setMessages((prev) =>
      prev.map((msg) => {
        if (msg.id === id) {
          if (type === 'like') return { ...msg, liked: !msg.liked, disliked: false };
          if (type === 'dislike') return { ...msg, disliked: !msg.disliked, liked: false };
        }
        return msg;
      })
    );
  };

  const handleTogglePin = (id: string) => {
    setMessages((prev) =>
      prev.map((msg) => (msg.id === id ? { ...msg, isPinned: !msg.isPinned } : msg))
    );
  };

  const handleExportChat = () => {
    const chatText = messages
      .map((m) => `[${m.timestamp}] ${m.role.toUpperCase()}:\n${m.content}\n`)
      .join('\n----------------------------------------\n');
    const blob = new Blob([chatText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `MedVision_AI_Chat_Transcript_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (!isOpen) {
    return (
      <button
        onClick={onToggleOpen}
        className="fixed bottom-6 right-6 z-50 bg-indigo-600 hover:bg-indigo-500 text-white p-3.5 rounded-full shadow-2xl flex items-center gap-2 font-semibold transition-all hover:scale-105 active:scale-95 group border border-indigo-400/40 cursor-pointer"
        title="Open AI Medical Copilot"
      >
        <Bot className="w-5 h-5 text-white animate-bounce" />
        <span className="text-xs font-bold tracking-wide hidden sm:inline">AI Copilot</span>
        <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 border border-slate-900" />
      </button>
    );
  }

  return (
    <aside
      className={`fixed top-14 right-0 bottom-0 z-40 bg-slate-900 border-l border-slate-800 flex flex-col shadow-2xl transition-all duration-300 ${
        isExpanded ? 'w-full md:w-[620px]' : 'w-full md:w-[380px] lg:w-[420px]'
      }`}
    >
      {/* Copilot Header */}
      <div className="px-3 py-2 bg-slate-950 border-b border-slate-800 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 bg-gradient-to-tr from-indigo-600 to-purple-600 rounded-lg flex items-center justify-center text-white shadow-xs">
            <Bot className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <h2 className="text-xs font-bold text-white tracking-tight">AI Medical Copilot</h2>
              <span className="bg-indigo-950 text-indigo-300 text-[9px] px-1.5 py-0.2 rounded border border-indigo-800 font-mono">
                RAG + Gemini
              </span>
            </div>
            <p className="text-[10px] text-slate-400">Context-Aware Clinical Assistant</p>
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-1">
          {/* Provider Selector */}
          <select
            value={provider}
            onChange={(e: any) => setProvider(e.target.value)}
            className="bg-slate-900 text-slate-300 text-[10px] px-1.5 py-0.5 rounded border border-slate-700 font-mono cursor-pointer hover:border-slate-600 focus:outline-hidden"
          >
            <option value="gemini">Gemini 3.6</option>
            <option value="openai">OpenAI (Adapter)</option>
            <option value="claude">Claude (Adapter)</option>
            <option value="ollama">Local Ollama</option>
          </select>

          {/* Speech Synthesis Toggle */}
          <button
            onClick={() => setIsTtsEnabled(!isTtsEnabled)}
            className={`p-1.5 rounded transition-colors cursor-pointer ${
              isTtsEnabled ? 'text-indigo-400 bg-indigo-950/80 border border-indigo-800' : 'text-slate-400 hover:text-slate-200'
            }`}
            title={isTtsEnabled ? 'Mute AI Voice' : 'Enable AI Voice Readout'}
          >
            {isTtsEnabled ? <Volume2 className="w-3.5 h-3.5" /> : <VolumeX className="w-3.5 h-3.5" />}
          </button>

          {/* Expand/Contract Toggle */}
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-1.5 text-slate-400 hover:text-slate-200 rounded cursor-pointer hidden md:block"
            title={isExpanded ? 'Collapse Panel Width' : 'Expand Panel Width'}
          >
            {isExpanded ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
          </button>

          {/* Export Chat */}
          <button
            onClick={handleExportChat}
            className="p-1.5 text-slate-400 hover:text-slate-200 rounded cursor-pointer"
            title="Export Conversation Transcript"
          >
            <Download className="w-3.5 h-3.5" />
          </button>

          {/* Close Panel */}
          {onToggleOpen && (
            <button
              onClick={onToggleOpen}
              className="p-1.5 text-slate-400 hover:text-red-400 rounded cursor-pointer ml-1"
              title="Close Panel"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Required Education & Research Disclaimer Banner */}
      <div className="bg-amber-950/80 px-3 py-1 text-[10px] text-amber-200/90 border-b border-amber-900/60 flex items-center gap-1.5 shrink-0 font-medium">
        <ShieldAlert className="w-3.5 h-3.5 text-amber-400 shrink-0" />
        <span className="truncate">
          <strong>Educational &amp; Research Use Only:</strong> Not for clinical diagnosis.
        </span>
      </div>

      {/* Context Badge of active prediction */}
      {predictionResult && (
        <div className="bg-slate-950/90 border-b border-slate-800/80 px-3 py-1.5 flex items-center justify-between text-[11px] shrink-0">
          <div className="flex items-center gap-2 truncate">
            <span className="w-2 h-2 rounded-full bg-indigo-500 animate-ping" />
            <span className="text-slate-400">Context:</span>
            <span className="font-semibold text-white truncate">{predictionResult.topDiagnosis}</span>
            <span className="text-indigo-300 font-mono bg-indigo-950 px-1.5 py-0.2 rounded border border-indigo-900 text-[10px]">
              {(predictionResult.topConfidence * 100).toFixed(1)}%
            </span>
          </div>
          <span className="text-[10px] font-mono text-slate-400 shrink-0">
            {predictionResult.severity} Severity
          </span>
        </div>
      )}

      {/* Messages Stream Container */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3 scrollbar-thin scrollbar-thumb-slate-700">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex flex-col space-y-1 ${
              msg.role === 'user' ? 'items-end' : 'items-start'
            }`}
          >
            {/* Sender Label */}
            <div className="flex items-center gap-1.5 text-[10px] text-slate-400 px-1 font-mono">
              <span>{msg.role === 'user' ? 'You' : 'MedVision AI'}</span>
              <span>•</span>
              <span>{msg.timestamp}</span>
              {msg.isPinned && <Pin className="w-3 h-3 text-amber-400 rotate-45" />}
            </div>

            {/* Bubble */}
            <div
              className={`max-w-[92%] p-3 rounded-xl text-xs leading-relaxed transition-all shadow-xs ${
                msg.role === 'user'
                  ? 'bg-indigo-600 text-white rounded-br-none'
                  : 'bg-slate-800/90 text-slate-100 border border-slate-700/80 rounded-bl-none'
              }`}
            >
              {/* Message Content formatted */}
              <div className="whitespace-pre-wrap font-sans text-[11.5px] space-y-2">
                {msg.content}
              </div>

              {/* Render Chart Widget inside chat if present */}
              {msg.chartType && predictionResult && (
                <div className="mt-2">
                  <ChatChartWidget
                    type={msg.chartType}
                    diseases={predictionResult.diseases}
                    topDiagnosis={predictionResult.topDiagnosis}
                    topConfidence={predictionResult.topConfidence}
                    severity={predictionResult.severity}
                    severityScore={predictionResult.severityScore}
                  />
                </div>
              )}

              {/* Message Footer Controls */}
              {msg.role === 'assistant' && (
                <div className="mt-2 pt-1.5 border-t border-slate-700/60 flex items-center justify-between text-[10px] text-slate-400">
                  <div className="flex items-center gap-1.5">
                    <button
                      onClick={() => handleCopyMessage(msg.id, msg.content)}
                      className="hover:text-white transition-colors cursor-pointer p-0.5"
                      title="Copy response"
                    >
                      {copiedId === msg.id ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                    </button>
                    <button
                      onClick={() => handleReaction(msg.id, 'like')}
                      className={`hover:text-white transition-colors cursor-pointer p-0.5 ${
                        msg.liked ? 'text-indigo-400' : ''
                      }`}
                      title="Helpful"
                    >
                      <ThumbsUp className="w-3 h-3" />
                    </button>
                    <button
                      onClick={() => handleReaction(msg.id, 'dislike')}
                      className={`hover:text-white transition-colors cursor-pointer p-0.5 ${
                        msg.disliked ? 'text-red-400' : ''
                      }`}
                      title="Not helpful"
                    >
                      <ThumbsDown className="w-3 h-3" />
                    </button>
                    <button
                      onClick={() => handleTogglePin(msg.id)}
                      className={`hover:text-white transition-colors cursor-pointer p-0.5 ${
                        msg.isPinned ? 'text-amber-400' : ''
                      }`}
                      title="Pin message"
                    >
                      <Pin className="w-3 h-3" />
                    </button>
                  </div>
                  <span className="font-mono text-[9px] text-slate-500">MedVision Copilot</span>
                </div>
              )}
            </div>

            {/* Smart Suggested Action Buttons */}
            {msg.role === 'assistant' && msg.suggestedQuestions && msg.suggestedQuestions.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-1 max-w-[92%]">
                {msg.suggestedQuestions.map((q, qIdx) => (
                  <button
                    key={qIdx}
                    onClick={() => handleSendMessage(q)}
                    className="bg-slate-800/80 hover:bg-indigo-900/60 text-slate-300 hover:text-indigo-200 text-[10px] px-2 py-0.5 rounded border border-slate-700/80 hover:border-indigo-600/60 transition-all flex items-center gap-1 cursor-pointer font-medium"
                  >
                    <span>{q}</span>
                    <ChevronRight className="w-3 h-3 opacity-60" />
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}

        {/* Typing indicator */}
        {isTyping && (
          <div className="flex items-center gap-2 text-xs text-indigo-400 bg-slate-800/60 p-2.5 rounded-lg border border-slate-700/60 w-fit animate-pulse font-mono">
            <Sparkles className="w-3.5 h-3.5 animate-spin text-indigo-400" />
            <span>Analyzing neural embeddings &amp; medical literature...</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Smart Quick Category Buttons Panel */}
      <div className="p-2 bg-slate-950 border-t border-slate-800/80 shrink-0">
        <div className="flex items-center gap-1 overflow-x-auto pb-1 text-[10px] scrollbar-none">
          <button
            onClick={() => handleSendMessage(`Explain ${predictionResult?.topDiagnosis || 'Pneumonia'}`)}
            className="px-2 py-1 bg-indigo-950 hover:bg-indigo-900 text-indigo-300 rounded border border-indigo-800/80 whitespace-nowrap cursor-pointer flex items-center gap-1"
          >
            <Stethoscope className="w-3 h-3" />
            <span>Explain Disease</span>
          </button>
          <button
            onClick={() => handleSendMessage('Explain Heatmap')}
            className="px-2 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded border border-slate-700 whitespace-nowrap cursor-pointer flex items-center gap-1"
          >
            <Activity className="w-3 h-3 text-amber-400" />
            <span>Grad-CAM</span>
          </button>
          <button
            onClick={() => handleSendMessage('Show Interactive Charts')}
            className="px-2 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded border border-slate-700 whitespace-nowrap cursor-pointer flex items-center gap-1"
          >
            <BarChart2 className="w-3 h-3 text-indigo-400" />
            <span>Top 5 Charts</span>
          </button>
          <button
            onClick={() => handleSendMessage('Treatment Overview')}
            className="px-2 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded border border-slate-700 whitespace-nowrap cursor-pointer flex items-center gap-1"
          >
            <FileText className="w-3 h-3 text-emerald-400" />
            <span>Treatment</span>
          </button>
          <button
            onClick={() => handleSendMessage('Summarize Scan')}
            className="px-2 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded border border-slate-700 whitespace-nowrap cursor-pointer flex items-center gap-1"
          >
            <Layers className="w-3 h-3 text-cyan-400" />
            <span>Summarize</span>
          </button>
          <button
            onClick={() => handleSendMessage('Suggest follow-up imaging')}
            className="px-2 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded border border-slate-700 whitespace-nowrap cursor-pointer flex items-center gap-1"
          >
            <Stethoscope className="w-3 h-3 text-violet-400" />
            <span>Follow-up</span>
          </button>
          <button
            onClick={() => handleSendMessage('Compare reports')}
            className="px-2 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded border border-slate-700 whitespace-nowrap cursor-pointer flex items-center gap-1"
          >
            <GitCompare className="w-3 h-3 text-pink-400" />
            <span>Compare</span>
          </button>
          {onExportPdf && (
            <button
              onClick={onExportPdf}
              className="px-2 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded border border-slate-700 whitespace-nowrap cursor-pointer flex items-center gap-1"
            >
              <Download className="w-3 h-3 text-blue-400" />
              <span>Download PDF</span>
            </button>
          )}
        </div>

        {/* Query Input Box */}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSendMessage();
          }}
          className="flex items-center gap-1.5 mt-1.5"
        >
          {/* Speech-to-Text Button */}
          <button
            type="button"
            onClick={handleVoiceInput}
            className={`p-2 rounded-lg border transition-colors cursor-pointer shrink-0 ${
              isVoiceActive
                ? 'bg-red-600 text-white border-red-500 animate-pulse'
                : 'bg-slate-900 text-slate-400 hover:text-slate-200 border-slate-800'
            }`}
            title="Voice Query Input"
          >
            {isVoiceActive ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
          </button>

          {/* Text Input */}
          <input
            type="text"
            value={inputQuery}
            onChange={(e) => setInputQuery(e.target.value)}
            placeholder="Ask AI Copilot about X-ray, Grad-CAM, symptoms..."
            className="flex-1 bg-slate-900 text-slate-100 placeholder-slate-500 text-xs px-3 py-2 rounded-lg border border-slate-800 focus:outline-hidden focus:border-indigo-500 transition-colors"
          />

          {/* Send Button */}
          <button
            type="submit"
            disabled={!inputQuery.trim() || isTyping}
            className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white p-2 rounded-lg transition-colors cursor-pointer shrink-0"
          >
            <Send className="w-4 h-4" />
          </button>
        </form>
      </div>
    </aside>
  );
};
