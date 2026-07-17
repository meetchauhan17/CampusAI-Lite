"use client";

import React, { useState } from "react";
import { 
  Sparkles, 
  Send, 
  RefreshCw, 
  CheckCircle, 
  AlertCircle, 
  Activity, 
  ShieldAlert, 
  Layers, 
  BookOpen, 
  Cpu, 
  Terminal 
} from "lucide-react";

interface ValidationResult {
  pipeline: string;
  final_answer: string;
  category: string;
  confidence: number;
  is_grounded: boolean;
  is_accurate: boolean;
  issues: string[];
  latency_ms: number;
  error?: string;
}

const BACKEND_URL = "http://localhost:8000";

export default function Home() {
  const [question, setQuestion] = useState("");
  const [pipeline, setPipeline] = useState<"crewai" | "langgraph" | "autogen" | "beeai">("langgraph");
  const [loading, setLoading] = useState(false);
  const [compareLoading, setCompareLoading] = useState(false);
  const [result, setResult] = useState<ValidationResult | null>(null);
  const [compareResults, setCompareResults] = useState<ValidationResult[] | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleAsk = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setCompareResults(null);
    setResult(null);
    setErrorMsg(null);

    try {
      const resp = await fetch(`${BACKEND_URL}/api/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, pipeline }),
      });

      if (!resp.ok) {
        throw new Error(`Server returned HTTP ${resp.status}`);
      }

      const data = await resp.json();
      setResult(data);
    } catch (err: any) {
      console.error(err);
      setErrorMsg(err.message || "Failed to contact the backend service.");
    } finally {
      setLoading(false);
    }
  };

  const handleCompare = async () => {
    if (!question.trim()) return;

    setCompareLoading(true);
    setResult(null);
    setCompareResults(null);
    setErrorMsg(null);

    try {
      const resp = await fetch(`${BACKEND_URL}/api/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      if (!resp.ok) {
        throw new Error(`Server returned HTTP ${resp.status}`);
      }

      const data = await resp.json();
      setCompareResults(data.results);
    } catch (err: any) {
      console.error(err);
      setErrorMsg(err.message || "Failed to run framework comparison.");
    } finally {
      setCompareLoading(false);
    }
  };

  // Find the fastest response in a compare run to apply styling focus
  const getFastestPipeline = () => {
    if (!compareResults) return "";
    let fastest = compareResults[0];
    for (const r of compareResults) {
      if (r.latency_ms < fastest.latency_ms && !r.error) {
        fastest = r;
      }
    }
    return fastest.pipeline;
  };

  const fastestId = getFastestPipeline();

  return (
    <main className="relative min-h-screen bg-void text-slate-100 grid-bg overflow-x-hidden">
      {/* ── Background Glow ── */}
      <div className="absolute top-[-10%] left-[20%] w-[600px] h-[600px] rounded-full bg-neural-indigo/5 blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[20%] right-[-10%] w-[500px] h-[500px] rounded-full bg-synapse-cyan/5 blur-[120px] pointer-events-none" />

      <div className="max-w-7xl mx-auto px-4 py-8 md:py-16 relative z-10 flex flex-col min-h-screen">
        
        {/* ── TOP HEADER ── */}
        <header className="flex justify-between items-center mb-12 md:mb-20 border-b border-white/5 pb-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-neural-indigo to-neural-violet flex items-center justify-center font-display font-bold text-xl text-white shadow-[0_0_20px_rgba(99,102,241,0.3)]">
              C
            </div>
            <div>
              <span className="font-display font-bold text-lg tracking-wide text-white">CAMPUS<span className="text-synapse-cyan">AI</span></span>
              <span className="text-[10px] block font-mono tracking-widest text-slate-400">LITE v1.0</span>
            </div>
          </div>
          <div className="flex items-center gap-6">
            <span className="hidden md:inline font-mono text-xs text-slate-400">
              [Status: <span className="text-synapse-cyan animate-pulse">LIVE // AGENTIC_ROUTING</span>]
            </span>
          </div>
        </header>

        {/* ── HERO ORB & BANNER ── */}
        <section className="flex flex-col items-center text-center mb-12 md:mb-16">
          
          {/* Animated 3D-Style Orb Container */}
          <div className="relative w-44 h-44 md:w-56 md:h-56 mb-8 flex items-center justify-center animate-float">
            {/* Outermost Ring */}
            <div className="absolute inset-0 rounded-full border border-neural-indigo/25 animate-spin-slow shadow-[0_0_35px_rgba(99,102,241,0.15)]" />
            {/* Middle Reverse Ring */}
            <div className="absolute w-[80%] h-[80%] rounded-full border border-dashed border-neural-violet/30 animate-spin-reverse-slow" />
            {/* Innermost Glowing Core */}
            <div className="absolute w-[50%] h-[50%] rounded-full bg-gradient-to-br from-neural-indigo via-neural-violet to-synapse-cyan opacity-80 blur-[8px] animate-pulse shadow-[0_0_40px_rgba(99,102,241,0.5)]" />
            <Cpu className="absolute w-8 h-8 text-white z-10 animate-pulse" />
          </div>

          <h2 className="font-display text-4xl md:text-6xl font-bold tracking-tight text-white mb-4 max-w-3xl">
            Cognitive orchestration <br className="hidden md:block"/>
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-neural-indigo via-neural-violet to-synapse-cyan">
              for your campus.
            </span>
          </h2>
          <p className="text-slate-400 font-sans text-sm md:text-base max-w-xl">
            Ask queries related to examinations, tuition fees, library resources, hostel guidelines, and academic schedules handled via 4 multi-agent frameworks.
          </p>
        </section>

        {/* ── INTERACTIVE TERMINAL INPUT ── */}
        <section className="max-w-3xl mx-auto w-full mb-12">
          
          {/* Pipeline Segmented Control */}
          <div className="flex justify-center mb-6">
            <div className="inline-flex p-1 rounded-full bg-surface/80 border border-white/5 backdrop-blur-md">
              {(["crewai", "langgraph", "autogen", "beeai"] as const).map((p) => (
                <button
                  key={p}
                  type="button"
                  onClick={() => setPipeline(p)}
                  className={`px-4 py-1.5 rounded-full font-mono text-xs uppercase transition-all duration-300 ${
                    pipeline === p 
                      ? "bg-neural-indigo text-white shadow-[0_0_15px_rgba(99,102,241,0.4)]" 
                      : "text-slate-400 hover:text-white"
                  }`}
                >
                  {p === "crewai" ? "CrewAI" : p === "langgraph" ? "LangGraph" : p === "autogen" ? "AG2" : "BeeAI (PoC)"}
                </button>
              ))}
            </div>
          </div>

          {/* Terminal Input Form */}
          <form onSubmit={handleAsk} className="p-4 md:p-6 bg-surface/60 border border-white/5 rounded-2xl shadow-2xl backdrop-blur-lg">
            <div className="flex items-center gap-2 text-slate-500 font-mono text-xs mb-3">
              <Terminal className="w-4 h-4 text-neural-indigo" />
              <span>STUDENT_PROMPT_INPUT_SHELL</span>
            </div>

            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Enter your university query (e.g. When is the Artificial Intelligence exam?)..."
              rows={3}
              className="w-full bg-transparent border-0 border-b border-white/10 pb-4 text-slate-100 font-mono text-sm focus:ring-0 focus:border-synapse-cyan placeholder-slate-600 resize-none outline-none transition-all"
            />

            <div className="flex flex-col sm:flex-row justify-between items-center gap-4 mt-4">
              {/* Presets */}
              <div className="flex flex-wrap gap-2 justify-center sm:justify-start">
                <button
                  type="button"
                  onClick={() => setQuestion("When is the Artificial Intelligence exam and what hall do I report to?")}
                  className="px-3 py-1 rounded bg-white/5 border border-white/5 text-[11px] font-mono text-slate-400 hover:text-white hover:border-white/10"
                >
                  Timetable
                </button>
                <button
                  type="button"
                  onClick={() => setQuestion("How much is the B.Tech tuition fee and what happens if I pay late?")}
                  className="px-3 py-1 rounded bg-white/5 border border-white/5 text-[11px] font-mono text-slate-400 hover:text-white hover:border-white/10"
                >
                  Fees
                </button>
                <button
                  type="button"
                  onClick={() => setQuestion("How many books can a postgraduate student borrow from the library?")}
                  className="px-3 py-1 rounded bg-white/5 border border-white/5 text-[11px] font-mono text-slate-400 hover:text-white hover:border-white/10"
                >
                  Library
                </button>
              </div>

              <div className="flex gap-3 w-full sm:w-auto">
                <button
                  type="button"
                  onClick={handleCompare}
                  disabled={loading || compareLoading || !question.trim()}
                  className="flex-1 sm:flex-initial px-5 py-2.5 rounded-xl border border-white/10 bg-white/5 hover:bg-white/10 font-mono text-xs text-white transition-all disabled:opacity-40"
                >
                  {compareLoading ? "Comparing..." : "Compare All 4"}
                </button>
                <button
                  type="submit"
                  disabled={loading || compareLoading || !question.trim()}
                  className="flex-1 sm:flex-initial px-6 py-2.5 rounded-xl bg-gradient-to-r from-neural-indigo to-neural-violet hover:shadow-[0_0_20px_rgba(99,102,241,0.4)] text-white font-mono text-xs font-semibold flex items-center justify-center gap-2 transition-all disabled:opacity-40"
                >
                  {loading ? (
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  ) : (
                    <Send className="w-3.5 h-3.5" />
                  )}
                  Execute
                </button>
              </div>
            </div>
          </form>

          {errorMsg && (
            <div className="mt-4 p-4 bg-red-950/20 border border-red-500/20 rounded-xl flex items-start gap-3">
              <ShieldAlert className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
              <div>
                <h4 className="font-mono text-xs font-semibold text-red-300">EXECUTION_ERROR</h4>
                <p className="text-xs text-red-400 mt-1">{errorMsg}</p>
              </div>
            </div>
          )}
        </section>

        {/* ── SINGLE PIPELINE STAGE SKELETONS / TIMELINE ── */}
        {(loading || result) && (
          <section className="max-w-3xl mx-auto w-full mb-16">
            <h3 className="font-display font-semibold text-xl text-white mb-6 flex items-center gap-2">
              <Activity className="w-5 h-5 text-neural-indigo" />
              Orchestration Sequence: <span className="font-mono text-neural-violet uppercase text-sm">{pipeline}</span>
            </h3>

            {loading ? (
              // Stage Skeletons
              <div className="space-y-6">
                {[1, 2, 3].map((num) => (
                  <div key={num} className="p-5 rounded-2xl bg-surface/30 border border-white/5 animate-pulse flex items-start gap-4">
                    <div className="w-8 h-8 rounded-lg bg-white/5 border border-white/10 flex items-center justify-center font-mono text-xs text-slate-500">
                      0{num}
                    </div>
                    <div className="space-y-2 flex-1">
                      <div className="h-3 w-1/4 bg-white/10 rounded" />
                      <div className="h-4 w-3/4 bg-white/5 rounded" />
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              result && (
                // Structured Blockchain-Style Timeline
                <div className="space-y-6">
                  
                  {/* Step 1: Planner Stage */}
                  <div className="relative p-5 rounded-2xl bg-surface/50 border border-white/5 backdrop-blur-md">
                    <div className="flex justify-between items-start mb-3">
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-lg bg-neural-indigo/10 border border-neural-indigo/20 flex items-center justify-center font-mono text-xs text-neural-indigo">
                          01
                        </div>
                        <div>
                          <h4 className="font-display font-semibold text-sm text-white">Planner Stage</h4>
                          <p className="text-[10px] font-mono text-slate-500">AGENT_ROLE: QUERY_PLANNER</p>
                        </div>
                      </div>
                      <span className="px-2.5 py-0.5 rounded-full bg-neural-indigo/10 border border-neural-indigo/20 text-[10px] font-mono text-neural-indigo uppercase">
                        {result.category}
                      </span>
                    </div>
                    <div className="pl-11">
                      <p className="text-xs text-slate-400 leading-relaxed font-mono">
                        Question categorized. Derived sub-tasks generated successfully. Ready for document ingestion retrieval.
                      </p>
                    </div>
                  </div>

                  {/* Step 2: Information/Retrieval Stage */}
                  <div className="relative p-5 rounded-2xl bg-surface/50 border border-white/5 backdrop-blur-md">
                    <div className="flex justify-between items-start mb-3">
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-lg bg-neural-violet/10 border border-neural-violet/20 flex items-center justify-center font-mono text-xs text-neural-violet">
                          02
                        </div>
                        <div>
                          <h4 className="font-display font-semibold text-sm text-white">Retrieval &amp; Draft Answer</h4>
                          <p className="text-[10px] font-mono text-slate-500">AGENT_ROLE: CORPUS_RETRIEVER</p>
                        </div>
                      </div>
                      <span className="px-2.5 py-0.5 rounded-full bg-white/5 border border-white/10 text-[10px] font-mono text-slate-400">
                        {result.latency_ms.toFixed(0)} ms
                      </span>
                    </div>
                    <div className="pl-11">
                      <div className="p-4 rounded-xl bg-void/50 border border-white/5 text-sm text-slate-300 leading-relaxed">
                        {result.final_answer}
                      </div>
                    </div>
                  </div>

                  {/* Step 3: PydanticAI Validation Stage */}
                  <div className="relative p-5 rounded-2xl bg-surface/80 border border-neural-indigo/20 shadow-[0_0_25px_rgba(99,102,241,0.08)] backdrop-blur-md">
                    <div className="flex justify-between items-start mb-3">
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-lg bg-synapse-cyan/10 border border-synapse-cyan/20 flex items-center justify-center font-mono text-xs text-synapse-cyan">
                          03
                        </div>
                        <div>
                          <h4 className="font-display font-semibold text-sm text-white">Fact Check &amp; Validation</h4>
                          <p className="text-[10px] font-mono text-slate-500">AGENT_ROLE: PYDANTIC_AI_VALIDATOR</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        {result.is_accurate && result.is_grounded ? (
                          <span className="animate-pulse px-2.5 py-0.5 rounded-full bg-synapse-cyan/10 border border-synapse-cyan/30 text-[10px] font-mono text-synapse-cyan font-bold flex items-center gap-1 shadow-[0_0_10px_rgba(34,211,238,0.2)]">
                            VALIDATED ✓
                          </span>
                        ) : (
                          <span className="px-2.5 py-0.5 rounded-full bg-red-950/30 border border-red-500/25 text-[10px] font-mono text-red-400 flex items-center gap-1">
                            ISSUES DETECTED ✗
                          </span>
                        )}
                      </div>
                    </div>
                    <div className="pl-11 space-y-4">
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                        <div className="p-3 rounded-lg bg-void/50 border border-white/5">
                          <p className="text-[10px] font-mono text-slate-500 uppercase">Grounded</p>
                          <p className={`font-mono text-sm font-semibold mt-1 ${result.is_grounded ? "text-synapse-cyan" : "text-red-400"}`}>
                            {result.is_grounded ? "Yes" : "No"}
                          </p>
                        </div>
                        <div className="p-3 rounded-lg bg-void/50 border border-white/5">
                          <p className="text-[10px] font-mono text-slate-500 uppercase">Accurate</p>
                          <p className={`font-mono text-sm font-semibold mt-1 ${result.is_accurate ? "text-synapse-cyan" : "text-red-400"}`}>
                            {result.is_accurate ? "Yes" : "No"}
                          </p>
                        </div>
                        <div className="p-3 rounded-lg bg-void/50 border border-white/5">
                          <p className="text-[10px] font-mono text-slate-500 uppercase">Confidence</p>
                          <p className="font-mono text-sm font-semibold text-white mt-1">
                            {(result.confidence * 100).toFixed(0)}%
                          </p>
                        </div>
                        <div className="p-3 rounded-lg bg-void/50 border border-white/5">
                          <p className="text-[10px] font-mono text-slate-500 uppercase">API Latency</p>
                          <p className="font-mono text-sm font-semibold text-slate-300 mt-1">
                            {result.latency_ms.toFixed(0)} ms
                          </p>
                        </div>
                      </div>

                      {result.issues.length > 0 && (
                        <div className="p-3 rounded-lg bg-red-950/10 border border-red-500/10">
                          <p className="text-[10px] font-mono text-red-400 font-semibold uppercase mb-1.5 flex items-center gap-1">
                            <AlertCircle className="w-3.5 h-3.5" /> Handled Inaccuracies / Hallucinations:
                          </p>
                          <ul className="list-disc pl-4 text-xs text-red-300 space-y-1">
                            {result.issues.map((issue, idx) => (
                              <li key={idx}>{issue}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  </div>

                </div>
              )
            )}
          </section>
        )}

        {/* ── COMPARE GRID: 4 GLASS CARDS ── */}
        {(compareLoading || compareResults) && (
          <section className="w-full mb-16">
            <h3 className="font-display font-semibold text-xl text-white mb-8 text-center flex items-center justify-center gap-2">
              <Layers className="w-5 h-5 text-neural-indigo" />
              Four-Framework Cognitive Analysis
            </h3>

            {compareLoading ? (
              // Compare Loading Skeletons
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {[1, 2, 3, 4].map((num) => (
                  <div key={num} className="p-6 rounded-2xl bg-surface/30 border border-white/5 animate-pulse min-h-[300px] flex flex-col justify-between">
                    <div>
                      <div className="h-4 w-1/3 bg-white/10 rounded mb-4" />
                      <div className="h-3 w-3/4 bg-white/5 rounded mb-2" />
                      <div className="h-3 w-1/2 bg-white/5 rounded" />
                    </div>
                    <div className="h-8 bg-white/5 rounded" />
                  </div>
                ))}
              </div>
            ) : (
              compareResults && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 items-stretch">
                  {compareResults.map((r) => {
                    const isFastest = r.pipeline === fastestId;
                    return (
                      <div 
                        key={r.pipeline}
                        className={`p-6 rounded-2xl bg-surface/50 border backdrop-blur-md flex flex-col justify-between transition-all duration-500 ${
                          isFastest 
                            ? "border-neural-indigo shadow-[0_0_30px_rgba(99,102,241,0.15)] md:scale-[1.03] relative z-10" 
                            : "border-white/5 hover:border-white/10"
                        }`}
                      >
                        {isFastest && (
                          <span className="absolute -top-3 left-1/2 transform -translate-x-1/2 px-3 py-1 rounded-full bg-neural-indigo text-[10px] font-mono text-white font-bold uppercase tracking-wider shadow-lg animate-bounce">
                            ⚡ FASTEST
                          </span>
                        )}

                        <div>
                          <div className="flex justify-between items-center mb-4">
                            <span className="font-display font-bold text-sm tracking-wide text-white uppercase">
                              {r.pipeline === "crewai" ? "CrewAI" : r.pipeline === "langgraph" ? "LangGraph" : r.pipeline === "autogen" ? "AG2 (AutoGen)" : "BeeAI (PoC)"}
                            </span>
                            <span className="font-mono text-[10px] text-slate-500">
                              {r.latency_ms.toFixed(0)} ms
                            </span>
                          </div>

                          <div className="space-y-4">
                            <p className="text-xs text-slate-400 font-mono line-clamp-6 leading-relaxed">
                              {r.final_answer}
                            </p>

                            <div className="pt-4 border-t border-white/5 grid grid-cols-2 gap-2 text-[10px] font-mono text-slate-500">
                              <div>
                                <span>ACCURATE:</span>
                                <span className={`block font-semibold mt-0.5 ${r.is_accurate ? "text-synapse-cyan" : "text-red-400"}`}>
                                  {r.is_accurate ? "TRUE" : "FALSE"}
                                </span>
                              </div>
                              <div>
                                <span>CONFIDENCE:</span>
                                <span className="block font-semibold text-white mt-0.5">
                                  {(r.confidence * 100).toFixed(0)}%
                                </span>
                              </div>
                            </div>
                          </div>
                        </div>

                        <div className="mt-6">
                          {r.error ? (
                            <div className="px-3 py-1.5 rounded-lg bg-red-950/20 border border-red-500/25 flex items-center gap-1.5 text-[10px] text-red-400 font-mono">
                              <AlertCircle className="w-3.5 h-3.5" /> ERROR_OCCURRED
                            </div>
                          ) : (
                            <div className="px-3 py-1.5 rounded-lg bg-synapse-cyan/5 border border-synapse-cyan/10 flex items-center gap-1.5 text-[10px] text-synapse-cyan font-mono">
                              <CheckCircle className="w-3.5 h-3.5" /> VERIFIED_OUTPUT
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )
            )}
          </section>
        )}

      </div>
    </main>
  );
}
