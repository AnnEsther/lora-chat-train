"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { HelpPanel } from "@/app/components/HelpPanel";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const MODEL_SERVER_URL = process.env.NEXT_PUBLIC_MODEL_SERVER_URL ?? "http://localhost:8001";
const POLL_INTERVAL_MS = 5000;
const MIN_TRAINING_SAMPLES = parseInt(process.env.NEXT_PUBLIC_MIN_TRAINING_SAMPLES ?? "10", 10);

// ── Types ─────────────────────────────────────────────────────────────────────

type SessionState =
  | "ACTIVE"
  | "PRE_SLEEP_WARNING"
  | "INSUFFICIENT_DATA"
  | "VALIDATING"
  | "SLEEPING"
  | "TRAINING"
  | "EVALUATING"
  | "DEPLOYING"
  | "READY"
  | "FAILED";

interface QAPair {
  id: string;
  question: string;
  answer: string;
  validated: boolean;
  edited: boolean;
  source_document_name?: string | null;
}

interface Message {
  role: "user" | "assistant" | "system";
  content: string;
  id?: string;
  streaming?: boolean;
  synthLoading?: boolean;
  segmentCount?: number;   // total segments the backend will process
  qaPairs?: QAPair[];
}

interface Session {
  id: string;
  state: SessionState;
  total_tokens: number;
  max_tokens: number;
  system_prompt?: string | null;
  training_system_prompt?: string | null;
  failure_reason?: string | null;
  created_at: string;
}

interface QACount {
  total_count: number;
  validated_count: number;
  min_required: number;
  ready_to_train: boolean;
}

interface TrainStatus {
  status: "idle" | "running" | "completed" | "failed";
  run_id?: string;
  progress: string;
  started_at: string | null;
  finished_at: string | null;
  vram_used_gb: number | null;
  vram_free_gb: number | null;
}

interface ModelHealth {
  status: string;
  model_loaded: boolean;
  adapter: string | null;
  training_active: boolean;
  gpu: {
    name: string;
    vram_total_gb: number;
    vram_used_gb: number;
  } | null;
}

interface OutputFile {
  name: string;
  path: string;
  size: string;
}

interface Adapter {
  id: string;
  version: string;
  path: string;
  trained_at: string | null;
  is_current?: boolean;
  is_base?: boolean;
}

// ── State colours / labels ────────────────────────────────────────────────────

const STATE_COLORS: Record<SessionState, string> = {
  ACTIVE:            "bg-green-100 text-green-800",
  PRE_SLEEP_WARNING: "bg-yellow-100 text-yellow-800",
  INSUFFICIENT_DATA: "bg-orange-100 text-orange-800",
  VALIDATING:        "bg-purple-100 text-purple-800",
  SLEEPING:          "bg-gray-100 text-gray-600",
  TRAINING:          "bg-blue-100 text-blue-800",
  EVALUATING:        "bg-purple-100 text-purple-800",
  DEPLOYING:         "bg-orange-100 text-orange-800",
  READY:             "bg-green-100 text-green-800",
  FAILED:            "bg-red-100 text-red-800",
};

const STATE_LABELS: Record<SessionState, string> = {
  ACTIVE:            "Active",
  PRE_SLEEP_WARNING: "⚠ Low tokens",
  INSUFFICIENT_DATA: "⚠ Need more data",
  VALIDATING:        "Processing…",
  SLEEPING:          "Training queued",
  TRAINING:          "Training…",
  EVALUATING:        "Evaluating…",
  DEPLOYING:         "Deploying…",
  READY:             "Ready — new adapter live",
  FAILED:            "Training failed — chat still active",
};

// ── Diagnostic panel sub-components ──────────────────────────────────────────

function GaugeBar({ value, max, color }: { value: number; max: number; color: string }) {
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
      <div className={`h-full rounded-full transition-all duration-500 ${color}`} style={{ width: `${pct}%` }} />
    </div>
  );
}

function StatRow({ label, value, sub, valueColor }: { label: string; value: string; sub?: string; valueColor?: string }) {
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-gray-100 last:border-0">
      <span className="text-xs text-gray-500">{label}</span>
      <div className="text-right">
        <span className={`text-xs font-medium ${valueColor ?? "text-gray-800"}`}>{value}</span>
        {sub && <span className="block text-xs text-gray-400">{sub}</span>}
      </div>
    </div>
  );
}

function SectionHeader({ title, dot }: { title: string; dot?: string }) {
  return (
    <div className="flex items-center gap-2 mb-2 mt-4 first:mt-0">
      {dot && <span className={`w-2 h-2 rounded-full flex-shrink-0 ${dot}`} />}
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">{title}</h3>
    </div>
  );
}

function PipelineStep({ label, state }: { label: string; state: "done" | "active" | "pending" | "failed" }) {
  const icons = { done: <span className="text-green-500 text-sm">✓</span>, active: <span className="animate-spin text-blue-500 text-sm inline-block">⟳</span>, pending: <span className="text-gray-300 text-sm">○</span>, failed: <span className="text-red-500 text-sm">✗</span> };
  const labels = { done: "text-green-700", active: "text-blue-700 font-medium", pending: "text-gray-400", failed: "text-red-600" };
  return (
    <div className="flex items-center gap-2 py-1">
      <div className="w-4 flex justify-center">{icons[state]}</div>
      <span className={`text-xs ${labels[state]}`}>{label}</span>
    </div>
  );
}

function getPipelineSteps(sessionState: SessionState, trainStatus?: TrainStatus) {
  const steps = ["Build dataset", "Train model", "Evaluate", "Deploy adapter"];
  const stateToActiveStep: Record<SessionState, number> = {
    TRAINING: 1, EVALUATING: 2, DEPLOYING: 3, READY: 3,
    FAILED: -1, ACTIVE: -1, PRE_SLEEP_WARNING: -1, INSUFFICIENT_DATA: -1, VALIDATING: 0, SLEEPING: 1,
  };
  if (trainStatus?.status === "completed") return steps.map((label) => ({ label, state: "done" as const }));
  const activeStep = stateToActiveStep[sessionState] ?? -1;
  return steps.map((label, i) => {
    if (sessionState === "FAILED") return { label, state: i < activeStep ? "done" : i === activeStep ? "failed" : "pending" } as const;
    if (i < activeStep) return { label, state: "done" } as const;
    if (i === activeStep) return { label, state: "active" } as const;
    return { label, state: "pending" } as const;
  });
}

function elapsed(isoStr: string | null): string {
  if (!isoStr) return "—";
  const secs = Math.floor((Date.now() - new Date(isoStr).getTime()) / 1000);
  if (secs < 60)   return `${secs}s`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ${secs % 60}s`;
  return `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`;
}

// ── Diagnostic panel ──────────────────────────────────────────────────────────

function DiagnosticPanel({
  session, health, trainStatus, lastPoll, selectedAdapter, adapters, qaCount, onRestartTraining,
}: {
  session: Session | null;
  health: ModelHealth | null;
  trainStatus: TrainStatus | null;
  lastPoll: Date | null;
  selectedAdapter: string;
  adapters: Adapter[];
  qaCount: QACount | null;
  onRestartTraining?: () => void;
}) {
  const gpu = health?.gpu ?? null;
  const vramPct = gpu ? gpu.vram_used_gb / gpu.vram_total_gb : 0;
  const tokenPct = session ? session.total_tokens / session.max_tokens : 0;
  const isTraining = trainStatus?.status === "running";
  const pipelineSteps = session ? getPipelineSteps(session.state, trainStatus ?? undefined) : [];
  const adapterVersion = adapters.find(a => a.id === selectedAdapter)?.version ?? "Base model";

  return (
    <aside className="w-72 min-w-72 h-screen overflow-y-auto bg-gray-50 border-l border-gray-200 px-4 py-4 flex flex-col gap-0 text-sm">
      <div className="flex items-center justify-between mb-3">
        <h2 className="font-semibold text-gray-700 text-sm">Diagnostics</h2>
        {lastPoll && <span className="text-xs text-gray-400">updated {elapsed(lastPoll.toISOString())} ago</span>}
      </div>

      <SectionHeader title="Model server" dot={health?.status === "ok" ? "bg-green-400" : "bg-red-400"} />
      <div className="bg-white rounded-lg border border-gray-200 px-3 py-2">
        <StatRow label="Status" value={health ? (health.model_loaded ? "Ready" : "Loading…") : "Unreachable"} />
        <StatRow label="Adapter" value={health?.adapter ? health.adapter.split(/[\\/]/).slice(-2).join("/") : "Base model"} />
        <StatRow label="Training" value={health?.training_active ? "🔥 In progress" : "Idle"} />
      </div>

      {gpu && (
        <>
          <SectionHeader title="GPU" dot="bg-purple-400" />
          <div className="bg-white rounded-lg border border-gray-200 px-3 py-2">
            <StatRow label="Device" value={gpu.name} />
            <StatRow label="VRAM used" value={`${gpu.vram_used_gb.toFixed(1)} / ${gpu.vram_total_gb.toFixed(1)} GB`} />
            <div className="py-1">
              <GaugeBar value={gpu.vram_used_gb} max={gpu.vram_total_gb} color={vramPct > 0.85 ? "bg-red-400" : vramPct > 0.65 ? "bg-yellow-400" : "bg-purple-400"} />
            </div>
          </div>
        </>
      )}

      {session && (
        <>
          <SectionHeader title="Session" dot="bg-blue-400" />
          <div className="bg-white rounded-lg border border-gray-200 px-3 py-2">
            <StatRow label="ID" value={session.id.slice(0, 8) + "…"} />
            <StatRow label="State" value={STATE_LABELS[session.state]} />
            <StatRow label="Tokens" value={`${session.total_tokens} / ${session.max_tokens}`} />
            <div className="py-1">
              <GaugeBar value={session.total_tokens} max={session.max_tokens} color={tokenPct > 0.85 ? "bg-red-400" : tokenPct > 0.7 ? "bg-yellow-400" : "bg-blue-400"} />
            </div>
            <StatRow label="Adapter" value={adapterVersion} />
          </div>
        </>
      )}

      {qaCount !== null && (
        <>
          <SectionHeader title="Training data" dot={qaCount.ready_to_train ? "bg-green-400" : "bg-amber-400"} />
          <div className="bg-white rounded-lg border border-gray-200 px-3 py-2">
            <StatRow label="Total Q&A pairs" value={String(qaCount.total_count)} />
            <StatRow label="Validated" value={`${qaCount.validated_count} / ${qaCount.min_required} needed`} valueColor={qaCount.ready_to_train ? "text-green-600" : "text-amber-600"} />
            <div className="py-1">
              <GaugeBar value={qaCount.validated_count} max={qaCount.min_required} color={qaCount.ready_to_train ? "bg-green-400" : "bg-amber-400"} />
            </div>
          </div>
        </>
      )}

      {session && !["ACTIVE", "PRE_SLEEP_WARNING"].includes(session.state) && (
        <>
          <SectionHeader title="Pipeline" dot="bg-amber-400" />
          <div className="bg-white rounded-lg border border-gray-200 px-3 py-2">
            {pipelineSteps.map((step) => <PipelineStep key={step.label} label={step.label} state={step.state} />)}
          </div>
        </>
      )}

      {trainStatus && trainStatus.status !== "idle" && (
        <>
          <SectionHeader title="Training" dot={trainStatus.status === "failed" ? "bg-red-400" : isTraining ? "bg-blue-400 animate-pulse" : "bg-green-400"} />
          <div className={`rounded-lg border px-3 py-2 ${trainStatus.status === "failed" ? "bg-red-50 border-red-200" : "bg-white border-gray-200"}`}>
            <StatRow label="Status" value={trainStatus.status.charAt(0).toUpperCase() + trainStatus.status.slice(1)} valueColor={trainStatus.status === "failed" ? "text-red-600" : undefined} />
            {trainStatus.run_id && <StatRow label="Run ID" value={trainStatus.run_id.slice(0, 8) + "…"} />}
            <div className="py-1.5">
              <p className="text-xs text-gray-500 mb-0.5">Progress</p>
              <p className={`text-xs leading-relaxed ${trainStatus.status === "failed" ? "text-red-700" : "text-gray-800"}`}>{trainStatus.progress || "—"}</p>
            </div>
            {trainStatus.started_at && <StatRow label="Running for" value={elapsed(trainStatus.started_at)} />}
            {trainStatus.finished_at && <StatRow label="Finished" value={new Date(trainStatus.finished_at).toLocaleTimeString()} />}
            {trainStatus.status === "failed" && session && (
              <button
                onClick={async () => {
                  try {
                    const resp = await fetch(`${API_URL}/sessions/${session.id}/restart-training`, { method: "POST" });
                    if (resp.ok && onRestartTraining) onRestartTraining();
                  } catch {}
                }}
                className="mt-2 w-full text-xs px-3 py-1.5 rounded bg-red-100 hover:bg-red-200 text-red-700"
              >
                Restart Training
              </button>
            )}
          </div>
        </>
      )}

      <SectionHeader title="Quick links" />
      <div className="bg-white rounded-lg border border-gray-200 px-3 py-2 space-y-1">
        {[
          { label: "Model health",  href: `${MODEL_SERVER_URL}/health` },
          { label: "Train status",  href: `${MODEL_SERVER_URL}/train/status` },
          { label: "API health",    href: `${API_URL}/health` },
        ].map((link) => (
          <a key={link.href} href={link.href} target="_blank" rel="noreferrer" className="block text-xs text-blue-600 hover:text-blue-800 hover:underline py-0.5">
            {link.label} ↗
          </a>
        ))}
      </div>
      <div className="h-6" />
    </aside>
  );
}

// ── Inline QA deck ────────────────────────────────────────────────────────────

function InlineDeck({
  pairs,
  sessionId,
  synthLoading,
  segmentCount,
  onUpdate,
  onDelete,
}: {
  pairs: QAPair[];
  sessionId: string;
  turnId: string | undefined;
  synthLoading: boolean;
  segmentCount: number;
  onUpdate: (qaId: string, updates: Partial<QAPair>) => void;
  onDelete: (qaId: string) => void;
}) {
  const [cardIdx, setCardIdx]           = useState(0);
  const [slideDir, setSlideDir]         = useState<"left" | "right" | null>(null);
  const [animating, setAnimating]       = useState(false);
  const [editing, setEditing]           = useState(false);
  const [question, setQuestion]         = useState("");
  const [answer, setAnswer]             = useState("");
  const [saving, setSaving]             = useState(false);
  const [deleting, setDeleting]         = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [validatingAll, setValidatingAll] = useState(false);

  const totalCards  = synthLoading ? Math.max(pairs.length, segmentCount) : pairs.length;
  const currentPair = pairs[cardIdx] ?? null;
  const allValidated = pairs.length > 0 && pairs.every(p => p.validated);

  // Clamp index when pairs arrive or get deleted
  useEffect(() => {
    if (cardIdx >= pairs.length && pairs.length > 0) setCardIdx(pairs.length - 1);
  }, [pairs.length, cardIdx]);

  // Sync edit buffers when the displayed card changes
  useEffect(() => {
    if (currentPair) { setQuestion(currentPair.question); setAnswer(currentPair.answer); }
    setEditing(false);
    setConfirmDelete(false);
    setSaving(false);
    setDeleting(false);
  }, [cardIdx, currentPair?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  // Sync buffers if pair content changes externally (session reload)
  useEffect(() => {
    if (currentPair && !editing) { setQuestion(currentPair.question); setAnswer(currentPair.answer); }
  }, [currentPair?.question, currentPair?.answer]); // eslint-disable-line react-hooks/exhaustive-deps

  const goTo = useCallback((target: number, dir: "left" | "right") => {
    if (animating || target === cardIdx) return;
    setSlideDir(dir);
    setAnimating(true);
    setTimeout(() => { setCardIdx(target); setSlideDir(null); setAnimating(false); }, 260);
  }, [animating, cardIdx]);

  const goNext = useCallback(() => {
    if (cardIdx + 1 < pairs.length) goTo(cardIdx + 1, "left");
  }, [cardIdx, pairs.length, goTo]);

  const goPrev = useCallback(() => {
    if (cardIdx - 1 >= 0) goTo(cardIdx - 1, "right");
  }, [cardIdx, goTo]);

  const handleSave = useCallback(async () => {
    if (!currentPair) return;
    setSaving(true);
    try {
      const resp = await fetch(`${API_URL}/sessions/${sessionId}/qa/${currentPair.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, answer }),
      });
      if (resp.ok) { onUpdate(currentPair.id, { question, answer, edited: true }); setEditing(false); }
    } catch {}
    setSaving(false);
  }, [currentPair, sessionId, question, answer, onUpdate]);

  const handleValidateToggle = useCallback(async () => {
    if (!currentPair) return;
    const newVal = !currentPair.validated;
    try {
      const resp = await fetch(`${API_URL}/sessions/${sessionId}/qa/${currentPair.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ validated: newVal }),
      });
      if (resp.ok) {
        onUpdate(currentPair.id, { validated: newVal });
        // Auto-advance on validate (not on un-validate)
        if (newVal) goNext();
      }
    } catch {}
  }, [currentPair, sessionId, onUpdate, goNext]);

  const handleValidateAll = useCallback(async () => {
    const unvalidated = pairs.filter(p => !p.validated);
    if (unvalidated.length === 0) return;
    setValidatingAll(true);
    try {
      await Promise.all(
        unvalidated.map(p =>
          fetch(`${API_URL}/sessions/${sessionId}/qa/${p.id}`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ validated: true }),
          }).then(resp => { if (resp.ok) onUpdate(p.id, { validated: true }); })
        )
      );
    } catch {}
    setValidatingAll(false);
    goTo(0, "right");
  }, [pairs, sessionId, onUpdate, goTo]);

  const handleDelete = useCallback(async () => {
    if (!currentPair) return;
    setDeleting(true);
    try {
      await fetch(`${API_URL}/sessions/${sessionId}/qa/${currentPair.id}`, { method: "DELETE" });
      onDelete(currentPair.id);
    } catch {}
    setDeleting(false);
    setConfirmDelete(false);
  }, [currentPair, sessionId, onDelete]);

  // Keyboard nav — only fires when this deck wrapper is focused
  const handleKeyDown = useCallback(async (e: React.KeyboardEvent) => {
    if (e.key === "ArrowRight") {
      e.preventDefault();
      if (editing) await handleSave();
      goNext();
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      if (editing) await handleSave();
      goPrev();
    }
  }, [editing, handleSave, goNext, goPrev]);

  const isSkeletonSlot = !currentPair && synthLoading;
  const exitTranslate  = slideDir === "left"  ? "-translate-x-full" : slideDir === "right" ? "translate-x-full"  : "translate-x-0";

  // Dot colour per card validation state
  const dotClass = (di: number) => {
    if (di >= pairs.length) return "bg-gray-200 animate-pulse cursor-default";
    const validated = pairs[di].validated;
    const isCurrent = di === cardIdx;
    if (isCurrent)   return validated ? "bg-green-500 ring-2 ring-green-200" : "bg-red-400 ring-2 ring-red-200";
    return validated ? "bg-green-300 hover:bg-green-400 cursor-pointer" : "bg-red-200 hover:bg-red-300 cursor-pointer";
  };

  return (
    <div
      className="mt-2 outline-none focus-within:ring-2 focus-within:ring-blue-200 rounded-2xl"
      tabIndex={-1}
      onKeyDown={handleKeyDown}
    >
      <div className="border border-gray-200 rounded-2xl bg-white shadow-sm overflow-hidden">

        {/* ── Nav header ── */}
        <div className="flex items-center gap-3 px-4 py-2.5 border-b border-gray-100 bg-gray-50">

          {/* Dot indicators */}
          <div className="flex items-center gap-1.5 flex-shrink-0">
            {Array.from({ length: totalCards }).map((_, di) => (
              <button
                key={di}
                onClick={() => di < pairs.length && goTo(di, di > cardIdx ? "left" : "right")}
                disabled={di >= pairs.length}
                className={`w-2.5 h-2.5 rounded-full transition-all ${dotClass(di)}`}
                title={di < pairs.length ? (pairs[di].validated ? `Card ${di + 1} — validated` : `Card ${di + 1} — not validated`) : "Generating…"}
              />
            ))}
          </div>

          {/* Counter + streaming indicator */}
          <div className="flex items-center gap-1.5 flex-shrink-0">
            {synthLoading && (
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500" />
              </span>
            )}
            <span className="text-xs text-gray-400 tabular-nums font-medium">
              {totalCards > 0
                ? synthLoading && pairs.length < segmentCount
                  ? `${pairs.length} / ${segmentCount} generating…`
                  : `${Math.min(cardIdx + 1, totalCards)} / ${totalCards}`
                : "Generating…"}
            </span>
          </div>

          {/* Spacer */}
          <div className="flex-1" />

          {/* Validate all button */}
          {pairs.length > 1 && !allValidated && (
            <button
              onClick={handleValidateAll}
              disabled={validatingAll}
              className="text-xs px-3 py-1.5 rounded-lg bg-green-600 hover:bg-green-700 text-white font-medium disabled:opacity-60 transition-colors flex-shrink-0"
            >
              {validatingAll ? "Validating…" : "Validate all"}
            </button>
          )}

          {/* Prev / Next */}
          <div className="flex items-center gap-1 flex-shrink-0">
            <button
              onClick={goPrev}
              disabled={cardIdx === 0 || animating}
              className="px-3 py-1.5 text-sm rounded-lg bg-white border border-gray-300 hover:bg-gray-100 text-gray-600 font-medium disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
              ←
            </button>
            <button
              onClick={goNext}
              disabled={cardIdx >= pairs.length - 1 || animating}
              className="px-3 py-1.5 text-sm rounded-lg bg-white border border-gray-300 hover:bg-gray-100 text-gray-600 font-medium disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
              →
            </button>
          </div>
        </div>

        {/* ── Card area ── */}
        <div className="relative overflow-hidden">
          <div className={`transition-transform duration-[260ms] ease-in-out ${animating ? exitTranslate + " opacity-0" : "translate-x-0 opacity-100"}`}>
            {isSkeletonSlot ? (
              <div className="px-5 py-4 space-y-3">
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 rounded-full bg-gray-200 animate-pulse flex-shrink-0 mt-0.5" />
                  <div className="flex-1 space-y-2">
                    <div className="h-3 w-3/5 rounded bg-gray-200 animate-pulse" />
                    <div className="h-3 w-2/5 rounded bg-gray-200 animate-pulse" />
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 rounded-full bg-gray-200 animate-pulse flex-shrink-0 mt-0.5" />
                  <div className="flex-1 space-y-2">
                    <div className="h-3 w-full rounded bg-gray-200 animate-pulse" />
                    <div className="h-3 w-full rounded bg-gray-200 animate-pulse" />
                    <div className="h-3 w-3/5 rounded bg-gray-200 animate-pulse" />
                  </div>
                </div>
              </div>
            ) : currentPair ? (
              <div className="px-5 py-4 space-y-3">

                {/* Question */}
                <div className="flex items-start gap-3">
                  <span className="w-6 h-6 rounded-full bg-gray-200 text-gray-500 text-xs font-bold flex items-center justify-center select-none flex-shrink-0 mt-0.5">Q</span>
                  <div className="flex-1 min-w-0">
                    {editing ? (
                      <textarea
                        value={question}
                        onChange={(e) => setQuestion(e.target.value)}
                        rows={2}
                        className="w-full rounded-xl border-2 border-blue-300 focus:border-blue-500 bg-gray-50 px-3 py-2 text-sm leading-relaxed outline-none resize-none transition-colors"
                      />
                    ) : (
                      <div className={`rounded-xl px-3 py-2.5 text-sm leading-relaxed whitespace-pre-wrap transition-colors ${
                        currentPair.validated
                          ? "bg-green-50 border border-green-200 border-l-4 border-l-green-400"
                          : "bg-red-50 border border-red-100"
                      }`}>
                        {currentPair.question}
                      </div>
                    )}
                  </div>
                </div>

                {/* Answer */}
                <div className="flex items-start gap-3">
                  <span className="w-6 h-6 rounded-full bg-gray-300 text-gray-600 text-xs font-bold flex items-center justify-center select-none flex-shrink-0 mt-0.5">A</span>
                  <div className="flex-1 min-w-0">
                    {editing ? (
                      <textarea
                        value={answer}
                        onChange={(e) => setAnswer(e.target.value)}
                        rows={4}
                        className="w-full rounded-xl border-2 border-blue-300 focus:border-blue-500 bg-white px-3 py-2 text-sm leading-relaxed outline-none resize-none transition-colors"
                      />
                    ) : (
                      <div className={`rounded-xl px-3 py-2.5 text-sm leading-relaxed whitespace-pre-wrap transition-colors ${
                        currentPair.validated
                          ? "bg-green-50 border border-green-200 border-l-4 border-l-green-400"
                          : "bg-red-50 border border-red-100"
                      }`}>
                        {currentPair.answer}
                      </div>
                    )}
                  </div>
                </div>

                {/* Action row */}
                <div className="flex items-center gap-2 pt-1 pl-9 flex-wrap">
                  {editing ? (
                    <>
                      <button
                        onClick={handleSave}
                        disabled={saving}
                        className="text-sm px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-medium disabled:opacity-50 transition-colors"
                      >
                        {saving ? "Saving…" : "Save"}
                      </button>
                      <button
                        onClick={() => { setQuestion(currentPair.question); setAnswer(currentPair.answer); setEditing(false); }}
                        className="text-sm px-4 py-2 rounded-lg bg-white border border-gray-300 hover:bg-gray-100 text-gray-600 font-medium transition-colors"
                      >
                        Cancel
                      </button>
                      <span className="text-xs text-gray-400 ml-1">← → saves &amp; navigates</span>
                    </>
                  ) : (
                    <>
                      {/* Validate toggle */}
                      {currentPair.validated ? (
                        <button
                          onClick={handleValidateToggle}
                          className="text-sm px-4 py-2 rounded-lg bg-green-100 border border-green-300 text-green-800 hover:bg-green-200 font-medium transition-colors"
                        >
                          ✓ Validated
                        </button>
                      ) : (
                        <button
                          onClick={handleValidateToggle}
                          className="text-sm px-4 py-2 rounded-lg bg-green-600 hover:bg-green-700 text-white font-medium transition-colors"
                        >
                          Mark validated
                        </button>
                      )}

                      {/* Edit */}
                      <button
                        onClick={() => setEditing(true)}
                        className="text-sm px-4 py-2 rounded-lg bg-white border border-gray-300 hover:bg-gray-100 text-gray-700 font-medium transition-colors"
                      >
                        Edit
                      </button>

                      {/* Delete */}
                      {confirmDelete ? (
                        <>
                          <button
                            onClick={handleDelete}
                            disabled={deleting}
                            className="text-sm px-4 py-2 rounded-lg bg-red-600 hover:bg-red-700 text-white font-medium disabled:opacity-50 transition-colors"
                          >
                            {deleting ? "Deleting…" : "Confirm delete"}
                          </button>
                          <button
                            onClick={() => setConfirmDelete(false)}
                            className="text-sm px-3 py-2 rounded-lg bg-white border border-gray-300 hover:bg-gray-100 text-gray-600 font-medium transition-colors"
                          >
                            Cancel
                          </button>
                        </>
                      ) : (
                        <button
                          onClick={() => setConfirmDelete(true)}
                          className="text-sm px-3 py-2 rounded-lg bg-white border border-gray-200 hover:border-red-300 hover:bg-red-50 text-gray-400 hover:text-red-500 font-medium transition-colors ml-auto"
                          title="Delete this pair"
                        >
                          ✕
                        </button>
                      )}
                    </>
                  )}
                </div>
              </div>
            ) : pairs.length === 0 && !synthLoading ? (
              <div className="px-5 py-4 text-sm text-gray-400">
                No Q&amp;A pairs could be generated for this passage.
              </div>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function ChatPage() {
  const [sessions, setSessions]         = useState<Session[]>([]);
  const [session, setSession]           = useState<Session | null>(null);
  const [messages, setMessages]         = useState<Message[]>([]);
  const [input, setInput]               = useState("");
  const [numQa, setNumQa]               = useState(5);
  const [loading, setLoading]           = useState(false);
  const [error, setError]               = useState<string | null>(null);
  const [health, setHealth]             = useState<ModelHealth | null>(null);
  const [trainStatus, setTrainStatus]   = useState<TrainStatus | null>(null);
  const [qaCount, setQaCount]           = useState<QACount | null>(null);
  const [outputFiles, setOutputFiles]   = useState<OutputFile[]>([]);
  const [adapters, setAdapters]         = useState<Adapter[]>([{ id: "base", version: "Base model", path: "", is_base: true, trained_at: null }]);
  const [selectedAdapter, setSelectedAdapter] = useState<string>("base");
  const [trainingSystemPrompt, setTrainingSystemPrompt] = useState<string>("");
  const [systemPrompt, setSystemPrompt] = useState<string>("");
  const [lastPoll, setLastPoll]         = useState<Date | null>(null);
  const [panelOpen, setPanelOpen]       = useState(true);
  const [startingTraining, setStartingTraining] = useState(false);
  const [uploadLoading, setUploadLoading]       = useState(false);
  const bottomRef    = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const prevSessionStateRef = useRef<SessionState | null>(null);

  // ── Scroll to bottom ──
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ── Fetch helpers ──
  const fetchHealth = async () => {
    try {
      const resp = await fetch(`${MODEL_SERVER_URL}/health`, { signal: AbortSignal.timeout(4000) });
      if (resp.ok) setHealth(await resp.json());
    } catch {}
  };

  const fetchTrainStatus = async () => {
    try {
      const resp = await fetch(`${MODEL_SERVER_URL}/train/status`, { signal: AbortSignal.timeout(4000) });
      if (resp.ok) setTrainStatus(await resp.json());
    } catch {}
  };

  const fetchOutputFiles = async () => {
    try {
      const resp = await fetch(`${API_URL}/outputs`, { signal: AbortSignal.timeout(4000) });
      if (resp.ok) setOutputFiles(await resp.json());
    } catch {}
  };

  const fetchQaCount = useCallback(async (sessionId: string) => {
    try {
      const resp = await fetch(`${API_URL}/sessions/${sessionId}/qa/count`, { signal: AbortSignal.timeout(4000) });
      if (resp.ok) setQaCount(await resp.json());
    } catch {}
  }, []);

  const fetchAdapters = async () => {
    const defaultAdapter = { id: "base", version: "Base model", path: "", is_base: true, trained_at: null };
    try {
      const resp = await fetch(`${MODEL_SERVER_URL}/adapters`, { signal: AbortSignal.timeout(4000) });
      if (resp.ok) {
        const data = await resp.json();
        const filtered = (data.adapters || []).filter((a: Adapter) => a.id !== "base");
        if (filtered.length > 0) { setAdapters([defaultAdapter, ...filtered]); return; }
      }
    } catch {}
    try {
      const resp = await fetch(`${API_URL}/adapters`, { signal: AbortSignal.timeout(4000) });
      if (resp.ok) {
        const data = await resp.json();
        const filtered = (data.adapters || []).filter((a: Adapter) => a.id !== "base");
        if (filtered.length > 0) { setAdapters([defaultAdapter, ...filtered]); return; }
      }
    } catch {}
    setAdapters([defaultAdapter]);
  };

  // ── Load sessions list ──
  const fetchSessions = useCallback(async () => {
    try {
      const resp = await fetch(`${API_URL}/sessions?limit=20`);
      if (resp.ok) {
        const data = await resp.json();
        const list: Session[] = Array.isArray(data) ? data : (data.sessions ?? []);
        setSessions(list);
        return list;
      }
    } catch {}
    return [];
  }, []);

  // ── Load turns with QA pairs ──
  const fetchTurns = useCallback(async (sessionId: string) => {
    try {
      const resp = await fetch(`${API_URL}/sessions/${sessionId}/turns`);
      if (!resp.ok) return;
      const turns: { role: "user" | "assistant" | "system"; content: string; id: string; qa_pairs?: QAPair[] }[] = await resp.json();
      if (turns.length > 0) {
        setMessages(turns.map((t) => ({
          role: t.role,
          content: t.content,
          id: t.id,
          qaPairs: t.qa_pairs && t.qa_pairs.length > 0 ? t.qa_pairs : undefined,
        })));
      }
    } catch {}
  }, []);

  // ── Restore session on mount ──
  useEffect(() => {
    const restore = async () => {
      const savedId = localStorage.getItem("lora_session_id");
      const allSessions = await fetchSessions();
      if (savedId && allSessions.length > 0) {
        const found = allSessions.find(s => s.id === savedId);
        if (found) {
          setSession(found);
          await fetchTurns(found.id);
          await fetchQaCount(found.id);
          return;
        }
      }
      const target = allSessions.find(s => !["READY", "FAILED"].includes(s.state)) ?? allSessions[0];
      if (target) {
        setSession(target);
        await fetchTurns(target.id);
        await fetchQaCount(target.id);
      } else {
        await createSession();
      }
    };
    restore();
  }, [fetchSessions]);

  useEffect(() => {
    if (session) localStorage.setItem("lora_session_id", session.id);
  }, [session]);

  useEffect(() => {
    if (session) prevSessionStateRef.current = session.state;
  }, [session?.id]);

  // ── Poll diagnostics ──
  useEffect(() => {
    const poll = async () => {
      await Promise.all([fetchHealth(), fetchTrainStatus(), fetchOutputFiles(), fetchAdapters()]);
      setLastPoll(new Date());
    };
    poll();
    const id = setInterval(poll, POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, []);

  // ── Poll QA count when session is active ──
  useEffect(() => {
    if (!session) return;
    fetchQaCount(session.id);
    const id = setInterval(() => fetchQaCount(session.id), 10000);
    return () => clearInterval(id);
  }, [session?.id]);

  // ── Poll session state while training/evaluating/deploying ──
  useEffect(() => {
    if (!session) return;
    if (["ACTIVE", "PRE_SLEEP_WARNING", "INSUFFICIENT_DATA", "READY", "FAILED"].includes(session.state)) return;

    const id = setInterval(async () => {
      try {
        const resp = await fetch(`${API_URL}/sessions/${session.id}`);
        if (resp.ok) {
          const data: Session = await resp.json();
          const prev = prevSessionStateRef.current;
          if (data.state === "FAILED" && prev !== "FAILED") {
            const reason = data.failure_reason ?? "An unknown error occurred.";
            setMessages((msgs) => [...msgs, { role: "system", content: `Training failed: ${reason} You can keep chatting or type /sleep to retry.` }]);
          }
          if (data.state === "READY" && prev !== "READY") {
            setMessages((msgs) => [...msgs, { role: "system", content: "Training complete! A new adapter is live. Start a new session to use it." }]);
          }
          prevSessionStateRef.current = data.state;
          setSession(data);
          if (["READY", "FAILED", "ACTIVE"].includes(data.state)) fetchAdapters();
        }
      } catch {}
    }, 3000);
    return () => clearInterval(id);
  }, [session?.id, session?.state]);

  // ── Create session ──
  const createSession = useCallback(async (adapterId?: string) => {
    if (adapterId && adapterId !== "base") {
      try {
        await fetch(`${API_URL}/load_adapter`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ adapter_id: adapterId }),
        });
      } catch {}
    } else if (adapterId === "base") {
      try {
        await fetch(`${MODEL_SERVER_URL}/reload_adapter`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ adapter_dir: "base" }),
        });
      } catch {}
    }
    if (adapterId) setSelectedAdapter(adapterId);
    try {
      const body: Record<string, string> = {};
      if (adapterId) body.adapter_id = adapterId;
      if (systemPrompt) body.system_prompt = systemPrompt;
      if (trainingSystemPrompt) body.training_system_prompt = trainingSystemPrompt;
      const resp = await fetch(`${API_URL}/sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data: Session = await resp.json();
      setSession(data);
      setMessages([]);
      setQaCount(null);
      setError(null);
      await fetchSessions();
    } catch {
      setError("Could not create session. Is the backend running?");
    }
  }, [systemPrompt, trainingSystemPrompt, fetchSessions]);

  // ── Update a QA pair in local message state ──
  const handleQAUpdate = useCallback((turnId: string | undefined, qaId: string, updates: Partial<QAPair>) => {
    setMessages((prev) => prev.map((msg) => {
      if (!msg.qaPairs) return msg;
      if (turnId && msg.id !== turnId) return msg;
      return {
        ...msg,
        qaPairs: msg.qaPairs.map((qa) => qa.id === qaId ? { ...qa, ...updates } : qa),
      };
    }));
    // Refresh count after validation change
    if (updates.validated !== undefined && session) fetchQaCount(session.id);
  }, [session, fetchQaCount]);

  // ── Delete a QA pair from local message state ──
  const handleQADelete = useCallback((turnId: string | undefined, qaId: string) => {
    setMessages((prev) => prev.map((msg) => {
      if (!msg.qaPairs) return msg;
      if (turnId && msg.id !== turnId) return msg;
      return { ...msg, qaPairs: msg.qaPairs.filter((qa) => qa.id !== qaId) };
    }));
    if (session) fetchQaCount(session.id);
  }, [session, fetchQaCount]);

  // ── Upload document ──
  const handleFileUpload = useCallback(async (file: File) => {
    if (!session || uploadLoading) return;
    if (!["ACTIVE", "PRE_SLEEP_WARNING", "INSUFFICIENT_DATA", "FAILED"].includes(session.state)) return;

    setUploadLoading(true);
    setError(null);

    // Add a synthetic message entry to host the Q&A cards, labelled with the filename
    const docMsgObj: Message = {
      role: "user",
      content: `[Document: ${file.name}]`,
      synthLoading: true,
    };
    setMessages((prev) => [...prev, docMsgObj]);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("num_qa", String(numQa));

      const resp = await fetch(`${API_URL}/sessions/${session.id}/upload`, {
        method: "POST",
        body: formData,
      });

      if (!resp.ok || !resp.body) {
        const errText = await resp.text().catch(() => `HTTP ${resp.status}`);
        throw new Error(errText || `HTTP ${resp.status}`);
      }

      const reader  = resp.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const raw = decoder.decode(value, { stream: true });
        for (const line of raw.split("\n").filter((l) => l.startsWith("data: "))) {
          try {
            const event = JSON.parse(line.slice(6));

            if (event.type === "start") {
              setMessages((prev) => {
                const copy = [...prev];
                const lastIdx = copy.length - 1;
                copy[lastIdx] = { ...copy[lastIdx], segmentCount: event.segment_count ?? 1, qaPairs: [] };
                return copy;
              });
            }

            if (event.type === "qa_pair" && event.pair) {
              setMessages((prev) => {
                const copy = [...prev];
                const lastIdx = copy.length - 1;
                const existing = copy[lastIdx].qaPairs ?? [];
                copy[lastIdx] = {
                  ...copy[lastIdx],
                  qaPairs: [...existing, event.pair as QAPair],
                };
                return copy;
              });
            }

            if (event.type === "qa_count") {
              setQaCount({
                total_count: event.total,
                validated_count: event.validated,
                min_required: event.min_required,
                ready_to_train: event.ready,
              });
            }

            if (event.type === "end") {
              setMessages((prev) => {
                const copy = [...prev];
                const lastIdx = copy.length - 1;
                copy[lastIdx] = { ...copy[lastIdx], synthLoading: false };
                return copy;
              });
            }

            if (event.type === "error") {
              setError(event.message ?? "Upload failed.");
            }
          } catch {}
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed.");
      // Clear the loading indicator on the synthetic message
      setMessages((prev) => {
        const copy = [...prev];
        const lastIdx = copy.length - 1;
        copy[lastIdx] = { ...copy[lastIdx], synthLoading: false };
        return copy;
      });
    } finally {
      setUploadLoading(false);
      // Reset the file input so the same file can be re-uploaded if needed
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }, [session, uploadLoading, numQa, fetchQaCount]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Send message ──
  const sendMessage = useCallback(async () => {
    if (!input.trim() || !session || loading) return;
    if (!["ACTIVE", "PRE_SLEEP_WARNING", "INSUFFICIENT_DATA", "FAILED"].includes(session.state)) return;

    const userMsg = input.trim();
    setInput("");
    setLoading(true);
    setError(null);

    // Add user message with a placeholder synthesis loading indicator
    const userMsgObj: Message = { role: "user", content: userMsg, synthLoading: true };
    setMessages((prev) => [...prev, userMsgObj]);

    // Handle /sleep command
    if (userMsg.trim() === "/sleep") {
      try {
        const resp = await fetch(`${API_URL}/sessions/${session.id}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: userMsg, num_qa: numQa }),
        });
        if (resp.ok && resp.body) {
          const reader = resp.body.getReader();
          const decoder = new TextDecoder();
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            const raw = decoder.decode(value, { stream: true });
            for (const line of raw.split("\n").filter((l) => l.startsWith("data: "))) {
              try {
                const event = JSON.parse(line.slice(6));
                if (["sleeping", "sleep_ack", "validating"].includes(event.type)) {
                  const msg = event.message ?? "Training pipeline started.";
                  setMessages((prev) => [...prev.slice(0, -1), { ...prev[prev.length - 1], synthLoading: false }, { role: "system", content: msg }]);
                  setSession((prev) => prev ? { ...prev, state: event.type === "validating" ? "VALIDATING" : "TRAINING" } : prev);
                }
              } catch {}
            }
          }
        }
      } catch {
        setError("Request failed.");
      }
      setLoading(false);
      return;
    }

    try {
      const resp = await fetch(`${API_URL}/sessions/${session.id}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMsg, num_qa: numQa }),
      });
      if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const raw = decoder.decode(value, { stream: true });
        for (const line of raw.split("\n").filter((l) => l.startsWith("data: "))) {
          try {
            const event = JSON.parse(line.slice(6));

            // First event — initialise qaPairs array and store expected segment count
            if (event.type === "start") {
              setMessages((prev) => {
                const copy = [...prev];
                const lastIdx = copy.length - 1;
                copy[lastIdx] = {
                  ...copy[lastIdx],
                  segmentCount: event.segment_count ?? 1,
                  qaPairs: [],
                };
                return copy;
              });
            }

            // heartbeat — keep-alive tick, no UI change needed
            // if (event.type === "heartbeat") { /* no-op */ }

            // One pair arrived — append it immediately so the card appears now
            if (event.type === "qa_pair" && event.pair) {
              const pair: QAPair = {
                id: event.pair.id,
                question: event.pair.question,
                answer: event.pair.answer,
                validated: event.pair.validated ?? false,
                edited: event.pair.edited ?? false,
              };
              setMessages((prev) => {
                const copy = [...prev];
                const lastIdx = copy.length - 1;
                const existing = copy[lastIdx].qaPairs ?? [];
                copy[lastIdx] = { ...copy[lastIdx], qaPairs: [...existing, pair] };
                return copy;
              });
            }

            if (event.type === "qa_count") {
              setQaCount({
                total_count: event.total,
                validated_count: event.validated,
                min_required: event.min_required,
                ready_to_train: event.ready,
              });
            }

            // Backend error event — show message, stop loading
            if (event.type === "error") {
              setError(`Synthesis failed: ${event.message ?? "unknown error"}`);
              setMessages((prev) => {
                const copy = [...prev];
                const lastIdx = copy.length - 1;
                copy[lastIdx] = { ...copy[lastIdx], synthLoading: false };
                return copy;
              });
            }

            // Stream complete — remove the loading skeleton
            if (event.type === "end") {
              setMessages((prev) => {
                const copy = [...prev];
                const lastIdx = copy.length - 1;
                copy[lastIdx] = { ...copy[lastIdx], synthLoading: false };
                return copy;
              });
            }
          } catch {}
        }
      }
    } catch {
      setError("Request failed — check backend connection.");
      setMessages((prev) => {
        const copy = [...prev];
        if (copy.length > 0) copy[copy.length - 1] = { ...copy[copy.length - 1], synthLoading: false };
        return copy;
      });
    } finally {
      setLoading(false);
    }
  }, [input, session, loading]);

  // ── Start Training ──
  const handleStartTraining = useCallback(async () => {
    if (!session || startingTraining) return;
    setStartingTraining(true);
    try {
      const resp = await fetch(`${API_URL}/sessions/${session.id}/start-training`, { method: "POST" });
      if (resp.ok) {
        setSession((prev) => prev ? { ...prev, state: "TRAINING" } : prev);
        setMessages((prev) => [...prev, { role: "system", content: "Training started! The model is being fine-tuned on your Q&A pairs. Check the diagnostics panel for progress." }]);
        await fetchSessions();
        await fetchTrainStatus();
      } else {
        const err = await resp.json().catch(() => ({ detail: "Unknown error" }));
        setError(`Could not start training: ${err.detail ?? resp.status}`);
      }
    } catch {
      setError("Could not start training — check backend connection.");
    }
    setStartingTraining(false);
  }, [session, startingTraining, fetchSessions]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  };

  const isAcceptingInput = session &&
    ["ACTIVE", "PRE_SLEEP_WARNING", "INSUFFICIENT_DATA", "FAILED"].includes(session.state) && !loading;

  const canStartTraining = !!(qaCount?.ready_to_train) && session &&
    ["ACTIVE", "PRE_SLEEP_WARNING", "INSUFFICIENT_DATA", "FAILED", "VALIDATING"].includes(session.state) &&
    !startingTraining;

  const tokenPct = session ? Math.min((session.total_tokens / session.max_tokens) * 100, 100) : 0;

  return (
    <div className="flex h-screen bg-gray-50 font-sans overflow-hidden">

      {/* ── Main chat area ── */}
      <div className="flex flex-col flex-1 min-w-0">

        {/* Header */}
        <header className="flex items-center justify-between px-6 py-3 bg-white border-b border-gray-200 shadow-sm flex-shrink-0">
          <div className="flex items-center gap-3">
            <h1 className="text-lg font-semibold text-gray-900">LoRA Chat</h1>
            {session && (
              <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${STATE_COLORS[session.state]}`}>
                {STATE_LABELS[session.state]}
              </span>
            )}
          </div>

          <div className="flex items-center gap-3">
            {/* Session selector */}
            <div className="flex items-center gap-1">
              <select
                key={sessions.length}
                value={session?.id ?? ""}
                onChange={async (e) => {
                  const s = sessions.find(s => s.id === e.target.value);
                  if (s) {
                    setSession(s);
                    await fetchTurns(s.id);
                    await fetchQaCount(s.id);
                  }
                }}
                className="text-xs px-2 py-1.5 rounded-md border border-gray-300 bg-white text-gray-700"
              >
                {!session && <option value="">No session</option>}
                {sessions.map((s) => (
                  <option key={s.id} value={s.id}>
                    {STATE_LABELS[s.state].replace(/[^\w\s]/g, "").trim()} {s.id.slice(0, 8)}
                  </option>
                ))}
              </select>
              <button onClick={fetchSessions} className="text-xs px-2 py-1.5 rounded-md bg-gray-100 hover:bg-gray-200 text-gray-600" title="Refresh sessions">↻</button>
            </div>

            {/* Token gauge */}
            {session && (
              <div className="flex items-center gap-2 text-xs text-gray-500">
                <span>{session.total_tokens} / {session.max_tokens}</span>
                <div className="w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div className={`h-full rounded-full transition-all ${tokenPct > 85 ? "bg-red-400" : tokenPct > 70 ? "bg-yellow-400" : "bg-blue-400"}`} style={{ width: `${tokenPct}%` }} />
                </div>
              </div>
            )}

            {/* Start Training button */}
            {session && (
              <div className="flex items-center gap-1">
                <button
                  onClick={handleStartTraining}
                  disabled={!canStartTraining}
                  title={!qaCount?.ready_to_train
                    ? `Need ${(qaCount?.min_required ?? MIN_TRAINING_SAMPLES) - (qaCount?.validated_count ?? 0)} more validated Q&A pairs`
                    : "Start training the model on your Q&A pairs"}
                  className={`text-xs px-3 py-1.5 rounded-md font-medium transition-colors ${
                    canStartTraining
                      ? "bg-green-600 hover:bg-green-700 text-white"
                      : "bg-gray-100 text-gray-400 cursor-not-allowed"
                  }`}
                >
                  {startingTraining
                    ? "Starting…"
                    : qaCount
                    ? `Start Training (${qaCount.validated_count}/${qaCount.min_required})`
                    : "Start Training"}
                </button>
              </div>
            )}

            {/* New session dropdown */}
            <div className="relative">
              <button
                onClick={() => document.getElementById("new-session-dd")?.classList.toggle("hidden")}
                className="text-xs px-3 py-1.5 rounded-md bg-blue-600 hover:bg-blue-700 text-white transition-colors"
              >
                New session ▾
              </button>
              <div id="new-session-dd" className="hidden absolute right-0 top-full mt-1 w-72 bg-white border border-gray-200 rounded-lg shadow-lg z-50 p-3 space-y-2">
                <div className="text-xs text-gray-500 font-medium">Chat system prompt (optional)</div>
                <textarea value={systemPrompt} onChange={(e) => setSystemPrompt(e.target.value)} placeholder="You are a helpful AI assistant..." className="w-full text-xs px-2 py-1.5 rounded border border-gray-200 text-gray-700 resize-none" rows={2} />
                <div className="text-xs text-gray-500 font-medium border-t border-gray-100 pt-2">Training system prompt (optional)</div>
                <textarea value={trainingSystemPrompt} onChange={(e) => setTrainingSystemPrompt(e.target.value)} placeholder="Prompt used when training the model..." className="w-full text-xs px-2 py-1.5 rounded border border-gray-200 text-gray-700 resize-none" rows={2} />
                <div className="text-xs text-gray-500 font-medium border-t border-gray-100 pt-2">Select adapter</div>
                <button onClick={() => { createSession("base"); document.getElementById("new-session-dd")?.classList.add("hidden"); }} className="w-full text-left px-2 py-1.5 text-xs hover:bg-gray-50 text-gray-700 rounded">Base model</button>
                {adapters.filter(a => a.id !== "base").map((a) => (
                  <button key={a.id} onClick={() => { createSession(a.id); document.getElementById("new-session-dd")?.classList.add("hidden"); }} className="w-full text-left px-2 py-1.5 text-xs hover:bg-gray-50 text-gray-700 rounded">
                    {a.version}{a.is_current ? " (live)" : ""}
                  </button>
                ))}
              </div>
            </div>

            <button onClick={() => setPanelOpen((v) => !v)} className="text-xs px-3 py-1.5 rounded-md bg-gray-100 hover:bg-gray-200 text-gray-700 transition-colors" title="Toggle diagnostics panel">
              {panelOpen ? "Hide panel" : "Show panel"}
            </button>
          </div>
        </header>

        {/* Messages */}
        <main className="flex-1 overflow-y-auto px-4 py-6">
          <div className="max-w-3xl mx-auto w-full space-y-6">
            {messages.length === 0 && !error && (
              <div className="text-center text-gray-400 text-sm mt-16 space-y-2">
                <p className="font-medium text-gray-500">Send a passage or upload a document to generate training data</p>
                <p className="text-xs">Paste text directly or use the paperclip button to upload a PDF, DOCX, TXT, or MD file.<br />
                  Each source generates Q&A pairs for fine-tuning. When you have {MIN_TRAINING_SAMPLES}+ validated pairs, the <span className="font-medium text-green-700">Start Training</span> button will activate.</p>
                <p className="text-xs text-gray-400 mt-3">Type <code className="bg-gray-100 px-1 rounded">/sleep</code> to immediately start training with all current Q&A pairs.</p>
              </div>
            )}
            {error && (
              <div className="bg-red-50 border border-red-200 text-red-700 text-sm px-4 py-3 rounded-lg">{error}</div>
            )}
            {messages.map((msg, i) => {
              if (msg.role === "system") {
                return <div key={i} className="text-center text-sm text-gray-500 italic py-1">{msg.content}</div>;
              }
              if (msg.role === "user") {
                const isDocUpload = msg.content.startsWith("[Document: ") && msg.content.endsWith("]");
                const docName = isDocUpload ? msg.content.slice(11, -1) : null;
                return (
                  <div key={i} className="space-y-2">
                    {/* User bubble — document uploads get a distinct pill style */}
                    <div className="flex justify-end">
                      {isDocUpload ? (
                        <div className="flex items-center gap-2 px-4 py-2.5 rounded-2xl rounded-br-sm bg-gray-100 border border-gray-300 text-gray-700 text-sm max-w-[80%]">
                          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4 flex-shrink-0 text-gray-500"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                          <span className="truncate font-medium">{docName}</span>
                        </div>
                      ) : (
                        <div className="max-w-[80%] px-4 py-3 rounded-2xl rounded-br-sm text-sm leading-relaxed whitespace-pre-wrap bg-blue-600 text-white">
                          {msg.content}
                        </div>
                      )}
                    </div>

                    {/* Inline QA deck */}
                    {(msg.qaPairs !== undefined || msg.synthLoading) && session && (
                      <InlineDeck
                        pairs={msg.qaPairs ?? []}
                        sessionId={session.id}
                        turnId={msg.id}
                        synthLoading={!!msg.synthLoading}
                        segmentCount={msg.segmentCount ?? 1}
                        onUpdate={(qaId, updates) => handleQAUpdate(msg.id, qaId, updates)}
                        onDelete={(qaId) => handleQADelete(msg.id, qaId)}
                      />
                    )}

                    {/* Pre-start loading indicator */}
                    {msg.synthLoading && msg.qaPairs === undefined && (
                      <div className="flex items-center gap-2 mt-1 pl-1">
                        <span className="relative flex h-2 w-2 flex-shrink-0">
                          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75" />
                          <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500" />
                        </span>
                        <span className="text-xs text-blue-500 font-medium">
                          {isDocUpload ? "Extracting & analysing document…" : "Analysing passage…"}
                        </span>
                      </div>
                    )}
                  </div>
                );
              }
              // assistant messages (from /sleep ack etc) should not normally appear in new flow
              return (
                <div key={i} className="flex justify-start">
                  <div className="max-w-[75%] px-4 py-3 rounded-2xl rounded-bl-sm text-sm leading-relaxed whitespace-pre-wrap bg-white border border-gray-200 text-gray-800 shadow-sm">
                    {msg.content}
                    {msg.streaming && <span className="inline-block w-1 h-4 ml-0.5 bg-current opacity-70 animate-pulse" />}
                  </div>
                </div>
              );
            })}
            <div ref={bottomRef} />
          </div>
        </main>

        {/* Input */}
        <footer className="bg-white border-t border-gray-200 px-4 py-4 flex-shrink-0">
          {/* Hidden file input — triggered by the paperclip button */}
          <input
            ref={fileInputRef}
            type="file"
            accept=".txt,.md,.pdf,.docx"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleFileUpload(file);
            }}
          />
          <div className="max-w-3xl mx-auto flex gap-3 items-end">
            {/* Paperclip upload button */}
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={!isAcceptingInput || uploadLoading}
              title={uploadLoading ? "Uploading…" : "Upload a document (PDF, DOCX, TXT, MD)"}
              className="flex items-center justify-center w-11 h-11 rounded-xl border border-gray-300 bg-white hover:bg-gray-50 text-gray-500 hover:text-gray-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors shrink-0"
            >
              {uploadLoading
                ? <span className="animate-spin text-blue-500 text-base leading-none">⟳</span>
                : <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>
              }
            </button>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                !isAcceptingInput
                  ? ["SLEEPING", "TRAINING", "EVALUATING", "DEPLOYING"].includes(session?.state ?? "")
                    ? "Training in progress — check the panel →"
                    : session?.state === "READY"
                    ? "New adapter is live — start a new session!"
                    : "Session closed"
                  : "Paste a passage to generate Q&A training data… (Enter to send)"
              }
              disabled={!isAcceptingInput}
              rows={1}
              className="flex-1 resize-none rounded-xl border border-gray-300 px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-50 disabled:text-gray-400 disabled:cursor-not-allowed max-h-48 overflow-y-auto"
              style={{ minHeight: "44px" }}
              onInput={(e) => {
                const t = e.target as HTMLTextAreaElement;
                t.style.height = "auto";
                t.style.height = Math.min(t.scrollHeight, 192) + "px";
              }}
            />
            <div className="flex flex-col items-center gap-0.5 shrink-0">
              <label className="text-xs text-gray-400 leading-none">Q&amp;A</label>
              <input
                type="number"
                min={1}
                max={20}
                value={numQa}
                onChange={(e) => setNumQa(Math.min(20, Math.max(1, parseInt(e.target.value) || 1)))}
                disabled={!isAcceptingInput}
                title="Number of Q&A pairs to generate"
                className="w-14 rounded-lg border border-gray-300 px-2 py-2.5 text-sm text-center focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-50 disabled:text-gray-400 disabled:cursor-not-allowed"
              />
            </div>
            <button
              onClick={sendMessage}
              disabled={!isAcceptingInput || !input.trim()}
              className="px-5 py-3 rounded-xl bg-blue-600 text-white text-sm font-medium hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors shrink-0"
            >
              {loading ? "…" : "Send"}
            </button>
          </div>
          {/* Training readiness hint */}
          {session && qaCount && !qaCount.ready_to_train && qaCount.total_count > 0 && (
            <p className="text-center text-xs text-amber-600 mt-2">
              {qaCount.validated_count} of {qaCount.min_required} validated pairs needed to start training — validate Q&A cards above or send more passages.
            </p>
          )}
          {session && qaCount?.ready_to_train && (
            <p className="text-center text-xs text-green-600 mt-2">
              {qaCount.validated_count} validated pairs ready — click <span className="font-medium">Start Training</span> when you&apos;re happy with the data.
            </p>
          )}
        </footer>
      </div>

      {/* ── Diagnostic panel ── */}
      {panelOpen && (
        <DiagnosticPanel
          session={session}
          health={health}
          trainStatus={trainStatus}
          lastPoll={lastPoll}
          selectedAdapter={selectedAdapter}
          adapters={adapters}
          qaCount={qaCount}
          onRestartTraining={fetchTrainStatus}
        />
      )}

      <HelpPanel />
    </div>
  );
}
