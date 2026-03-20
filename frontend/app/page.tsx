"use client";

import { useState, useEffect, useRef, useCallback } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const MODEL_SERVER_URL = process.env.NEXT_PUBLIC_MODEL_SERVER_URL ?? "http://localhost:8001";
const POLL_INTERVAL_MS = 5000;

type SessionState =
  | "ACTIVE"
  | "PRE_SLEEP_WARNING"
  | "SLEEPING"
  | "TRAINING"
  | "EVALUATING"
  | "DEPLOYING"
  | "READY"
  | "FAILED";

interface Message {
  role: "user" | "assistant" | "system";
  content: string;
  streaming?: boolean;
}

interface Session {
  id: string;
  state: SessionState;
  total_tokens: number;
  max_tokens: number;
  created_at: string;
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

const STATE_COLORS: Record<SessionState, string> = {
  ACTIVE:            "bg-green-100 text-green-800",
  PRE_SLEEP_WARNING: "bg-yellow-100 text-yellow-800",
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
  SLEEPING:          "Sleeping — training queued",
  TRAINING:          "Training…",
  EVALUATING:        "Evaluating…",
  DEPLOYING:         "Deploying…",
  READY:             "Ready — new adapter live",
  FAILED:            "Failed",
};

// ── Diagnostic panel sub-components ──────────────────────────────────────────

function GaugeBar({ value, max, color }: { value: number; max: number; color: string }) {
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
      <div
        className={`h-full rounded-full transition-all duration-500 ${color}`}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

function StatRow({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-gray-100 last:border-0">
      <span className="text-xs text-gray-500">{label}</span>
      <div className="text-right">
        <span className="text-xs font-medium text-gray-800">{value}</span>
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

function PipelineStep({
  label,
  state,
}: {
  label: string;
  state: "done" | "active" | "pending" | "failed";
}) {
  const icons = {
    done:    <span className="text-green-500 text-sm">✓</span>,
    active:  <span className="animate-spin text-blue-500 text-sm inline-block">⟳</span>,
    pending: <span className="text-gray-300 text-sm">○</span>,
    failed:  <span className="text-red-500 text-sm">✗</span>,
  };
  const labels = {
    done:    "text-green-700",
    active:  "text-blue-700 font-medium",
    pending: "text-gray-400",
    failed:  "text-red-600",
  };
  return (
    <div className="flex items-center gap-2 py-1">
      <div className="w-4 flex justify-center">{icons[state]}</div>
      <span className={`text-xs ${labels[state]}`}>{label}</span>
    </div>
  );
}

// Map session state → which pipeline steps are done/active/pending
function getPipelineSteps(sessionState: SessionState, trainStatus?: TrainStatus) {
  const steps = [
    "Extract candidates",
    "Curate & score",
    "Build dataset",
    "Train (RTX 4060)",
    "Evaluate",
    "Deploy adapter",
  ];

  const stateOrder: SessionState[] = [
    "SLEEPING", "TRAINING", "EVALUATING", "DEPLOYING", "READY", "FAILED",
  ];
  const rank = stateOrder.indexOf(sessionState);

  return steps.map((label, i) => {
    if (sessionState === "FAILED") {
      return { label, state: i < rank ? "done" : i === rank ? "failed" : "pending" } as const;
    }
    if (i < rank)       return { label, state: "done"    } as const;
    if (i === rank)     return { label, state: "active"  } as const;
    if (sessionState === "READY" && i <= 5) return { label, state: "done" } as const;
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
  session,
  health,
  trainStatus,
  outputFiles,
  lastPoll,
}: {
  session: Session | null;
  health: ModelHealth | null;
  trainStatus: TrainStatus | null;
  outputFiles: OutputFile[];
  lastPoll: Date | null;
}) {
  const gpu = health?.gpu ?? null;
  const vramPct = gpu ? gpu.vram_used_gb / gpu.vram_total_gb : 0;
  const tokenPct = session ? session.total_tokens / session.max_tokens : 0;

  const isTraining = trainStatus?.status === "running";
  const pipelineSteps = session ? getPipelineSteps(session.state, trainStatus ?? undefined) : [];

  return (
    <aside className="w-72 min-w-72 h-screen overflow-y-auto bg-gray-50 border-l border-gray-200 px-4 py-4 flex flex-col gap-0 text-sm">

      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <h2 className="font-semibold text-gray-700 text-sm">Diagnostics</h2>
        {lastPoll && (
          <span className="text-xs text-gray-400">
            updated {elapsed(lastPoll.toISOString())} ago
          </span>
        )}
      </div>

      {/* ── Model server ── */}
      <SectionHeader
        title="Model server"
        dot={health?.status === "ok" ? "bg-green-400" : "bg-red-400"}
      />
      <div className="bg-white rounded-lg border border-gray-200 px-3 py-2">
        <StatRow
          label="Status"
          value={health ? (health.model_loaded ? "Ready" : "Loading…") : "Unreachable"}
        />
        <StatRow
          label="Adapter"
          value={health?.adapter
            ? health.adapter.split(/[\\/]/).slice(-2).join("/")
            : "Base model"}
        />
        <StatRow
          label="Training"
          value={health?.training_active ? "🔥 In progress" : "Idle"}
        />
      </div>

      {/* ── GPU ── */}
      {gpu && (
        <>
          <SectionHeader title="GPU" dot="bg-purple-400" />
          <div className="bg-white rounded-lg border border-gray-200 px-3 py-2">
            <StatRow label="Device" value={gpu.name} />
            <StatRow
              label="VRAM used"
              value={`${gpu.vram_used_gb.toFixed(1)} / ${gpu.vram_total_gb.toFixed(1)} GB`}
            />
            <div className="py-1">
              <GaugeBar
                value={gpu.vram_used_gb}
                max={gpu.vram_total_gb}
                color={vramPct > 0.85 ? "bg-red-400" : vramPct > 0.65 ? "bg-yellow-400" : "bg-purple-400"}
              />
            </div>
            {trainStatus?.vram_used_gb != null && (
              <StatRow
                label="Training VRAM"
                value={`${trainStatus.vram_used_gb.toFixed(1)} GB`}
                sub={trainStatus.vram_free_gb != null
                  ? `${trainStatus.vram_free_gb.toFixed(1)} GB free`
                  : undefined}
              />
            )}
          </div>
        </>
      )}

      {/* ── Session ── */}
      {session && (
        <>
          <SectionHeader title="Session" dot="bg-blue-400" />
          <div className="bg-white rounded-lg border border-gray-200 px-3 py-2">
            <StatRow label="ID" value={session.id.slice(0, 8) + "…"} />
            <StatRow
              label="State"
              value={STATE_LABELS[session.state]}
            />
            <StatRow
              label="Tokens"
              value={`${session.total_tokens} / ${session.max_tokens}`}
            />
            <div className="py-1">
              <GaugeBar
                value={session.total_tokens}
                max={session.max_tokens}
                color={tokenPct > 0.85 ? "bg-red-400" : tokenPct > 0.7 ? "bg-yellow-400" : "bg-blue-400"}
              />
            </div>
            <StatRow
              label="Started"
              value={new Date(session.created_at).toLocaleTimeString()}
            />
          </div>
        </>
      )}

      {/* ── Pipeline ── */}
      {session && !["ACTIVE", "PRE_SLEEP_WARNING"].includes(session.state) && (
        <>
          <SectionHeader title="Pipeline" dot="bg-amber-400" />
          <div className="bg-white rounded-lg border border-gray-200 px-3 py-2">
            {pipelineSteps.map((step) => (
              <PipelineStep key={step.label} label={step.label} state={step.state} />
            ))}
          </div>
        </>
      )}

      {/* ── Training progress ── */}
      {trainStatus && trainStatus.status !== "idle" && (
        <>
          <SectionHeader title="Training" dot={isTraining ? "bg-blue-400 animate-pulse" : "bg-green-400"} />
          <div className="bg-white rounded-lg border border-gray-200 px-3 py-2">
            <StatRow
              label="Status"
              value={trainStatus.status.charAt(0).toUpperCase() + trainStatus.status.slice(1)}
            />
            {trainStatus.run_id && (
              <StatRow label="Run ID" value={trainStatus.run_id.slice(0, 8) + "…"} />
            )}
            <div className="py-1.5">
              <p className="text-xs text-gray-500 mb-0.5">Progress</p>
              <p className="text-xs text-gray-800 leading-relaxed">
                {trainStatus.progress || "—"}
              </p>
            </div>
            {trainStatus.started_at && (
              <StatRow
                label="Running for"
                value={elapsed(trainStatus.started_at)}
              />
            )}
            {trainStatus.finished_at && (
              <StatRow
                label="Finished"
                value={new Date(trainStatus.finished_at).toLocaleTimeString()}
              />
            )}
          </div>
        </>
      )}

      {/* ── Output files ── */}
      {outputFiles.length > 0 && (
        <>
          <SectionHeader title="Output files" dot="bg-teal-400" />
          <div className="bg-white rounded-lg border border-gray-200 px-3 py-2">
            {outputFiles.map((f) => (
              <div key={f.path} className="flex items-center justify-between py-1 border-b border-gray-100 last:border-0">
                <span className="text-xs text-gray-600 truncate max-w-[160px]" title={f.path}>
                  {f.name}
                </span>
                <span className="text-xs text-gray-400 ml-2 flex-shrink-0">{f.size}</span>
              </div>
            ))}
          </div>
        </>
      )}

      {/* ── Quick links ── */}
      <SectionHeader title="Quick links" />
      <div className="bg-white rounded-lg border border-gray-200 px-3 py-2 space-y-1">
        {[
          { label: "Model health",    href: `${MODEL_SERVER_URL}/health` },
          { label: "Train status",    href: `${MODEL_SERVER_URL}/train/status` },
          { label: "API health",      href: `${API_URL}/health` },
          { label: "Training runs",   href: `${API_URL}/training/runs` },
        ].map((link) => (
          <a
            key={link.href}
            href={link.href}
            target="_blank"
            rel="noreferrer"
            className="block text-xs text-blue-600 hover:text-blue-800 hover:underline py-0.5"
          >
            {link.label} ↗
          </a>
        ))}
      </div>

      {/* bottom padding */}
      <div className="h-6" />
    </aside>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function ChatPage() {
  const [session, setSession]         = useState<Session | null>(null);
  const [messages, setMessages]       = useState<Message[]>([]);
  const [input, setInput]             = useState("");
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState<string | null>(null);
  const [health, setHealth]           = useState<ModelHealth | null>(null);
  const [trainStatus, setTrainStatus] = useState<TrainStatus | null>(null);
  const [outputFiles, setOutputFiles] = useState<OutputFile[]>([]);
  const [lastPoll, setLastPoll]       = useState<Date | null>(null);
  const [panelOpen, setPanelOpen]     = useState(true);
  const bottomRef = useRef<HTMLDivElement>(null);

  // ── Create session on mount ──
  useEffect(() => { createSession(); }, []);

  // ── Scroll to bottom ──
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ── Poll diagnostics ──
  useEffect(() => {
    const poll = async () => {
      await Promise.all([fetchHealth(), fetchTrainStatus(), fetchOutputFiles()]);
      setLastPoll(new Date());
    };
    poll();
    const id = setInterval(poll, POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, []);

  // ── Poll session state while not ACTIVE ──
  useEffect(() => {
    if (!session) return;
    if (["ACTIVE", "PRE_SLEEP_WARNING", "READY", "FAILED"].includes(session.state)) return;

    const id = setInterval(async () => {
      try {
        const resp = await fetch(`${API_URL}/sessions/${session.id}`);
        if (resp.ok) {
          const data: Session = await resp.json();
          setSession(data);
        }
      } catch {}
    }, 3000);
    return () => clearInterval(id);
  }, [session?.id, session?.state]);

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

  const createSession = async () => {
    try {
      const resp = await fetch(`${API_URL}/sessions`, { method: "POST" });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data: Session = await resp.json();
      setSession(data);
      setMessages([]);
      setError(null);
    } catch {
      setError("Could not create session. Is the backend running?");
    }
  };

  const sendMessage = useCallback(async () => {
    if (!input.trim() || !session || loading) return;
    if (!["ACTIVE", "PRE_SLEEP_WARNING"].includes(session.state)) return;

    const userMsg = input.trim();
    setInput("");
    setLoading(true);
    setError(null);
    setMessages((prev) => [...prev, { role: "user", content: userMsg }]);
    setMessages((prev) => [...prev, { role: "assistant", content: "", streaming: true }]);

    try {
      const resp = await fetch(`${API_URL}/sessions/${session.id}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMsg }),
      });
      if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);

      const reader  = resp.body.getReader();
      const decoder = new TextDecoder();
      let assistantText = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const raw = decoder.decode(value, { stream: true });
        for (const line of raw.split("\n").filter((l) => l.startsWith("data: "))) {
          try {
            const event = JSON.parse(line.slice(6));
            if (event.type === "chunk") {
              assistantText += event.text;
              setMessages((prev) => {
                const copy = [...prev];
                copy[copy.length - 1] = { role: "assistant", content: assistantText, streaming: true };
                return copy;
              });
            }
            if (event.type === "status") {
              setSession((prev) => prev ? {
                ...prev,
                state: event.session_state ?? prev.state,
                total_tokens: prev.max_tokens - (event.remaining_tokens ?? 0),
              } : prev);
            }
            if (["sleeping", "sleep_ack", "sleep_warning"].includes(event.type)) {
              setMessages((prev) => [...prev, { role: "system", content: event.message ?? "Session sleeping." }]);
              setSession((prev) => prev ? { ...prev, state: "SLEEPING" } : prev);
            }
            if (event.type === "end") {
              setMessages((prev) => {
                const copy = [...prev];
                if (copy[copy.length - 1]?.streaming)
                  copy[copy.length - 1] = { ...copy[copy.length - 1], streaming: false };
                return copy;
              });
            }
          } catch {}
        }
      }
    } catch {
      setError("Request failed — check backend connection.");
      setMessages((prev) => prev.filter((m) => !m.streaming));
    } finally {
      setLoading(false);
    }
  }, [input, session, loading]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  };

  const isAcceptingInput = session &&
    ["ACTIVE", "PRE_SLEEP_WARNING"].includes(session.state) && !loading;
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
          <div className="flex items-center gap-4">
            {session && (
              <div className="flex items-center gap-2 text-xs text-gray-500">
                <span>{session.total_tokens} / {session.max_tokens} tokens</span>
                <div className="w-28 h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all ${
                      tokenPct > 85 ? "bg-red-400" : tokenPct > 70 ? "bg-yellow-400" : "bg-blue-400"
                    }`}
                    style={{ width: `${tokenPct}%` }}
                  />
                </div>
              </div>
            )}
            <button onClick={createSession}
              className="text-xs px-3 py-1.5 rounded-md bg-gray-100 hover:bg-gray-200 text-gray-700 transition-colors">
              New session
            </button>
            <button
              onClick={() => setPanelOpen((v) => !v)}
              className="text-xs px-3 py-1.5 rounded-md bg-gray-100 hover:bg-gray-200 text-gray-700 transition-colors"
              title="Toggle diagnostics panel"
            >
              {panelOpen ? "Hide panel" : "Show panel"}
            </button>
          </div>
        </header>

        {/* Messages */}
        <main className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
          <div className="max-w-3xl mx-auto w-full space-y-4">
            {messages.length === 0 && !error && (
              <p className="text-center text-gray-400 text-sm mt-16">
                Start chatting. Type{" "}
                <code className="bg-gray-100 px-1 rounded">/sleep</code> to end the session and trigger fine-tuning.
              </p>
            )}
            {error && (
              <div className="bg-red-50 border border-red-200 text-red-700 text-sm px-4 py-3 rounded-lg">
                {error}
              </div>
            )}
            {messages.map((msg, i) => {
              if (msg.role === "system") {
                return (
                  <div key={i} className="text-center text-sm text-gray-500 italic py-2">
                    {msg.content}
                  </div>
                );
              }
              return (
                <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                  <div className={`max-w-[75%] px-4 py-3 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap ${
                    msg.role === "user"
                      ? "bg-blue-600 text-white rounded-br-sm"
                      : "bg-white border border-gray-200 text-gray-800 rounded-bl-sm shadow-sm"
                  } ${msg.streaming ? "opacity-90" : ""}`}>
                    {msg.content}
                    {msg.streaming && (
                      <span className="inline-block w-1 h-4 ml-0.5 bg-current opacity-70 animate-pulse" />
                    )}
                  </div>
                </div>
              );
            })}
            <div ref={bottomRef} />
          </div>
        </main>

        {/* Input */}
        <footer className="bg-white border-t border-gray-200 px-4 py-4 flex-shrink-0">
          <div className="max-w-3xl mx-auto flex gap-3 items-end">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                isAcceptingInput
                  ? "Type a message… (Enter to send, /sleep to end session)"
                  : ["SLEEPING","TRAINING","EVALUATING","DEPLOYING"].includes(session?.state ?? "")
                  ? "Training in progress — check the panel →"
                  : session?.state === "READY"
                  ? "New adapter is live — start a new session!"
                  : "Session closed"
              }
              disabled={!isAcceptingInput}
              rows={1}
              className="flex-1 resize-none rounded-xl border border-gray-300 px-4 py-3 text-sm
                         focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                         disabled:bg-gray-50 disabled:text-gray-400 disabled:cursor-not-allowed
                         max-h-40 overflow-y-auto"
              style={{ minHeight: "44px" }}
              onInput={(e) => {
                const t = e.target as HTMLTextAreaElement;
                t.style.height = "auto";
                t.style.height = Math.min(t.scrollHeight, 160) + "px";
              }}
            />
            <button onClick={sendMessage} disabled={!isAcceptingInput || !input.trim()}
              className="px-5 py-3 rounded-xl bg-blue-600 text-white text-sm font-medium
                         hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors shrink-0">
              {loading ? "…" : "Send"}
            </button>
          </div>
          {session?.state === "PRE_SLEEP_WARNING" && (
            <p className="text-center text-xs text-yellow-600 mt-2">
              ⚠ Approaching token limit — session will close after your next reply.
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
          outputFiles={outputFiles}
          lastPoll={lastPoll}
        />
      )}
    </div>
  );
}
