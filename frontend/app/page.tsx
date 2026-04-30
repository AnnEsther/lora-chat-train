"use client";

import { useState, useEffect, useRef, useCallback } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const MODEL_SERVER_URL = process.env.NEXT_PUBLIC_MODEL_SERVER_URL ?? "http://localhost:8001";
const POLL_INTERVAL_MS = 5000;

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
  system_prompt?: string | null;
  training_system_prompt?: string | null;
  failure_reason?: string | null;
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

interface Adapter {
  id: string;
  version: string;
  path: string;
  trained_at: string | null;
  is_current?: boolean;
  is_base?: boolean;
}

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
  VALIDATING:        "⚠ Review training data",
  SLEEPING:          "Sleeping — training queued",
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
      <div
        className={`h-full rounded-full transition-all duration-500 ${color}`}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

function StatRow({ label, value, sub, valueColor }: { label: string; value: string; sub?: string; valueColor?: string }) {
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-gray-100 last:border-0">
      <span className="text-xs text-gray-500">{label}</span>
      <div className="text-right">
        <span className={`text-xs font-medium ${valueColor || "text-gray-800"}`}>{value}</span>
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

  const stateToActiveStep: Record<SessionState, number> = {
    VALIDATING: 0,
    SLEEPING: 3,
    TRAINING: 3,
    EVALUATING: 4,
    DEPLOYING: 5,
    READY: 5,
    FAILED: -1,
    ACTIVE: -1,
    PRE_SLEEP_WARNING: -1,
    INSUFFICIENT_DATA: -1,
  };

  const isTrainingComplete = trainStatus?.status === "completed";

  if (isTrainingComplete) {
    return steps.map((label, i) => ({ label, state: "done" as const }));
  }

  const activeStep = stateToActiveStep[sessionState] ?? -1;

  return steps.map((label, i) => {
    if (sessionState === "FAILED") {
      return { label, state: i < activeStep ? "done" : i === activeStep ? "failed" : "pending" } as const;
    }
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
  session,
  health,
  trainStatus,
  lastPoll,
  selectedAdapter,
  adapters,
  onRestartTraining,
}: {
  session: Session | null;
  health: ModelHealth | null;
  trainStatus: TrainStatus | null;
  lastPoll: Date | null;
  selectedAdapter: string;
  adapters: Adapter[];
  onRestartTraining?: () => void;
}) {
  const gpu = health?.gpu ?? null;
  const vramPct = gpu ? gpu.vram_used_gb / gpu.vram_total_gb : 0;
  const tokenPct = session ? session.total_tokens / session.max_tokens : 0;

  const isTraining = trainStatus?.status === "running";
  const pipelineSteps = session ? getPipelineSteps(session.state, trainStatus ?? undefined) : [];
  
  const currentAdapter = adapters.find(a => a.id === selectedAdapter);
  const adapterVersion = currentAdapter?.version || "Base model";

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
            <StatRow
              label="Adapter"
              value={adapterVersion}
            />
            {session.system_prompt && (
              <div className="mt-2 pt-2 border-t border-gray-100">
                <p className="text-xs text-gray-500 mb-1">Chat system prompt</p>
                <p className="text-xs text-gray-700 bg-gray-50 rounded p-2 max-h-16 overflow-y-auto">
                  {session.system_prompt}
                </p>
              </div>
            )}
            {session.training_system_prompt && (
              <div className="mt-2 pt-2 border-t border-gray-100">
                <p className="text-xs text-gray-500 mb-1">Training system prompt</p>
                <p className="text-xs text-gray-700 bg-gray-50 rounded p-2 max-h-16 overflow-y-auto">
                  {session.training_system_prompt}
                </p>
              </div>
            )}
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
          <SectionHeader 
            title="Training" 
            dot={trainStatus.status === "failed" ? "bg-red-400" : isTraining ? "bg-blue-400 animate-pulse" : "bg-green-400"} 
          />
          <div className={`rounded-lg border px-3 py-2 ${trainStatus.status === "failed" ? "bg-red-50 border-red-200" : "bg-white border-gray-200"}`}>
            <StatRow
              label="Status"
              value={trainStatus.status.charAt(0).toUpperCase() + trainStatus.status.slice(1)}
              valueColor={trainStatus.status === "failed" ? "text-red-600" : undefined}
            />
            {trainStatus.run_id && (
              <StatRow label="Run ID" value={trainStatus.run_id.slice(0, 8) + "…"} />
            )}
            <div className="py-1.5">
              <p className="text-xs text-gray-500 mb-0.5">Progress</p>
              <p className={`text-xs leading-relaxed ${trainStatus.status === "failed" ? "text-red-700" : "text-gray-800"}`}>
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
            {trainStatus.status === "failed" && session && (
              <button
                onClick={async () => {
                  try {
                    const resp = await fetch(`${API_URL}/sessions/${session.id}/restart-training`, { method: "POST" });
                    if (resp.ok && onRestartTraining) {
                      onRestartTraining();
                    }
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
  const [sessions, setSessions]       = useState<Session[]>([]);
  const [session, setSession]         = useState<Session | null>(null);
  const [messages, setMessages]       = useState<Message[]>([]);
  const [input, setInput]             = useState("");
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState<string | null>(null);
  const [health, setHealth]           = useState<ModelHealth | null>(null);
  const [trainStatus, setTrainStatus] = useState<TrainStatus | null>(null);
  const [outputFiles, setOutputFiles] = useState<OutputFile[]>([]);
  const [adapters, setAdapters]       = useState<Adapter[]>([{ id: "base", version: "Base model", path: "", is_base: true, trained_at: null }]);
  const [selectedAdapter, setSelectedAdapter] = useState<string>("base");
  const [trainingSystemPrompt, setTrainingSystemPrompt] = useState<string>("");
  const [lastPoll, setLastPoll]       = useState<Date | null>(null);
  const [panelOpen, setPanelOpen]     = useState(true);
  const [pollActive, setPollActive]   = useState(false);
  const [systemPrompt, setSystemPrompt] = useState<string>("");
  const [qaReviewOpen, setQaReviewOpen] = useState(false);
  const [qaItems, setQaItems] = useState<{id: string; question: string; answer: string; validated: boolean; edited: boolean; retry_count: number; validation_notes: string}[]>([]);
  const [qaCurrentIndex, setQaCurrentIndex] = useState(0);
  const [qaLoading, setQaLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const prevSessionStateRef = useRef<SessionState | null>(null);

  // ── Load sessions list ──
  const fetchSessions = useCallback(async () => {
    try {
      const resp = await fetch(`${API_URL}/sessions?limit=20`);
      if (resp.ok) {
        const data = await resp.json();
        const sessionsList: Session[] = Array.isArray(data) ? data : (data.sessions ?? []);
        setSessions(sessionsList);
        return sessionsList;
      }
    } catch {}
    return [];
  }, []);

  // ── Restore last session from localStorage ──
  useEffect(() => {
    const restoreSession = async () => {
      const savedId = localStorage.getItem("lora_session_id");
      const allSessions = await fetchSessions();
      
      if (savedId && allSessions.length > 0) {
        const found = allSessions.find(s => s.id === savedId);
        if (found) {
          setSession(found);
          return;
        }
      }
      // Fall back to most recent non-READY session, or latest session
      const target = allSessions.find(s => !["READY", "FAILED"].includes(s.state)) ?? allSessions[0];
      if (target) {
        setSession(target);
      } else {
        // No sessions exist, create one
        await createSession();
      }
    };
    restoreSession();
  }, [fetchSessions]);

  // ── Save session to localStorage when changed ──
  useEffect(() => {
    if (session) {
      localStorage.setItem("lora_session_id", session.id);
    }
  }, [session]);

  // ── Keep prevSessionStateRef in sync (for transition detection in polling) ──
  useEffect(() => {
    if (session) {
      prevSessionStateRef.current = session.state;
    }
  }, [session?.id]);  // reset ref only when session changes, not on every state poll

  // ── Scroll to bottom ──
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ── QA Review functions ──
  const fetchQaItems = async () => {
    if (!session) return;
    setQaLoading(true);
    console.log("Fetching QA items for session:", session.id);
    try {
      const resp = await fetch(`${API_URL}/sessions/${session.id}/qa`);
      console.log("QA response:", resp.status, resp.ok);
      if (resp.ok) {
        const data = await resp.json();
        console.log("QA data:", data);
        setQaItems(data);
        setQaCurrentIndex(0);
      } else {
        const err = await resp.text();
        console.error("QA fetch error:", err);
      }
    } catch (e) {
      console.error("QA fetch catch:", e);
    }
    setQaLoading(false);
  };

  const updateQaItem = async (id: string, updates: {question?: string; answer?: string; validated?: boolean}) => {
    if (!session) return;
    try {
      await fetch(`${API_URL}/sessions/${session.id}/qa/${id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(updates),
      });
      await fetchQaItems();
    } catch {}
  };

  const markAllValidated = async () => {
    if (!session) return;
    try {
      await fetch(`${API_URL}/sessions/${session.id}/qa/validate-mark`, {
        method: "POST",
      });
      await fetchQaItems();
    } catch {}
  };

  const openQaReview = () => {
    setQaReviewOpen(true);
    fetchQaItems();
  };

  // Auto-open QA modal when session enters VALIDATING state
  useEffect(() => {
    if (session?.state === "VALIDATING" && !qaReviewOpen) {
      openQaReview();
    }
  }, [session?.state]);

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

  // ── Poll session state while not ACTIVE ──
  useEffect(() => {
    if (!session) return;
    if (["ACTIVE", "PRE_SLEEP_WARNING", "INSUFFICIENT_DATA", "READY", "FAILED"].includes(session.state)) {
      setPollActive(false);
      return;
    }

    setPollActive(true);
    const id = setInterval(async () => {
      try {
        const resp = await fetch(`${API_URL}/sessions/${session.id}`);
        if (resp.ok) {
          const data: Session = await resp.json();
          const prev = prevSessionStateRef.current;

          // Inject a system message when session first enters FAILED
          if (data.state === "FAILED" && prev !== "FAILED") {
            const reason = data.failure_reason ?? "An unknown error occurred.";
            setMessages((msgs) => [
              ...msgs,
              {
                role: "system",
                content: `Training failed: ${reason} You can keep chatting or type /sleep to retry fine-tuning.`,
              },
            ]);
          }

          // Inject a system message when session first enters INSUFFICIENT_DATA
          if (data.state === "INSUFFICIENT_DATA" && prev !== "INSUFFICIENT_DATA") {
            setMessages((msgs) => [
              ...msgs,
              {
                role: "system",
                content:
                  "Not enough usable training data was found (at least 10 good examples are needed). Keep chatting to add more — type /sleep again when you're ready to retry fine-tuning.",
              },
            ]);
          }

          prevSessionStateRef.current = data.state;
          setSession(data);
          if (["READY", "FAILED", "ACTIVE"].includes(data.state)) {
            setPollActive(false);
          }
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

  const fetchAdapters = async () => {
    const defaultAdapter = { id: "base", version: "Base model", path: "", is_base: true, trained_at: null };
    console.log("Fetching adapters from", MODEL_SERVER_URL);
    try {
      const resp = await fetch(`${MODEL_SERVER_URL}/adapters`, { signal: AbortSignal.timeout(4000) });
      console.log("Model server response:", resp.status, resp.ok);
      if (resp.ok) {
        const data = await resp.json();
        console.log("Model server data:", data);
        const fetchedAdapters = data.adapters || [];
        // Filter out base, keep everything else
        const filtered = fetchedAdapters.filter((a: Adapter) => a.id !== "base");
        console.log("Filtered adapters:", filtered);
        if (filtered.length > 0) {
          setAdapters([defaultAdapter, ...filtered]);
          return;
        }
      }
    } catch (e) { console.log("Model server adapters error:", e); }
    console.log("Fetching adapters from", API_URL);
    try {
      const resp = await fetch(`${API_URL}/adapters`, { signal: AbortSignal.timeout(4000) });
      console.log("Backend response:", resp.status, resp.ok);
      if (resp.ok) {
        const data = await resp.json();
        console.log("Backend data:", data);
        const fetchedAdapters = data.adapters || [];
        // Filter out base
        const filtered = fetchedAdapters.filter((a: Adapter) => a.id !== "base");
        console.log("Filtered adapters:", filtered);
        if (filtered.length > 0) {
          setAdapters([defaultAdapter, ...filtered]);
          return;
        }
      }
    } catch (e) { console.log("Backend adapters error:", e); }
    console.log("Using default adapter only");
    setAdapters([defaultAdapter]);
  };

  const createSession = async (adapterId?: string) => {
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
    if (adapterId) {
      setSelectedAdapter(adapterId);
    }
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
      setError(null);
      await fetchSessions();
    } catch {
      setError("Could not create session. Is the backend running?");
    }
  };

  const sendMessage = useCallback(async () => {
    if (!input.trim() || !session || loading) return;
    if (!["ACTIVE", "PRE_SLEEP_WARNING", "INSUFFICIENT_DATA", "FAILED"].includes(session.state)) return;

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
    ["ACTIVE", "PRE_SLEEP_WARNING", "INSUFFICIENT_DATA", "FAILED"].includes(session.state) && !loading;
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
            <div className="flex items-center gap-1">
              <select
                key={sessions.length}
                value={session?.id ?? ""}
                onChange={async (e) => {
                  const s = sessions.find(s => s.id === e.target.value);
                  if (s) setSession(s);
                }}
                className="text-xs px-2 py-1.5 rounded-md border border-gray-300 bg-white text-gray-700"
              >
                {!session && <option value="">No session</option>}
                {(sessions || []).map((s) => (
                  <option key={s.id} value={s.id}>
                    {STATE_LABELS[s.state].replace(/[^\w]/g, "")} {s.id.slice(0,8)}
                  </option>
                ))}
              </select>
              <button
                onClick={fetchSessions}
                className="text-xs px-2 py-1.5 rounded-md bg-gray-100 hover:bg-gray-200 text-gray-600"
                title="Refresh sessions"
              >
                ↻
              </button>
            </div>
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
            <div className="relative">
              <button 
                onClick={() => {
                  const dd = document.getElementById('new-session-dd');
                  dd?.classList.toggle('hidden');
                }}
                className="text-xs px-3 py-1.5 rounded-md bg-blue-600 hover:bg-blue-700 text-white transition-colors"
              >
                New session ▾
              </button>
              <div id="new-session-dd" className="hidden absolute right-0 top-full mt-1 w-72 bg-white border border-gray-200 rounded-lg shadow-lg z-50 p-3 space-y-2">
                <div className="text-xs text-gray-500 font-medium">Chat system prompt (optional)</div>
                <textarea
                  value={systemPrompt}
                  onChange={(e) => setSystemPrompt(e.target.value)}
                  placeholder="You are a helpful AI assistant..."
                  className="w-full text-xs px-2 py-1.5 rounded border border-gray-200 text-gray-700 resize-none"
                  rows={2}
                />
                <div className="text-xs text-gray-500 font-medium border-t border-gray-100 pt-2">Training system prompt (optional)</div>
                <textarea
                  value={trainingSystemPrompt}
                  onChange={(e) => setTrainingSystemPrompt(e.target.value)}
                  placeholder="Prompt used when training the model..."
                  className="w-full text-xs px-2 py-1.5 rounded border border-gray-200 text-gray-700 resize-none"
                  rows={2}
                />
                <div className="text-xs text-gray-500 font-medium border-t border-gray-100 pt-2">Select adapter</div>
                <button
                  onClick={() => { createSession("base"); document.getElementById('new-session-dd')?.classList.add('hidden'); }}
                  className="w-full text-left px-2 py-1.5 text-xs hover:bg-gray-50 text-gray-700 rounded"
                >
                  Base model
                </button>
                {adapters.filter(a => a.id !== "base").map((a) => (
                  <button
                    key={a.id}
                    onClick={() => { createSession(a.id); document.getElementById('new-session-dd')?.classList.add('hidden'); }}
                    className="w-full text-left px-2 py-1.5 text-xs hover:bg-gray-50 text-gray-700 rounded"
                  >
                    {a.version}{a.is_current ? " (live)" : ""}
                  </button>
                ))}
              </div>
            </div>
            {session && ["ACTIVE", "PRE_SLEEP_WARNING", "INSUFFICIENT_DATA", "FAILED"].includes(session.state) && (
              <button
                onClick={openQaReview}
                className="text-xs px-3 py-1.5 rounded-md bg-purple-600 hover:bg-purple-700 text-white transition-colors"
                title="Review and validate training data"
              >
                Review Training Data
              </button>
            )}
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
                session?.state === "INSUFFICIENT_DATA"
                  ? "Type more messages to add training data… (/sleep when ready)"
                  : session?.state === "FAILED"
                  ? "Training failed — keep chatting or type /sleep to retry fine-tuning"
                  : isAcceptingInput
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
          {session?.state === "INSUFFICIENT_DATA" && (
            <p className="text-center text-xs text-orange-600 mt-2">
              ⚠ Not enough training data — keep chatting to add more, then type{" "}
              <code className="bg-orange-100 px-1 rounded">/sleep</code>{" "}
              to trigger fine-tuning.
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
          onRestartTraining={fetchTrainStatus}
        />
      )}

      {/* ── QA Review Modal ── */}
      {qaReviewOpen && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl max-h-[80vh] flex flex-col">
            <div className="flex items-center justify-between px-4 py-3 border-b">
              <h2 className="font-semibold text-gray-700">Review Training Data</h2>
              <button onClick={() => setQaReviewOpen(false)} className="text-gray-500 hover:text-gray-700">✕</button>
            </div>
            <div className="flex-1 overflow-y-auto p-4">
              {qaLoading ? (
                <p className="text-center text-gray-500 py-8">Loading...</p>
              ) : qaItems.length === 0 ? (
                <p className="text-center text-gray-500 py-8">No training data to review yet.</p>
              ) : (
                <div className="space-y-4">
                  <div className="flex items-center justify-between text-sm text-gray-500">
                    <span>Item {qaCurrentIndex + 1} of {qaItems.length}</span>
                    <span>{qaItems[qaCurrentIndex]?.validated ? "✓ Validated" : qaItems[qaCurrentIndex]?.retry_count >= 3 ? "⚠ Needs review" : "Pending"}</span>
                  </div>
                  <div>
                    <label className="block text-xs text-gray-600 mb-1">Question</label>
                    <textarea
                      value={qaItems[qaCurrentIndex]?.question || ""}
                      onChange={(e) => {
                        const newItems = [...qaItems];
                        newItems[qaCurrentIndex].question = e.target.value;
                        newItems[qaCurrentIndex].edited = true;
                        setQaItems(newItems);
                      }}
                      className="w-full text-sm px-3 py-2 rounded border border-gray-300 resize-none"
                      rows={3}
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-600 mb-1">Answer</label>
                    <textarea
                      value={qaItems[qaCurrentIndex]?.answer || ""}
                      onChange={(e) => {
                        const newItems = [...qaItems];
                        newItems[qaCurrentIndex].answer = e.target.value;
                        newItems[qaCurrentIndex].edited = true;
                        setQaItems(newItems);
                      }}
                      className="w-full text-sm px-3 py-2 rounded border border-gray-300 resize-none"
                      rows={6}
                    />
                  </div>
                  {qaItems[qaCurrentIndex]?.validation_notes && (
                    <p className="text-xs text-gray-500 bg-gray-50 p-2 rounded">{qaItems[qaCurrentIndex].validation_notes}</p>
                  )}
                  <div className="flex items-center justify-between">
                    <button
                      onClick={() => setQaCurrentIndex(Math.max(0, qaCurrentIndex - 1))}
                      disabled={qaCurrentIndex === 0}
                      className="text-xs px-3 py-1.5 rounded bg-gray-100 hover:bg-gray-200 disabled:opacity-50"
                    >
                      Previous
                    </button>
                    <button
                      onClick={() => updateQaItem(qaItems[qaCurrentIndex].id, {validated: true})}
                      className="text-xs px-3 py-1.5 rounded bg-green-100 hover:bg-green-200 text-green-700"
                    >
                      Mark Validated
                    </button>
                    <button
                      onClick={() => setQaCurrentIndex(Math.min(qaItems.length - 1, qaCurrentIndex + 1))}
                      disabled={qaCurrentIndex === qaItems.length - 1}
                      className="text-xs px-3 py-1.5 rounded bg-gray-100 hover:bg-gray-200 disabled:opacity-50"
                    >
                      Next
                    </button>
                  </div>
                </div>
              )}
            </div>
            <div className="flex items-center justify-between px-4 py-3 border-t">
              <button
                onClick={() => {
                  if (qaItems[qaCurrentIndex]?.edited) {
                    updateQaItem(qaItems[qaCurrentIndex].id, {
                      question: qaItems[qaCurrentIndex].question,
                      answer: qaItems[qaCurrentIndex].answer,
                    });
                  }
                }}
                className="text-xs px-3 py-1.5 rounded bg-gray-100 hover:bg-gray-200"
              >
                Save Current
              </button>
              <button
                onClick={async () => {
                  await markAllValidated();
                  if (session) {
                    try {
                      const resp = await fetch(`${API_URL}/sessions/${session.id}/start-training`, { method: "POST" });
                      if (resp.ok) {
                        setQaReviewOpen(false);
                        await fetchSessions();
                        await fetchTrainStatus();
                      }
                    } catch {}
                  }
                }}
                className="text-xs px-4 py-1.5 rounded bg-blue-600 hover:bg-blue-700 text-white"
              >
                Validate All & Start Training
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
