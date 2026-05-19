"use client";

import { useState, useEffect, useCallback, useRef } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ── Types (mirrored from page.tsx) ────────────────────────────────────────────

export interface QAPair {
  id: string;
  question: string;
  answer: string;
  validated: boolean;
  edited: boolean;
}

export interface DeckMessage {
  id?: string;
  content: string;
  synthLoading?: boolean;
  segmentCount?: number;
  qaPairs?: QAPair[];
}

interface QADeckProps {
  messages: DeckMessage[];
  sessionId: string;
  initialPassageIdx: number;
  initialCardIdx: number;
  onClose: () => void;
  onUpdate: (turnId: string | undefined, qaId: string, updates: Partial<QAPair>) => void;
  onDelete: (turnId: string | undefined, qaId: string) => void;
}

// ── Single card in the deck ───────────────────────────────────────────────────

function DeckCard({
  pair,
  sessionId,
  turnId,
  onUpdate,
  onDelete,
  onNavigate,
}: {
  pair: QAPair;
  sessionId: string;
  turnId: string | undefined;
  onUpdate: (qaId: string, updates: Partial<QAPair>) => void;
  onDelete: (qaId: string) => void;
  /** Called when user presses arrow key while editing — direction: -1 | 1 */
  onNavigate: (direction: -1 | 1) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [question, setQuestion] = useState(pair.question);
  const [answer, setAnswer] = useState(pair.answer);
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);

  // Sync if parent updates (e.g. session reload)
  useEffect(() => { setQuestion(pair.question); }, [pair.question]);
  useEffect(() => { setAnswer(pair.answer); }, [pair.answer]);

  // Reset edit state when the card changes (pair.id changes)
  useEffect(() => {
    setEditing(false);
    setConfirmDelete(false);
    setSaving(false);
    setDeleting(false);
  }, [pair.id]);

  const handleSave = useCallback(async () => {
    setSaving(true);
    try {
      const resp = await fetch(`${API_URL}/sessions/${sessionId}/qa/${pair.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, answer }),
      });
      if (resp.ok) {
        onUpdate(pair.id, { question, answer, edited: true });
        setEditing(false);
      }
    } catch {}
    setSaving(false);
  }, [sessionId, pair.id, question, answer, onUpdate]);

  const handleCancel = () => {
    setQuestion(pair.question);
    setAnswer(pair.answer);
    setEditing(false);
  };

  const handleValidateToggle = async () => {
    try {
      const resp = await fetch(`${API_URL}/sessions/${sessionId}/qa/${pair.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ validated: !pair.validated }),
      });
      if (resp.ok) onUpdate(pair.id, { validated: !pair.validated });
    } catch {}
  };

  const handleDelete = async () => {
    setDeleting(true);
    try {
      await fetch(`${API_URL}/sessions/${sessionId}/qa/${pair.id}`, { method: "DELETE" });
      onDelete(pair.id);
    } catch {}
    setDeleting(false);
    setConfirmDelete(false);
  };

  // Arrow key navigation while editing — auto-save then navigate
  const handleEditKeyDown = useCallback(async (e: React.KeyboardEvent) => {
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      e.preventDefault();
      e.stopPropagation();
      await handleSave();
      onNavigate(e.key === "ArrowRight" ? 1 : -1);
    }
  }, [handleSave, onNavigate]);

  return (
    <div className="flex flex-col h-full">
      {/* Q/A content area — scrollable if content is tall */}
      <div className="flex-1 overflow-y-auto space-y-4 px-1">

        {/* Question */}
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <span className="w-6 h-6 rounded-full bg-gray-200 text-gray-500 text-xs font-bold flex items-center justify-center select-none flex-shrink-0">Q</span>
            <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Question</span>
          </div>
          {editing ? (
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleEditKeyDown}
              rows={3}
              className="w-full rounded-xl border-2 border-blue-300 focus:border-blue-500 bg-gray-50 px-4 py-3 text-sm leading-relaxed outline-none resize-none transition-colors"
            />
          ) : (
            <div
              className={`rounded-xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap shadow-sm ${
                pair.validated
                  ? "bg-gray-50 border-l-4 border-green-400 border border-gray-100"
                  : "bg-gray-50 border border-gray-200"
              }`}
            >
              {pair.question}
            </div>
          )}
        </div>

        {/* Answer */}
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <span className="w-6 h-6 rounded-full bg-gray-300 text-gray-600 text-xs font-bold flex items-center justify-center select-none flex-shrink-0">A</span>
            <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Answer</span>
          </div>
          {editing ? (
            <textarea
              value={answer}
              onChange={(e) => setAnswer(e.target.value)}
              onKeyDown={handleEditKeyDown}
              rows={6}
              className="w-full rounded-xl border-2 border-blue-300 focus:border-blue-500 bg-white px-4 py-3 text-sm leading-relaxed outline-none resize-none transition-colors"
            />
          ) : (
            <div
              className={`rounded-xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap shadow-sm ${
                pair.validated
                  ? "bg-white border-l-4 border-green-400 border border-gray-100"
                  : "bg-white border border-gray-200"
              }`}
            >
              {pair.answer}
            </div>
          )}
        </div>
      </div>

      {/* Action row — pinned to bottom of card */}
      <div className="pt-4 border-t border-gray-100 flex items-center gap-3 flex-wrap">
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
              onClick={handleCancel}
              className="text-sm px-4 py-2 rounded-lg border border-gray-200 hover:bg-gray-100 text-gray-500 transition-colors"
            >
              Cancel
            </button>
            <span className="text-xs text-gray-400 ml-auto">Arrow keys save &amp; navigate</span>
          </>
        ) : (
          <>
            {/* Validate toggle */}
            {pair.validated ? (
              <button
                onClick={handleValidateToggle}
                className="text-sm px-4 py-2 rounded-lg bg-green-100 text-green-700 hover:bg-green-200 font-medium transition-colors flex items-center gap-1.5"
              >
                <span>✓</span> Validated
              </button>
            ) : (
              <button
                onClick={handleValidateToggle}
                className="text-sm px-4 py-2 rounded-lg border border-green-200 bg-green-50 text-green-700 hover:bg-green-100 font-medium transition-colors"
              >
                Mark validated
              </button>
            )}

            {/* Edit */}
            <button
              onClick={() => setEditing(true)}
              className="text-sm px-4 py-2 rounded-lg border border-gray-200 hover:bg-gray-100 text-gray-600 transition-colors"
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
                  className="text-sm px-3 py-2 text-gray-400 hover:text-gray-600 transition-colors"
                >
                  Cancel
                </button>
              </>
            ) : (
              <button
                onClick={() => setConfirmDelete(true)}
                className="text-sm px-3 py-2 text-gray-300 hover:text-red-400 transition-colors ml-auto"
                title="Delete this pair"
              >
                ✕
              </button>
            )}
          </>
        )}
      </div>
    </div>
  );
}

// ── Skeleton card ─────────────────────────────────────────────────────────────

function SkeletonCard() {
  return (
    <div className="flex flex-col h-full space-y-4 px-1">
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded-full bg-gray-200 animate-pulse" />
          <div className="h-3 w-16 rounded bg-gray-200 animate-pulse" />
        </div>
        <div className="rounded-xl bg-gray-100 animate-pulse px-4 py-3 space-y-2">
          <div className="h-3 w-3/5 rounded bg-gray-200" />
          <div className="h-3 w-2/5 rounded bg-gray-200" />
        </div>
      </div>
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded-full bg-gray-200 animate-pulse" />
          <div className="h-3 w-12 rounded bg-gray-200 animate-pulse" />
        </div>
        <div className="rounded-xl bg-gray-50 animate-pulse px-4 py-3 space-y-2">
          <div className="h-3 w-full rounded bg-gray-200" />
          <div className="h-3 w-full rounded bg-gray-200" />
          <div className="h-3 w-3/5 rounded bg-gray-200" />
        </div>
      </div>
    </div>
  );
}

// ── QADeck overlay ────────────────────────────────────────────────────────────

export default function QADeck({
  messages,
  sessionId,
  initialPassageIdx,
  initialCardIdx,
  onClose,
  onUpdate,
  onDelete,
}: QADeckProps) {
  // Only messages that have QA pairs (or are loading)
  const passages = messages.filter(
    (m) => m.qaPairs !== undefined || m.synthLoading
  );

  const [passageIdx, setPassageIdx] = useState(() =>
    Math.max(0, Math.min(initialPassageIdx, passages.length - 1))
  );
  const [cardIdx, setCardIdx] = useState(initialCardIdx);

  // Slide animation state
  // direction: "left" = new card comes from right, "right" = new card comes from left
  const [slideDir, setSlideDir] = useState<"left" | "right" | null>(null);
  const [animating, setAnimating] = useState(false);
  // pendingNav holds the target indices while animation plays
  const pendingNav = useRef<{ p: number; c: number } | null>(null);

  const currentPassage = passages[passageIdx] ?? null;
  const pairs = currentPassage?.qaPairs ?? [];
  const totalCards = currentPassage?.synthLoading
    ? Math.max(pairs.length, currentPassage.segmentCount ?? 1)
    : pairs.length;

  // Clamp cardIdx when passage or pairs change
  useEffect(() => {
    if (pairs.length > 0 && cardIdx >= pairs.length) {
      setCardIdx(pairs.length - 1);
    }
  }, [pairs.length, cardIdx]);

  // Navigate to a specific passage+card, triggering slide animation
  const navigateTo = useCallback(
    (targetPassage: number, targetCard: number, dir: "left" | "right") => {
      if (animating) return;
      pendingNav.current = { p: targetPassage, c: targetCard };
      setSlideDir(dir);
      setAnimating(true);
      setTimeout(() => {
        if (pendingNav.current) {
          setPassageIdx(pendingNav.current.p);
          setCardIdx(pendingNav.current.c);
          pendingNav.current = null;
        }
        setSlideDir(null);
        setAnimating(false);
      }, 280);
    },
    [animating]
  );

  const goNext = useCallback(() => {
    if (animating) return;
    const nextCard = cardIdx + 1;
    if (nextCard < pairs.length) {
      navigateTo(passageIdx, nextCard, "left");
    } else if (passageIdx + 1 < passages.length) {
      navigateTo(passageIdx + 1, 0, "left");
    }
  }, [animating, cardIdx, pairs.length, passageIdx, passages.length, navigateTo]);

  const goPrev = useCallback(() => {
    if (animating) return;
    const prevCard = cardIdx - 1;
    if (prevCard >= 0) {
      navigateTo(passageIdx, prevCard, "right");
    } else if (passageIdx - 1 >= 0) {
      const prevPairs = passages[passageIdx - 1]?.qaPairs ?? [];
      navigateTo(passageIdx - 1, Math.max(0, prevPairs.length - 1), "right");
    }
  }, [animating, cardIdx, passageIdx, passages, navigateTo]);

  // Keyboard handler — global within the deck overlay
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Escape") { onClose(); return; }
      // Arrow keys only if not editing (textarea has focus)
      if (document.activeElement?.tagName === "TEXTAREA") return;
      if (e.key === "ArrowRight") { e.preventDefault(); goNext(); }
      if (e.key === "ArrowLeft")  { e.preventDefault(); goPrev(); }
    },
    [onClose, goNext, goPrev]
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  // Called by DeckCard when user presses arrow while editing
  const handleEditNavigate = useCallback(
    (direction: -1 | 1) => {
      if (direction === 1) goNext();
      else goPrev();
    },
    [goNext, goPrev]
  );

  // Flat card counter across all passages
  const flatTotal = passages.reduce((sum, p) => sum + Math.max(p.qaPairs?.length ?? 0, p.synthLoading ? (p.segmentCount ?? 1) : 0), 0);
  const flatCurrent = passages.slice(0, passageIdx).reduce((sum, p) => sum + Math.max(p.qaPairs?.length ?? 0, p.synthLoading ? (p.segmentCount ?? 1) : 0), 0) + cardIdx + 1;

  const hasNext = cardIdx + 1 < pairs.length || passageIdx + 1 < passages.length;
  const hasPrev = cardIdx > 0 || passageIdx > 0;

  // Slide animation classes
  // Before animation: card slides out; entering card slides in from opposite side
  const exitClass = slideDir === "left"
    ? "translate-x-[-110%] opacity-0"
    : slideDir === "right"
    ? "translate-x-[110%] opacity-0"
    : "";
  const enterClass = slideDir === "left"
    ? "translate-x-[110%] opacity-0"
    : slideDir === "right"
    ? "translate-x-[-110%] opacity-0"
    : "";

  if (passages.length === 0) {
    return (
      <div className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl shadow-2xl w-full max-w-2xl p-8 text-center">
          <p className="text-gray-400 text-sm">No Q&A pairs yet.</p>
          <button onClick={onClose} className="mt-4 text-sm px-4 py-2 rounded-lg bg-gray-100 hover:bg-gray-200 text-gray-600 transition-colors">Close</button>
        </div>
      </div>
    );
  }

  const currentPair = pairs[cardIdx] ?? null;
  const isSkeletonCard = !currentPair && currentPassage?.synthLoading;

  return (
    <div
      className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-2xl flex flex-col"
           style={{ maxHeight: "calc(100vh - 2rem)", minHeight: "520px" }}>

        {/* ── Deck header ── */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100 flex-shrink-0">
          <div className="flex items-center gap-3 min-w-0">
            {/* Passage tabs */}
            <div className="flex items-center gap-1 overflow-x-auto">
              {passages.map((p, pi) => {
                const pairCount = p.qaPairs?.length ?? 0;
                const validCount = p.qaPairs?.filter(qa => qa.validated).length ?? 0;
                return (
                  <button
                    key={pi}
                    onClick={() => navigateTo(pi, 0, pi > passageIdx ? "left" : "right")}
                    className={`flex-shrink-0 flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-lg font-medium transition-colors ${
                      pi === passageIdx
                        ? "bg-blue-600 text-white"
                        : "bg-gray-100 text-gray-500 hover:bg-gray-200"
                    }`}
                  >
                    <span>Passage {pi + 1}</span>
                    {p.synthLoading && (
                      <span className="relative flex h-1.5 w-1.5">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-300 opacity-75" />
                        <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-blue-400" />
                      </span>
                    )}
                    {!p.synthLoading && pairCount > 0 && (
                      <span className={`${pi === passageIdx ? "bg-blue-500 text-white" : "bg-gray-200 text-gray-600"} rounded-full text-xs px-1.5 py-0.5 leading-none`}>
                        {validCount}/{pairCount}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Card counter + close */}
          <div className="flex items-center gap-3 flex-shrink-0 ml-3">
            <span className="text-xs text-gray-400 font-medium tabular-nums">
              {flatTotal > 0 ? `${flatCurrent} / ${flatTotal}` : "—"}
            </span>
            <button
              onClick={onClose}
              className="w-7 h-7 flex items-center justify-center rounded-full bg-gray-100 hover:bg-gray-200 text-gray-500 text-sm transition-colors"
              title="Close deck (Esc)"
            >
              ✕
            </button>
          </div>
        </div>

        {/* ── Card area with slide animation ── */}
        <div className="flex-1 overflow-hidden relative px-6 py-6">
          <div
            className={`absolute inset-0 px-6 py-6 flex flex-col transition-all duration-[280ms] ease-in-out ${
              animating ? exitClass : "translate-x-0 opacity-100"
            }`}
          >
            {isSkeletonCard ? (
              <SkeletonCard />
            ) : currentPair ? (
              <DeckCard
                key={currentPair.id}
                pair={currentPair}
                sessionId={sessionId}
                turnId={currentPassage?.id}
                onUpdate={(qaId, updates) => {
                  onUpdate(currentPassage?.id, qaId, updates);
                }}
                onDelete={(qaId) => {
                  onDelete(currentPassage?.id, qaId);
                  // If we deleted the last card in this passage, move back
                  if (cardIdx >= (pairs.length - 1) && cardIdx > 0) {
                    setCardIdx(cardIdx - 1);
                  }
                }}
                onNavigate={handleEditNavigate}
              />
            ) : (
              <div className="flex-1 flex items-center justify-center text-gray-400 text-sm">
                No pairs for this passage yet.
              </div>
            )}
          </div>

          {/* Entering card (slides in from opposite side) */}
          {animating && (
            <div
              className={`absolute inset-0 px-6 py-6 flex flex-col transition-none ${enterClass}`}
              style={{ animation: `slideIn 280ms ease-in-out forwards` }}
            />
          )}
        </div>

        {/* ── Navigation footer ── */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-gray-100 flex-shrink-0">
          {/* Prev button */}
          <button
            onClick={goPrev}
            disabled={!hasPrev || animating}
            className="flex items-center gap-2 text-sm px-4 py-2 rounded-lg border border-gray-200 hover:bg-gray-50 text-gray-600 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            ← Prev
          </button>

          {/* Dot indicators for cards within current passage */}
          <div className="flex items-center gap-1.5">
            {Array.from({ length: totalCards }).map((_, di) => (
              <button
                key={di}
                onClick={() => navigateTo(passageIdx, di, di > cardIdx ? "left" : "right")}
                disabled={di >= pairs.length && !currentPassage?.synthLoading}
                className={`w-2 h-2 rounded-full transition-colors ${
                  di === cardIdx
                    ? "bg-blue-600"
                    : di < pairs.length
                    ? "bg-gray-300 hover:bg-gray-400"
                    : "bg-gray-200 animate-pulse"
                }`}
                title={`Card ${di + 1}`}
              />
            ))}
          </div>

          {/* Next button */}
          <button
            onClick={goNext}
            disabled={!hasNext || animating}
            className="flex items-center gap-2 text-sm px-4 py-2 rounded-lg border border-gray-200 hover:bg-gray-50 text-gray-600 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            Next →
          </button>
        </div>
      </div>
    </div>
  );
}
