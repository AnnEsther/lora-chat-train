"use client";

import { useState } from "react";

// ── Types ──────────────────────────────────────────────────────────────────────

interface Step {
  number: number;
  title: string;
  description: string;
  note?: string;
  code?: string;
}

// ── Data ───────────────────────────────────────────────────────────────────────

const STEPS: Step[] = [
  {
    number: 1,
    title: "End your chat session",
    description:
      "When you're done chatting, type /sleep to trigger training immediately.",
    code: "/sleep",
    note: "If you don't type /sleep, training will start automatically once the token budget runs out.",
  },
  {
    number: 2,
    title: "Wait for training data",
    description:
      "The system processes your conversation and generates Q&A training data. Once ready, you'll get a notification in the",
    note: "lora-chat-train channel on Mattermost.",
  },
  {
    number: 3,
    title: "Review training data",
    description:
      'Click "Review Training Data" to see all Q&A pairs from your session. Edit or delete any entries as needed before approving.',
  },
  {
    number: 4,
    title: "Validate & start training",
    description:
      'Once everything looks good, click "Validate All & Start Training". Progress notifications will appear in the Mattermost channel.',
  },
  {
    number: 5,
    title: "Test the new adapter",
    description:
      "When training finishes, the new adapter will appear in the session selector. Start a new session with it and ask a few questions to verify.",
  },
  {
    number: 6,
    title: "Repeat",
    description:
      "Start a new session to begin the next training cycle. Each session builds on the last.",
  },
];

// ── Component ──────────────────────────────────────────────────────────────────

export function HelpPanel() {
  const [open, setOpen] = useState(false);

  return (
    <>
      {/* ── Backdrop ── */}
      {open && (
        <div
          className="fixed inset-0 z-40 bg-black/20 backdrop-blur-sm"
          onClick={() => setOpen(false)}
        />
      )}

      {/* ── Floating panel ── */}
      {open && (
        <div className="fixed bottom-20 right-6 z-50 w-80 rounded-2xl bg-white border border-gray-200 shadow-2xl overflow-hidden flex flex-col max-h-[70vh]">
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100 flex-shrink-0">
            <div>
              <p className="text-sm font-semibold text-gray-800">How it works</p>
              <p className="text-xs text-gray-400">Training workflow</p>
            </div>
            <button
              onClick={() => setOpen(false)}
              className="w-7 h-7 rounded-full bg-gray-100 hover:bg-gray-200 flex items-center justify-center text-gray-500 transition-colors"
              aria-label="Close help panel"
            >
              ✕
            </button>
          </div>

          {/* Steps */}
          <div className="overflow-y-auto px-4 py-3 space-y-4 flex-1">
            {STEPS.map((step, i) => (
              <div key={step.number} className="flex gap-3">
                {/* Step number + connector */}
                <div className="flex flex-col items-center">
                  <div className="w-6 h-6 rounded-full bg-blue-600 text-white text-xs font-semibold flex items-center justify-center flex-shrink-0">
                    {step.number}
                  </div>
                  {i < STEPS.length - 1 && (
                    <div className="w-px flex-1 bg-gray-200 mt-1" />
                  )}
                </div>

                {/* Content */}
                <div className="pb-4 flex-1 min-w-0">
                  <p className="text-xs font-semibold text-gray-700 mb-1">
                    {step.title}
                  </p>
                  <p className="text-xs text-gray-500 leading-relaxed">
                    {step.description}
                  </p>

                  {step.code && (
                    <code className="inline-block mt-1.5 px-2 py-0.5 bg-gray-100 text-gray-700 text-xs rounded font-mono">
                      {step.code}
                    </code>
                  )}

                  {step.note && (
                    <p className="mt-1.5 text-xs text-amber-600 bg-amber-50 rounded-lg px-2.5 py-1.5 leading-relaxed">
                      {step.note}
                    </p>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* Footer */}
          <div className="px-4 py-2.5 border-t border-gray-100 bg-gray-50 flex-shrink-0">
            <p className="text-xs text-gray-400 text-center">
              Chat → <code className="font-mono">/sleep</code> → review → validate → train → test → repeat
            </p>
          </div>
        </div>
      )}

      {/* ── Floating trigger button ── */}
      <button
        onClick={() => setOpen((v) => !v)}
        className={`fixed bottom-6 right-6 z-50 w-11 h-11 rounded-full shadow-lg flex items-center justify-center text-base font-semibold transition-all duration-200
          ${open
            ? "bg-gray-800 text-white scale-95"
            : "bg-blue-600 hover:bg-blue-700 text-white hover:scale-105"
          }`}
        aria-label="Toggle help panel"
        title="How it works"
      >
        {open ? "✕" : "?"}
      </button>
    </>
  );
}