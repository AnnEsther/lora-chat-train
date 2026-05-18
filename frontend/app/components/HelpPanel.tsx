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
    title: "Send a passage",
    description:
      "Paste any text you want the model to learn from and press Send. The system will automatically generate Q&A training pairs from it.",
    note: "You can send as many passages as you like — each one adds more training data.",
  },
  {
    number: 2,
    title: "Review Q&A pairs inline",
    description:
      "Generated Q&A cards appear directly below each passage in the chat. Edit the question or answer text, then click Save changes.",
  },
  {
    number: 3,
    title: "Validate pairs",
    description:
      'Click "Mark validated" on each Q&A card you are happy with. Validated pairs count toward the training threshold shown in the Start Training button.',
  },
  {
    number: 4,
    title: "Start Training",
    description:
      'Once you have enough validated pairs the "Start Training" button in the header turns green. Click it to begin fine-tuning.',
    note: "You can also type /sleep to immediately start training using all current Q&A pairs.",
    code: "/sleep",
  },
  {
    number: 5,
    title: "Test the new adapter",
    description:
      "When training finishes, the new adapter appears in the New Session dropdown. Start a new session with it and verify the results.",
  },
  {
    number: 6,
    title: "Repeat",
    description:
      "Start a new session and send more passages to keep improving the model. Each cycle builds on the last.",
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
              Send passages → review Q&A → validate → <code className="font-mono">Start Training</code> → test → repeat
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