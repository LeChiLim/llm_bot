"use client";

import React from "react";
import { Search, BookOpen, Scale, FileText, Gavel } from "lucide-react";
import { Button } from "@/components/ui/button";

interface GreetingProps {
  onSuggestion: (text: string) => void;
}

const SKILL_CHIPS = [
  { label: "Summarise case", icon: FileText },
  { label: "Find precedents", icon: Search },
  { label: "HK mode", icon: Scale },
  { label: "SG mode", icon: Gavel },
  { label: "Compare jurisdictions", icon: BookOpen },
];

const SUGGESTIONS = [
  {
    icon: "⚖️",
    label: "Penalty clauses",
    prompt: "What is the HK courts' position on penalty clauses following Cavendish Square v Makdessi?",
    tag: "HK · Contract",
  },
  {
    icon: "🏛️",
    label: "Duty of care",
    prompt: "Compare the duty of care test in Singapore (Spandeck) vs Hong Kong (Caparo approach)",
    tag: "HK + SG · Tort",
  },
  {
    icon: "🤝",
    label: "Director duties",
    prompt: "Leading cases on director fiduciary duties — how do HK and SG approaches differ?",
    tag: "HK + SG · Corporate",
  },
  {
    icon: "™️",
    label: "Trade mark passing off",
    prompt: "Key Singapore cases on passing off and trade mark distinctiveness requirements",
    tag: "SG · IP",
  },
];

// Simple SVG capybara lawyer mascot
function CapybaraIcon() {
  return (
    <svg
      width="80"
      height="80"
      viewBox="0 0 80 80"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className="opacity-80"
    >
      {/* Body */}
      <ellipse cx="40" cy="52" rx="22" ry="18" fill="#c4a882" opacity="0.3" />
      {/* Head */}
      <ellipse cx="40" cy="34" rx="18" ry="15" fill="#c4a882" opacity="0.4" />
      {/* Snout */}
      <ellipse cx="40" cy="40" rx="9" ry="7" fill="#b8956a" opacity="0.4" />
      {/* Eyes */}
      <circle cx="33" cy="31" r="2.5" fill="#5a3e2b" opacity="0.6" />
      <circle cx="47" cy="31" r="2.5" fill="#5a3e2b" opacity="0.6" />
      {/* Nostrils */}
      <circle cx="37.5" cy="39" r="1.2" fill="#5a3e2b" opacity="0.4" />
      <circle cx="42.5" cy="39" r="1.2" fill="#5a3e2b" opacity="0.4" />
      {/* Wig/hat (judge's wig) */}
      <ellipse cx="40" cy="21" rx="19" ry="6" fill="#f0ede8" opacity="0.7" />
      <rect x="21" y="16" width="38" height="8" rx="4" fill="#e8e4dc" opacity="0.6" />
      {/* Scroll in paw */}
      <rect x="54" y="46" width="12" height="16" rx="2" fill="#f5f0e8" opacity="0.6" stroke="#c4a882" strokeWidth="1" opacity="0.4" />
      <line x1="57" y1="51" x2="63" y2="51" stroke="#c4a882" strokeWidth="1" opacity="0.5" />
      <line x1="57" y1="54" x2="63" y2="54" stroke="#c4a882" strokeWidth="1" opacity="0.5" />
      <line x1="57" y1="57" x2="61" y2="57" stroke="#c4a882" strokeWidth="1" opacity="0.5" />
    </svg>
  );
}

export function Greeting({ onSuggestion }: GreetingProps) {
  const hour = new Date().getHours();
  const greeting =
    hour < 12 ? "Good morning" : hour < 17 ? "Good afternoon" : "Good evening";

  return (
    <div className="flex flex-col items-center justify-center h-full px-6 py-8 max-w-2xl mx-auto w-full">
      {/* Mascot + greeting */}
      <div className="fade-up flex flex-col items-center mb-8">
        <CapybaraIcon />
        <h1 className="mt-4 text-2xl font-semibold text-foreground tracking-tight text-center">
          {greeting}, Le Chi Lim
        </h1>
        <p className="mt-1.5 text-sm text-muted-foreground text-center max-w-sm">
          Ask me about Hong Kong and Singapore open court cases.{" "}
          <span className="text-primary">RAG-powered</span>, cited sources.
        </p>
      </div>

      {/* Skill chips */}
      <div className="fade-up fade-up-delay-1 flex flex-wrap gap-2 justify-center mb-8">
        {SKILL_CHIPS.map((chip) => (
          <Button
            key={chip.label}
            variant="outline"
            size="sm"
            className="h-8 text-xs gap-1.5 rounded-full border-border/60 text-muted-foreground hover:text-primary hover:border-primary/40 hover:bg-primary/5 transition-colors"
            onClick={() => onSuggestion(chip.label)}
          >
            <chip.icon className="h-3 w-3" />
            {chip.label}
          </Button>
        ))}
      </div>

      {/* Suggestion cards */}
      <div className="fade-up fade-up-delay-2 grid grid-cols-1 sm:grid-cols-2 gap-2.5 w-full">
        {SUGGESTIONS.map((s) => (
          <button
            key={s.label}
            onClick={() => onSuggestion(s.prompt)}
            className="group text-left rounded-xl border border-border/60 bg-card p-3.5 hover:border-primary/40 hover:bg-primary/5 transition-all duration-200 shadow-sm hover:shadow-md"
          >
            <div className="flex items-start gap-2.5">
              <span className="text-lg leading-none mt-0.5">{s.icon}</span>
              <div>
                <p className="text-xs font-medium text-foreground group-hover:text-primary transition-colors">
                  {s.prompt}
                </p>
                <span className="inline-block mt-1.5 text-[10px] text-muted-foreground bg-muted px-2 py-0.5 rounded-full">
                  {s.tag}
                </span>
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
