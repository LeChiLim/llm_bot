"use client";

import React, { useRef, useEffect, useState, useCallback } from "react";
import { Send, Paperclip, StopCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { ChatMessage, CaseSource } from "@/lib/types";
import { Greeting } from "./Greeting";

// Simple markdown-ish renderer (bold, lists, code)
function renderMarkdown(text: string): React.ReactNode {
  const lines = text.split("\n");
  const elements: React.ReactNode[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Heading
    if (line.startsWith("**") && line.endsWith("**") && line.length > 4) {
      elements.push(
        <p key={i} className="font-semibold text-foreground mt-3 mb-1 first:mt-0">
          {line.slice(2, -2)}
        </p>
      );
    }
    // Numbered list
    else if (/^\d+\.\s/.test(line)) {
      const listItems: string[] = [];
      while (i < lines.length && /^\d+\.\s/.test(lines[i])) {
        listItems.push(lines[i].replace(/^\d+\.\s/, ""));
        i++;
      }
      elements.push(
        <ol key={`list-${i}`} className="list-decimal list-inside space-y-1 my-2 text-sm">
          {listItems.map((item, j) => (
            <li key={j} className="text-foreground/90">
              {renderInline(item)}
            </li>
          ))}
        </ol>
      );
      continue;
    }
    // Bullet list
    else if (line.startsWith("- ") || line.startsWith("• ")) {
      const listItems: string[] = [];
      while (i < lines.length && (lines[i].startsWith("- ") || lines[i].startsWith("• "))) {
        listItems.push(lines[i].slice(2));
        i++;
      }
      elements.push(
        <ul key={`ul-${i}`} className="list-disc list-inside space-y-1 my-2 text-sm">
          {listItems.map((item, j) => (
            <li key={j} className="text-foreground/90">
              {renderInline(item)}
            </li>
          ))}
        </ul>
      );
      continue;
    }
    // Empty line
    else if (line.trim() === "") {
      elements.push(<div key={i} className="h-1.5" />);
    }
    // Normal paragraph
    else {
      elements.push(
        <p key={i} className="text-sm text-foreground/90 leading-relaxed">
          {renderInline(line)}
        </p>
      );
    }
    i++;
  }

  return <div className="space-y-0.5">{elements}</div>;
}

function renderInline(text: string): React.ReactNode {
  // Handle **bold** and *italic*
  const parts = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={i} className="font-semibold text-foreground">{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("*") && part.endsWith("*")) {
      return <em key={i} className="italic">{part.slice(1, -1)}</em>;
    }
    return <React.Fragment key={i}>{part}</React.Fragment>;
  });
}

// Jurisdiction badge colors
function JurBadge({ jur }: { jur: string }) {
  const colors: Record<string, string> = {
    HK: "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400",
    SG: "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400",
    UK: "bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400",
  };
  return (
    <span
      className={cn(
        "inline-flex items-center px-1.5 py-0.5 rounded text-[9px] font-bold tracking-wide",
        colors[jur] || "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-300"
      )}
    >
      {jur}
    </span>
  );
}

// Source citation card
function SourceCard({
  source,
  onOpen,
}: {
  source: CaseSource;
  onOpen: (source: CaseSource) => void;
}) {
  return (
    <button
      onClick={() => onOpen(source)}
      className="group flex items-start gap-2.5 text-left w-full rounded-lg border border-border/60 bg-background hover:border-primary/40 hover:bg-primary/5 p-2.5 transition-all duration-150 shadow-sm hover:shadow"
    >
      <JurBadge jur={source.jurisdiction} />
      <div className="min-w-0 flex-1">
        <p className="text-xs font-medium text-foreground/90 group-hover:text-primary transition-colors truncate">
          {source.name}
        </p>
        <p className="text-[10px] text-muted-foreground mt-0.5 font-mono">{source.citation}</p>
        <div className="flex items-center gap-2 mt-1">
          <div className="flex-1 h-1 bg-border/60 rounded-full overflow-hidden">
            <div
              className="h-full bg-primary/60 rounded-full"
              style={{ width: `${source.relevance}%` }}
            />
          </div>
          <span className="text-[10px] text-muted-foreground shrink-0">{source.relevance}%</span>
        </div>
      </div>
    </button>
  );
}

// Typing indicator
function TypingIndicator() {
  return (
    <div className="flex items-center gap-1 px-1">
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          className="typing-dot w-1.5 h-1.5 rounded-full bg-muted-foreground/50"
        />
      ))}
    </div>
  );
}

// Single message bubble
function MessageBubble({
  message,
  onSourceOpen,
}: {
  message: ChatMessage;
  onSourceOpen: (source: CaseSource) => void;
}) {
  const isUser = message.role === "user";

  return (
    <div
      className={cn(
        "message-animate flex gap-3 px-1",
        isUser ? "flex-row-reverse" : "flex-row"
      )}
    >
      {/* Avatar */}
      <Avatar className="h-7 w-7 shrink-0 mt-0.5">
        <AvatarFallback
          className={cn(
            "text-[10px] font-semibold",
            isUser
              ? "bg-primary text-white"
              : "bg-[var(--harvey-warm-mid)] text-foreground"
          )}
        >
          {isUser ? "LC" : "⚖️"}
        </AvatarFallback>
      </Avatar>

      {/* Content */}
      <div className={cn("flex flex-col gap-2 max-w-[80%]", isUser ? "items-end" : "items-start")}>
        {/* Bubble */}
        <div
          className={cn(
            "rounded-2xl px-4 py-3 text-sm shadow-sm",
            isUser
              ? "bg-primary text-white rounded-tr-sm"
              : "bg-card border border-border/60 rounded-tl-sm"
          )}
        >
          {message.isStreaming && message.content === "" ? (
            <TypingIndicator />
          ) : isUser ? (
            <p className="leading-relaxed">{message.content}</p>
          ) : (
            renderMarkdown(message.content)
          )}
          {message.isStreaming && message.content !== "" && (
            <span className="inline-block w-0.5 h-3.5 bg-primary/60 ml-0.5 animate-pulse" />
          )}
        </div>

        {/* Sources */}
        {!isUser && message.sources && message.sources.length > 0 && !message.isStreaming && (
          <div className="w-full">
            <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-1.5 px-0.5">
              Sources retrieved
            </p>
            <div className="grid grid-cols-1 gap-1.5">
              {message.sources.map((source) => (
                <SourceCard key={source.id} source={source} onOpen={onSourceOpen} />
              ))}
            </div>
          </div>
        )}

        {/* Sources skeleton while streaming */}
        {!isUser && message.isStreaming && (
          <div className="w-full space-y-1.5 mt-1">
            <Skeleton className="h-3 w-24" />
            <Skeleton className="h-14 w-full rounded-lg" />
            <Skeleton className="h-14 w-full rounded-lg" />
          </div>
        )}
      </div>
    </div>
  );
}

interface ChatWindowProps {
  messages: ChatMessage[];
  isLoading: boolean;
  onSendMessage: (text: string) => void;
  onSourceOpen: (source: CaseSource) => void;
  onStopGeneration?: () => void;
}

export function ChatWindow({
  messages,
  isLoading,
  onSendMessage,
  onSourceOpen,
  onStopGeneration,
}: ChatWindowProps) {
  const [input, setInput] = useState("");
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto scroll to bottom
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Auto-resize textarea
  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    const ta = e.target;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 160) + "px";
  };

  const handleSend = useCallback(() => {
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;
    onSendMessage(trimmed);
    setInput("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  }, [input, isLoading, onSendMessage]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSuggestion = (text: string) => {
    onSendMessage(text);
  };

  const isEmpty = messages.length === 0;

  return (
    <div className="flex flex-col h-full">
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto" ref={scrollAreaRef}>
        {isEmpty ? (
          <Greeting onSuggestion={handleSuggestion} />
        ) : (
          <div className="max-w-2xl mx-auto w-full px-4 py-6 space-y-6">
            {messages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} onSourceOpen={onSourceOpen} />
            ))}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* Input area */}
      <div className="shrink-0 border-t border-border/60 bg-background/80 backdrop-blur-sm px-4 py-3">
        <div className="max-w-2xl mx-auto">
          <div className="relative flex items-end gap-2 rounded-2xl border border-border/80 bg-card shadow-sm focus-within:border-primary/50 focus-within:shadow-md transition-all duration-200 px-3 py-2">
            {/* Attachment (future) */}
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7 shrink-0 text-muted-foreground hover:text-foreground mb-0.5"
                    disabled
                  >
                    <Paperclip className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Attach file (coming soon)</TooltipContent>
              </Tooltip>
            </TooltipProvider>

            <Textarea
              ref={textareaRef}
              value={input}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              placeholder="Ask about HK or SG case law…"
              rows={1}
              className="flex-1 resize-none border-0 shadow-none p-0 focus-visible:ring-0 bg-transparent text-sm placeholder:text-muted-foreground/50 min-h-[28px] max-h-[160px] leading-relaxed"
            />

            {/* Send / Stop button */}
            {isLoading ? (
              <Button
                size="icon"
                className="h-8 w-8 shrink-0 rounded-xl bg-destructive/10 hover:bg-destructive/20 text-destructive border-0 shadow-none mb-0.5"
                onClick={onStopGeneration}
              >
                <StopCircle className="h-4 w-4" />
              </Button>
            ) : (
              <Button
                size="icon"
                className="h-8 w-8 shrink-0 rounded-xl bg-primary hover:bg-primary/90 text-white border-0 shadow-none mb-0.5 disabled:opacity-30"
                onClick={handleSend}
                disabled={!input.trim()}
              >
                <Send className="h-3.5 w-3.5" />
              </Button>
            )}
          </div>

          <p className="text-center text-[10px] text-muted-foreground/50 mt-2">
            Harvey Lite retrieves from HKLII and Singapore Judiciary open databases.{" "}
            <span className="text-primary/60">Always verify citations.</span>
          </p>
        </div>
      </div>
    </div>
  );
}
