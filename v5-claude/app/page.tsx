"use client";

import React, { useState, useCallback, useRef } from "react";
import { PanelLeftOpen, Moon, Sun, ChevronDown, Zap } from "lucide-react";
import { useTheme } from "next-themes";
import { Button } from "@/components/ui/button";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { Sidebar } from "@/components/Sidebar";
import { ChatWindow } from "@/components/ChatWindow";
import { PDFViewer } from "@/components/PDFViewer";
import { ChatMessage, CaseSource, ModelOption } from "@/lib/types";

const MODEL_OPTIONS: ModelOption[] = [
  { id: "claude-sonnet-4-5", label: "Claude Sonnet 4.5", provider: "Anthropic", available: true },
  { id: "grok-2", label: "Grok 2", provider: "xAI", available: false },
  { id: "ollama-llama3", label: "Llama 3 (Ollama)", provider: "Local", available: false },
  { id: "gpt-4o", label: "GPT-4o", provider: "OpenAI", available: false },
];

let messageIdCounter = 0;
const newId = () => `msg-${++messageIdCounter}`;

export default function HomePage() {
  const { theme, setTheme } = useTheme();
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [selectedModel, setSelectedModel] = useState(MODEL_OPTIONS[0]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [activePDF, setActivePDF] = useState<CaseSource | null>(null);
  const [currentSessionId, setCurrentSessionId] = useState<string | undefined>(undefined);
  const abortRef = useRef<AbortController | null>(null);

  const handleSendMessage = useCallback(async (text: string) => {
    if (isLoading) return;
    const userMsg: ChatMessage = { id: newId(), role: "user", content: text, timestamp: new Date() };
    const assistantMsg: ChatMessage = { id: newId(), role: "assistant", content: "", timestamp: new Date(), isStreaming: true };
    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setIsLoading(true);
    abortRef.current = new AbortController();
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: [...messages, userMsg].map((m) => ({ role: m.role, content: m.content })) }),
        signal: abortRef.current.signal,
      });
      if (!res.body) throw new Error("No response body");
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let accText = "";
      let sources: CaseSource[] = [];
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n").filter((l) => l.startsWith("data: "));
        for (const line of lines) {
          const data = line.slice(6).trim();
          if (data === "[DONE]") break;
          try {
            const parsed = JSON.parse(data);
            if (parsed.type === "sources") {
              sources = parsed.sources;
              setMessages((prev) => prev.map((m) => m.id === assistantMsg.id ? { ...m, sources } : m));
            } else if (parsed.type === "text") {
              accText += parsed.content;
              setMessages((prev) => prev.map((m) => m.id === assistantMsg.id ? { ...m, content: accText, isStreaming: true } : m));
            }
          } catch { /* skip malformed chunk */ }
        }
      }
      setMessages((prev) => prev.map((m) => m.id === assistantMsg.id ? { ...m, content: accText, sources, isStreaming: false } : m));
    } catch (err: any) {
      if (err.name !== "AbortError") {
        setMessages((prev) => prev.map((m) => m.id === assistantMsg.id ? { ...m, content: "Sorry, something went wrong. Please try again.", isStreaming: false } : m));
      }
    } finally {
      setIsLoading(false);
    }
  }, [isLoading, messages]);

  const handleStopGeneration = useCallback(() => {
    abortRef.current?.abort();
    setIsLoading(false);
    setMessages((prev) => prev.map((m) => (m.isStreaming ? { ...m, isStreaming: false } : m)));
  }, []);

  const handleSourceOpen = useCallback((source: CaseSource) => { setActivePDF(source); }, []);
  const handleNewChat = useCallback(() => { setMessages([]); setActivePDF(null); setCurrentSessionId(undefined); }, []);
  const handleSelectSession = useCallback((id: string) => { setCurrentSessionId(id); setMessages([]); setActivePDF(null); }, []);

  return (
    <TooltipProvider>
      <div className="flex h-screen overflow-hidden bg-background">
        <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} onNewChat={handleNewChat} currentSessionId={currentSessionId} onSelectSession={handleSelectSession} />
        <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
          <header className="shrink-0 h-14 border-b border-border/60 bg-background/80 backdrop-blur-sm flex items-center gap-3 px-4 z-10">
            {!sidebarOpen && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground hover:text-foreground" onClick={() => setSidebarOpen(true)}>
                    <PanelLeftOpen className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Open sidebar</TooltipContent>
              </Tooltip>
            )}
            {!sidebarOpen && (
              <div className="flex items-center gap-2 mr-2">
                <span className="text-base font-semibold tracking-tight text-foreground">Harvey Lite</span>
                <span className="text-xs text-muted-foreground">HK/SG</span>
              </div>
            )}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="sm" className="h-8 gap-1.5 text-xs font-medium border-border/60 hover:border-primary/40 hover:bg-primary/5 text-foreground">
                  <Zap className="h-3 w-3 text-primary" />
                  {selectedModel.label}
                  <ChevronDown className="h-3 w-3 text-muted-foreground" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="start" className="w-52">
                {MODEL_OPTIONS.map((model, i) => (
                  <React.Fragment key={model.id}>
                    {i > 0 && MODEL_OPTIONS[i - 1].provider !== model.provider && <DropdownMenuSeparator />}
                    <DropdownMenuItem onClick={() => model.available && setSelectedModel(model)} className={cn("flex items-center justify-between", !model.available && "opacity-50 cursor-not-allowed")} disabled={!model.available}>
                      <div>
                        <p className="text-xs font-medium">{model.label}</p>
                        <p className="text-[10px] text-muted-foreground">{model.provider}</p>
                      </div>
                      {!model.available && <Badge variant="secondary" className="text-[9px] px-1.5 py-0 h-4">Soon</Badge>}
                      {model.id === selectedModel.id && <div className="w-1.5 h-1.5 rounded-full bg-primary" />}
                    </DropdownMenuItem>
                  </React.Fragment>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
            <div className="flex items-center gap-1.5 ml-1">
              <Badge className="text-[10px] px-2 py-0.5 h-5 bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400 border-0 font-bold">HK</Badge>
              <Badge className="text-[10px] px-2 py-0.5 h-5 bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400 border-0 font-bold">SG</Badge>
            </div>
            <div className="flex-1" />
            <Badge variant="outline" className="text-[10px] px-2 py-0.5 h-5 border-amber-300 text-amber-600 dark:border-amber-700 dark:text-amber-400 hidden sm:flex">Free plan</Badge>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground hover:text-foreground" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>
                  <Sun className="h-4 w-4 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
                  <Moon className="absolute h-4 w-4 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Toggle theme</TooltipContent>
            </Tooltip>
            <Avatar className="h-7 w-7">
              <AvatarFallback className="text-[11px] font-semibold bg-primary/10 text-primary">LC</AvatarFallback>
            </Avatar>
          </header>

          {/* Content — chat + optional PDF panel side by side */}
          <div className="flex flex-1 min-h-0">
            <div className={cn("flex flex-col min-h-0 transition-all duration-300", activePDF ? "flex-1" : "w-full")}>
              <ChatWindow messages={messages} isLoading={isLoading} onSendMessage={handleSendMessage} onSourceOpen={handleSourceOpen} onStopGeneration={handleStopGeneration} />
            </div>
            {activePDF && (
              <div className="w-[400px] shrink-0 border-l border-border/60 flex flex-col min-h-0">
                <PDFViewer source={activePDF} onClose={() => setActivePDF(null)} />
              </div>
            )}
          </div>

        </div>
      </div>
    </TooltipProvider>
  );
}
