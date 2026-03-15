"use client";

import React from "react";
import { MessageSquare, Plus, Scale, Settings, ChevronLeft, Clock } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { ChatSession } from "@/lib/types";

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
  onNewChat: () => void;
  currentSessionId?: string;
  onSelectSession: (id: string) => void;
}

// Mock past sessions
const MOCK_SESSIONS: ChatSession[] = [
  {
    id: "sess-1",
    title: "Penalty clauses post-Cavendish",
    preview: "HK courts' position on liquidated damages...",
    timestamp: new Date(Date.now() - 1000 * 60 * 45),
    messageCount: 6,
  },
  {
    id: "sess-2",
    title: "Director fiduciary duties HK vs SG",
    preview: "Comparing approaches to director obligations...",
    timestamp: new Date(Date.now() - 1000 * 60 * 60 * 3),
    messageCount: 11,
  },
  {
    id: "sess-3",
    title: "Spandeck duty of care test",
    preview: "Singapore's two-stage framework for...",
    timestamp: new Date(Date.now() - 1000 * 60 * 60 * 24),
    messageCount: 8,
  },
  {
    id: "sess-4",
    title: "Passing off in trade marks",
    preview: "Leading SG cases on distinctiveness...",
    timestamp: new Date(Date.now() - 1000 * 60 * 60 * 48),
    messageCount: 4,
  },
];

function formatRelativeTime(date: Date): string {
  const diff = Date.now() - date.getTime();
  const minutes = Math.floor(diff / 60000);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

export function Sidebar({
  isOpen,
  onClose,
  onNewChat,
  currentSessionId,
  onSelectSession,
}: SidebarProps) {
  return (
    <TooltipProvider>
      <aside
        className={cn(
          "flex flex-col h-full bg-[var(--harvey-warm)] dark:bg-sidebar border-r border-sidebar-border sidebar-transition overflow-hidden",
          isOpen ? "w-64" : "w-0 opacity-0 pointer-events-none md:w-16 md:opacity-100 md:pointer-events-auto"
        )}
      >
        {/* Logo area */}
        <div className="flex items-center justify-between px-3 py-4 border-b border-sidebar-border h-14 shrink-0">
          {isOpen ? (
            <div className="flex items-center gap-2.5">
              <div className="w-7 h-7 rounded-lg bg-primary/10 flex items-center justify-center shrink-0">
                <Scale className="w-4 h-4 text-primary" />
              </div>
              <span className="font-semibold text-sm text-foreground tracking-tight">Harvey Lite</span>
            </div>
          ) : (
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="w-7 h-7 rounded-lg bg-primary/10 flex items-center justify-center mx-auto cursor-pointer">
                  <Scale className="w-4 h-4 text-primary" />
                </div>
              </TooltipTrigger>
              <TooltipContent side="right">Harvey Lite HK/SG</TooltipContent>
            </Tooltip>
          )}

          {isOpen && (
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7 text-muted-foreground hover:text-foreground"
              onClick={onClose}
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
          )}
        </div>

        {/* New Chat button */}
        <div className="px-2 py-3 shrink-0">
          {isOpen ? (
            <Button
              onClick={onNewChat}
              className="w-full justify-start gap-2 bg-primary/10 hover:bg-primary/20 text-primary border-0 shadow-none font-medium"
              variant="outline"
              size="sm"
            >
              <Plus className="h-4 w-4" />
              New chat
            </Button>
          ) : (
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  onClick={onNewChat}
                  variant="ghost"
                  size="icon"
                  className="w-full h-9 text-muted-foreground hover:text-primary hover:bg-primary/10"
                >
                  <Plus className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="right">New chat</TooltipContent>
            </Tooltip>
          )}
        </div>

        {/* Session list */}
        <ScrollArea className="flex-1 px-2">
          {isOpen && (
            <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-widest px-1 mb-2">
              Recent
            </p>
          )}

          <div className="space-y-0.5">
            {MOCK_SESSIONS.map((session) =>
              isOpen ? (
                <button
                  key={session.id}
                  onClick={() => onSelectSession(session.id)}
                  className={cn(
                    "w-full text-left rounded-lg px-2.5 py-2.5 group transition-colors",
                    currentSessionId === session.id
                      ? "bg-primary/10 text-foreground"
                      : "hover:bg-black/5 dark:hover:bg-white/5 text-muted-foreground"
                  )}
                >
                  <div className="flex items-start gap-2">
                    <MessageSquare className="h-3.5 w-3.5 mt-0.5 shrink-0 text-muted-foreground" />
                    <div className="min-w-0 flex-1">
                      <p className="text-xs font-medium text-foreground truncate leading-tight">
                        {session.title}
                      </p>
                      <p className="text-[10px] text-muted-foreground truncate mt-0.5">
                        {session.preview}
                      </p>
                      <div className="flex items-center gap-1.5 mt-1">
                        <Clock className="h-2.5 w-2.5 text-muted-foreground/50" />
                        <span suppressHydrationWarning className="text-[10px] text-muted-foreground/60">
  {formatRelativeTime(session.timestamp)}
                        </span>
                      </div>
                    </div>
                  </div>
                </button>
              ) : (
                <Tooltip key={session.id}>
                  <TooltipTrigger asChild>
                    <button
                      onClick={() => onSelectSession(session.id)}
                      className={cn(
                        "w-full rounded-lg p-2 flex items-center justify-center transition-colors",
                        currentSessionId === session.id
                          ? "bg-primary/10"
                          : "hover:bg-black/5 dark:hover:bg-white/5"
                      )}
                    >
                      <MessageSquare className="h-4 w-4 text-muted-foreground" />
                    </button>
                  </TooltipTrigger>
                  <TooltipContent side="right">{session.title}</TooltipContent>
                </Tooltip>
              )
            )}
          </div>
        </ScrollArea>

        {/* Bottom: User profile */}
        <div className="border-t border-sidebar-border p-2 shrink-0">
          {isOpen ? (
            <div className="flex items-center gap-2.5 px-1.5 py-2 rounded-lg hover:bg-black/5 dark:hover:bg-white/5 cursor-pointer transition-colors group">
              <Avatar className="h-7 w-7 shrink-0">
                <AvatarFallback className="text-[11px] font-medium bg-primary/10 text-primary">
                  LC
                </AvatarFallback>
              </Avatar>
              <div className="min-w-0 flex-1">
                <p className="text-xs font-medium text-foreground truncate">Le Chi Lim</p>
                <Badge
                  variant="secondary"
                  className="text-[9px] px-1.5 py-0 h-4 font-normal bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400"
                >
                  Free plan
                </Badge>
              </div>
              <Settings className="h-3.5 w-3.5 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
            </div>
          ) : (
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex items-center justify-center p-1 cursor-pointer">
                  <Avatar className="h-7 w-7">
                    <AvatarFallback className="text-[11px] font-medium bg-primary/10 text-primary">
                      LC
                    </AvatarFallback>
                  </Avatar>
                </div>
              </TooltipTrigger>
              <TooltipContent side="right">Le Chi Lim · Free plan</TooltipContent>
            </Tooltip>
          )}
        </div>
      </aside>
    </TooltipProvider>
  );
}
