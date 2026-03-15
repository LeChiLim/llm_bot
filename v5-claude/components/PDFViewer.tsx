"use client";

import React, { useState, useCallback } from "react";
import { X, ExternalLink, ZoomIn, ZoomOut, ChevronLeft, ChevronRight, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Slider } from "@/components/ui/slider";
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { CaseSource } from "@/lib/types";

interface PDFViewerProps {
  source: CaseSource | null;
  onClose: () => void;
}

// Jurisdiction badge color
function JurBadge({ jur }: { jur: string }) {
  const colors: Record<string, string> = {
    HK: "bg-red-100 text-red-700 border-red-200 dark:bg-red-900/30 dark:text-red-400 dark:border-red-800",
    SG: "bg-blue-100 text-blue-700 border-blue-200 dark:bg-blue-900/30 dark:text-blue-400 dark:border-blue-800",
    UK: "bg-purple-100 text-purple-700 border-purple-200 dark:bg-purple-900/30 dark:text-purple-400 dark:border-purple-800",
  };
  return (
    <span
      className={cn(
        "inline-flex items-center px-2 py-0.5 rounded-md text-[10px] font-bold tracking-wide border",
        colors[jur] || "bg-gray-100 text-gray-600 border-gray-200"
      )}
    >
      {jur}
    </span>
  );
}

// Placeholder PDF page (used when react-pdf not available or PDF loading)
function PlaceholderPage({
  source,
  scale,
  page,
  totalPages,
}: {
  source: CaseSource;
  scale: number;
  page: number;
  totalPages: number;
}) {
  const lines = [
    `${source.court.toUpperCase()}`,
    "",
    source.name,
    source.citation,
    "",
    "─".repeat(40),
    "",
    "JUDGMENT",
    "",
    `Page ${page} of ${totalPages}`,
    "",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod",
    "tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim",
    "veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea",
    "commodo consequat.",
    "",
    "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum",
    "dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non",
    "proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
    "",
    "HELD: ...",
    "",
    "The court finds that...",
  ];

  return (
    <div
      className="bg-white shadow-lg mx-auto my-4 font-mono text-xs text-gray-800 p-8 leading-relaxed"
      style={{
        width: `${595 * scale}px`,
        minHeight: `${842 * scale}px`,
        fontSize: `${12 * scale}px`,
        transformOrigin: "top center",
      }}
    >
      {lines.map((line, i) => (
        <div key={i} className={cn("", !line && "h-3")}>
          {line || ""}
        </div>
      ))}
    </div>
  );
}

export function PDFViewer({ source, onClose }: PDFViewerProps) {
  const [scale, setScale] = useState(0.9);
  const [page, setPage] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const TOTAL_PAGES = 12; // Mock

  const handleZoomIn = () => setScale((s) => Math.min(s + 0.15, 2));
  const handleZoomOut = () => setScale((s) => Math.max(s - 0.15, 0.4));
  const handlePrevPage = () => setPage((p) => Math.max(p - 1, 1));
  const handleNextPage = () => setPage((p) => Math.min(p + 1, TOTAL_PAGES));

  if (!source) return null;

  return (
    <TooltipProvider>
      <div className="flex flex-col h-full slide-in-right">
        {/* Header */}
        <div className="shrink-0 border-b border-border/60 bg-background/80 backdrop-blur-sm px-4 py-2.5">
          <div className="flex items-start justify-between gap-2 mb-1.5">
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2 mb-0.5">
                <JurBadge jur={source.jurisdiction} />
                <span className="text-[10px] text-muted-foreground font-mono">
                  {source.citation}
                </span>
              </div>
              <h3 className="text-sm font-semibold text-foreground leading-tight truncate">
                {source.name}
              </h3>
              <p className="text-[10px] text-muted-foreground mt-0.5">{source.court}</p>
            </div>

            <div className="flex items-center gap-1 shrink-0">
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7 text-muted-foreground hover:text-primary"
                    onClick={() => window.open(source.url, "_blank")}
                  >
                    <ExternalLink className="h-3.5 w-3.5" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Open original source</TooltipContent>
              </Tooltip>

              <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7 text-muted-foreground hover:text-destructive"
                onClick={onClose}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          </div>

          {/* Controls */}
          <div className="flex items-center gap-2">
            {/* Zoom */}
            <div className="flex items-center gap-1">
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6 text-muted-foreground"
                onClick={handleZoomOut}
              >
                <ZoomOut className="h-3.5 w-3.5" />
              </Button>
              <span className="text-[10px] font-mono text-muted-foreground w-8 text-center">
                {Math.round(scale * 100)}%
              </span>
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6 text-muted-foreground"
                onClick={handleZoomIn}
              >
                <ZoomIn className="h-3.5 w-3.5" />
              </Button>
            </div>

            <div className="w-px h-4 bg-border" />

            {/* Pagination */}
            <div className="flex items-center gap-1">
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6 text-muted-foreground"
                onClick={handlePrevPage}
                disabled={page === 1}
              >
                <ChevronLeft className="h-3.5 w-3.5" />
              </Button>
              <span className="text-[10px] font-mono text-muted-foreground whitespace-nowrap">
                {page} / {TOTAL_PAGES}
              </span>
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6 text-muted-foreground"
                onClick={handleNextPage}
                disabled={page === TOTAL_PAGES}
              >
                <ChevronRight className="h-3.5 w-3.5" />
              </Button>
            </div>

            {/* Relevance badge */}
            <div className="ml-auto flex items-center gap-1.5">
              <span className="text-[10px] text-muted-foreground">Relevance</span>
              <Badge
                variant="secondary"
                className="text-[10px] px-1.5 py-0 h-4 bg-primary/10 text-primary border-0"
              >
                {source.relevance}%
              </Badge>
            </div>
          </div>
        </div>

        {/* PDF Content area */}
        <div className="flex-1 overflow-auto bg-muted/30">
          {isLoading ? (
            <div className="flex items-center justify-center h-full">
              <div className="flex flex-col items-center gap-3 text-muted-foreground">
                <Loader2 className="h-6 w-6 animate-spin" />
                <span className="text-xs">Loading document…</span>
              </div>
            </div>
          ) : (
            <div className="overflow-x-auto pb-4">
              {/* 
                In production, replace PlaceholderPage with react-pdf:
                
                import { Document, Page } from 'react-pdf';
                import { pdfjs } from 'react-pdf';
                pdfjs.GlobalWorkerOptions.workerSrc = new URL(
                  'pdfjs-dist/build/pdf.worker.min.mjs',
                  import.meta.url,
                ).toString();

                <Document file={source.pdfUrl || source.url}>
                  <Page pageNumber={page} scale={scale} />
                </Document>
              */}
              <PlaceholderPage
                source={source}
                scale={scale}
                page={page}
                totalPages={TOTAL_PAGES}
              />
            </div>
          )}
        </div>

        {/* Footer note */}
        <div className="shrink-0 border-t border-border/60 px-4 py-2 bg-background">
          <p className="text-[10px] text-muted-foreground text-center">
            Source:{" "}
            <button
              className="text-primary hover:underline"
              onClick={() => window.open(source.url, "_blank")}
            >
              {source.jurisdiction === "HK" ? "HKLII" : "Singapore Judiciary / eLitigation"}
            </button>{" "}
            · Open access · {source.year}
          </p>
        </div>
      </div>
    </TooltipProvider>
  );
}
