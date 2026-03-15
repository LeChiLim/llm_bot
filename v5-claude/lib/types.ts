export type Jurisdiction = "HK" | "SG" | "UK" | "BOTH";

export interface CaseSource {
  id: string;
  name: string;
  citation: string;
  jurisdiction: Jurisdiction | string;
  court: string;
  year: number;
  url: string;
  relevance: number; // 0-100
  pdfUrl?: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: CaseSource[];
  timestamp: Date;
  isStreaming?: boolean;
}

export interface ChatSession {
  id: string;
  title: string;
  preview: string;
  timestamp: Date;
  messageCount: number;
}

export type ModelOption = {
  id: string;
  label: string;
  provider: string;
  available: boolean;
};
