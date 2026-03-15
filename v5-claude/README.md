# Harvey Lite HK/SG ⚖️

> RAG-powered legal AI assistant for Hong Kong and Singapore open court cases.

A clean, minimal chat interface inspired by modern AI tools — soft teal/blue palette, lots of whitespace, warm friendly feel. Built with Next.js 15, Tailwind, and shadcn/ui.

![Harvey Lite Screenshot](./public/screenshot.png)

---

## Features

- 💬 **Streaming chat** — SSE-based streaming with word-by-word rendering
- 📚 **RAG citations** — Source cards with jurisdiction badges (HK/SG), relevance scores, and clickable PDF preview
- 📄 **PDF viewer** — Resizable split panel with zoom, pagination, and link to original source
- 🌙 **Dark mode** — Full dark/light theme support
- 🗂️ **Collapsible sidebar** — Chat history, practice area filtering
- ⚡ **Model selector** — Swap between Claude, Grok, Ollama (extensible)
- 📱 **Responsive** — Desktop split view, mobile-first stacking

---

## Tech Stack

| Layer | Choice |
|---|---|
| Framework | Next.js 15 (App Router) |
| Language | TypeScript |
| Styling | Tailwind CSS |
| UI Components | shadcn/ui |
| Icons | lucide-react |
| PDF | react-pdf (pdfjs-dist) |
| Fonts | Geist (Vercel) |
| Themes | next-themes |

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/yourusername/harvey-lite
cd harvey-lite
npm install
```

### 2. Install shadcn/ui components

```bash
npx shadcn@latest init
# When prompted: TypeScript=yes, style=default, baseColor=slate, CSS variables=yes

# Install required components:
npx shadcn@latest add button
npx shadcn@latest add badge
npx shadcn@latest add avatar
npx shadcn@latest add card
npx shadcn@latest add textarea
npx shadcn@latest add scroll-area
npx shadcn@latest add skeleton
npx shadcn@latest add tooltip
npx shadcn@latest add dropdown-menu
npx shadcn@latest add slider
npx shadcn@latest add resizable
npx shadcn@latest add separator
```

### 3. Install additional deps

```bash
npm install react-pdf next-themes geist
```

### 4. Set up environment variables

```bash
cp .env.example .env.local
```

```env
# .env.local

# Required: Anthropic API key for real RAG responses
ANTHROPIC_API_KEY=sk-ant-...

# Optional: Your backend RAG API (Python FastAPI/Flask)
BACKEND_URL=http://localhost:8000

# Optional: For JWT/session (future)
NEXTAUTH_SECRET=your-secret-here
```

### 5. Run development server

```bash
npm run dev
# → http://localhost:3000
```

---

## Project Structure

```
harvey-lite/
├── app/
│   ├── layout.tsx              # Root layout with ThemeProvider
│   ├── page.tsx                # Main app shell (header, sidebar, panels)
│   ├── globals.css             # Tailwind + design tokens + animations
│   └── api/
│       └── chat/
│           └── route.ts        # Mock streaming API (replace with real RAG)
├── components/
│   ├── Sidebar.tsx             # Collapsible sidebar with chat history
│   ├── ChatWindow.tsx          # Chat messages + input area
│   ├── Greeting.tsx            # Empty state with suggestions
│   ├── PDFViewer.tsx           # Case document viewer
│   └── ThemeProvider.tsx       # next-themes wrapper
│   └── ui/                     # shadcn/ui components (auto-generated)
├── lib/
│   ├── types.ts                # TypeScript types
│   └── utils.ts                # cn() helper
├── public/
└── package.json
```

---

## Connecting Your Python Backend

Replace the mock API route with a call to your Python RAG backend:

```typescript
// app/api/chat/route.ts

export async function POST(req: NextRequest) {
  const { messages } = await req.json();

  // Forward to your FastAPI/Flask backend
  const res = await fetch(`${process.env.BACKEND_URL}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages }),
  });

  // Stream response back to client
  return new Response(res.body, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
    },
  });
}
```

Your Python backend should return SSE events in this format:

```
data: {"type": "sources", "sources": [...]}

data: {"type": "text", "content": "word "}

data: [DONE]
```

---

## Enabling Real PDF Rendering

The `PDFViewer` component includes a placeholder renderer. To enable real PDF rendering:

```bash
npm install react-pdf
```

In `components/PDFViewer.tsx`, replace the `PlaceholderPage` with:

```tsx
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';

// Set up worker
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url,
).toString();

// In your render:
<Document
  file={source.pdfUrl || source.url}
  onLoadSuccess={({ numPages }) => setTotalPages(numPages)}
>
  <Page pageNumber={page} scale={scale} />
</Document>
```

---

## Data Sources

| Jurisdiction | Source | Notes |
|---|---|---|
| 🇭🇰 Hong Kong | [HKLII](https://www.hklii.hk) | Free, open access |
| 🇸🇬 Singapore | [Singapore Judiciary](https://www.judiciary.gov.sg) | Free, open access |
| 🇸🇬 Singapore | [eLitigation](https://www.elitigation.sg) | Court documents |

---

## Roadmap

- [ ] Connect to real Python RAG backend
- [ ] Implement actual PDF URL fetching from HKLII/SG Judiciary
- [ ] Add semantic search with pgvector or Qdrant
- [ ] Authentication (NextAuth)
- [ ] Upgrade to Pro (Stripe integration)
- [ ] Case bookmarking and annotation
- [ ] Export to Word/PDF with citations
- [ ] Multi-jurisdiction comparison mode

---

## License

MIT
