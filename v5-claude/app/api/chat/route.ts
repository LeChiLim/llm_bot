// import { NextRequest } from "next/server";

// // Mock sources returned with each response
// const MOCK_SOURCES = {
//   penalty: [
//     {
//       id: "hk-cathay-2020",
//       name: "Cathay Pacific v Lufthansa Technik",
//       citation: "[2020] HKCFI 2092",
//       jurisdiction: "HK",
//       court: "Court of First Instance",
//       year: 2020,
//       url: "https://www.hklii.hk/en/cases/hkcfi/2020/2092",
//       relevance: 94,
//     },
//     {
//       id: "sg-denka-2021",
//       name: "Denka Advantech v Seraya Energy",
//       citation: "[2021] 1 SLR 631",
//       jurisdiction: "SG",
//       court: "Court of Appeal",
//       year: 2021,
//       url: "https://www.elitigation.sg/gd/s/2020_SGCA_119",
//       relevance: 87,
//     },
//     {
//       id: "uk-cavendish-2015",
//       name: "Cavendish Square v Makdessi",
//       citation: "[2015] UKSC 67",
//       jurisdiction: "UK",
//       court: "UK Supreme Court",
//       year: 2015,
//       url: "https://www.hklii.hk",
//       relevance: 82,
//     },
//   ],
//   duty: [
//     {
//       id: "sg-spandeck-2007",
//       name: "Spandeck Engineering v DSTA",
//       citation: "[2007] SGCA 37",
//       jurisdiction: "SG",
//       court: "Court of Appeal",
//       year: 2007,
//       url: "https://www.elitigation.sg",
//       relevance: 97,
//     },
//     {
//       id: "hk-bank-east-asia",
//       name: "Bank of East Asia v Tsien Wui",
//       citation: "[1998] 1 HKLRD 188",
//       jurisdiction: "HK",
//       court: "Court of Appeal",
//       year: 1998,
//       url: "https://www.hklii.hk",
//       relevance: 76,
//     },
//   ],
//   default: [
//     {
//       id: "hk-default-1",
//       name: "HKSAR v Lam Kwong Wai",
//       citation: "[2006] 3 HKLRD 808",
//       jurisdiction: "HK",
//       court: "Court of Final Appeal",
//       year: 2006,
//       url: "https://www.hklii.hk",
//       relevance: 65,
//     },
//     {
//       id: "sg-default-1",
//       name: "Yeo Tiong Min v Public Prosecutor",
//       citation: "[2011] 2 SLR 1156",
//       jurisdiction: "SG",
//       court: "Court of Appeal",
//       year: 2011,
//       url: "https://www.elitigation.sg",
//       relevance: 61,
//     },
//   ],
// };

// const MOCK_RESPONSES = {
//   penalty: `**Penalty Clauses in Hong Kong: Post-Cavendish Position**

// Hong Kong courts have largely embraced the reformulated test from the UK Supreme Court in *Cavendish Square Holding BV v Makdessi* [2015] UKSC 67, departing from the traditional "genuine pre-estimate of loss" standard.

// **The New Test**

// The key question is now whether the clause is **out of all proportion** to the legitimate interest the innocent party has in the performance of the primary obligation. This is a significantly higher threshold for striking down a clause than the old approach.

// **Leading HK Authority**

// In *Cathay Pacific Airways Ltd v Lufthansa Technik AG* [2020] HKCFI 2092, the Court of First Instance applied the Cavendish framework and emphasised that:

// 1. The commercial context and bargaining sophistication of the parties is paramount
// 2. Courts should be slow to interfere with freely negotiated commercial arrangements
// 3. A clause serving a legitimate business interest will rarely be characterised as a penalty

// **Singapore Approach**

// Singapore has also moved in a similar direction via *Denka Advantech Pte Ltd v Seraya Energy Pte Ltd* [2021] 1 SLR 631, where the Court of Appeal confirmed the Cavendish test applies in Singapore.

// Both jurisdictions now afford considerably more latitude to liquidated damages clauses in commercial contracts between sophisticated parties.`,

//   duty: `**Duty of Care: HK vs Singapore — A Comparative Analysis**

// The two jurisdictions have diverged significantly in their approach to establishing a duty of care in negligence.

// **Singapore: The Spandeck Framework**

// Singapore developed its own two-stage test in *Spandeck Engineering (S) Pte Ltd v Defence Science & Technology Agency* [2007] SGCA 37:

// 1. **Threshold stage**: Factual foreseeability of harm
// 2. **Main stage**: (a) Proximity, and (b) Policy considerations that may negate a prima facie duty

// The Spandeck test is applied **universally** regardless of the type of loss (physical, psychiatric, pure economic). This creates a more unified and predictable framework.

// **Hong Kong: The Caparo Approach**

// HK courts continue to apply the tripartite *Caparo* test from English law:
// 1. Foreseeability of damage
// 2. Sufficient proximity of relationship  
// 3. Fair, just and reasonable to impose a duty

// The HK Court of Final Appeal has not definitively departed from Caparo, leaving room for incremental development by analogy with established categories.

// **Practical Implication**

// For pure economic loss, Singapore's Spandeck framework — particularly through the proximity analysis — tends to produce more expansive recovery than HK's approach, which applies Caparo with greater conservatism in non-physical loss cases.`,

//   default: `**Research Results**

// Based on the available HK and SG open case law database, here are the most relevant authorities I've identified for your query.

// Both Hong Kong and Singapore share a common law foundation inherited from English law, but have developed distinct jurisprudential approaches over the decades since their respective periods of legal independence.

// Key points of divergence typically arise in:

// - **Duty of care** in tort (Singapore's Spandeck vs HK's Caparo approach)  
// - **Implied terms** in contract (Singapore's stricter business efficacy test)
// - **Sentencing principles** in criminal law
// - **Statutory interpretation** methodology

// I'd recommend narrowing your query to a specific area of law or citing a particular case for more targeted results. You can also use the **HK Mode** or **SG Mode** chips to restrict retrieval to one jurisdiction.`,
// };

// function getResponseType(message: string): keyof typeof MOCK_RESPONSES {
//   const lower = message.toLowerCase();
//   if (lower.includes("penalty") || lower.includes("cavendish") || lower.includes("liquidated")) {
//     return "penalty";
//   }
//   if (
//     lower.includes("duty") ||
//     lower.includes("negligence") ||
//     lower.includes("spandeck") ||
//     lower.includes("caparo")
//   ) {
//     return "duty";
//   }
//   return "default";
// }

// export async function POST(req: NextRequest) {
//   const { messages } = await req.json();
//   const lastMessage = messages[messages.length - 1]?.content || "";
//   const responseType = getResponseType(lastMessage);

//   const responseText = MOCK_RESPONSES[responseType];
//   const sources = MOCK_SOURCES[responseType];

//   // Simulate streaming with chunked response
//   const encoder = new TextEncoder();

//   const stream = new ReadableStream({
//     async start(controller) {
//       // First send sources as a special JSON chunk
//       const sourcesChunk = `data: ${JSON.stringify({ type: "sources", sources })}\n\n`;
//       controller.enqueue(encoder.encode(sourcesChunk));

//       // Small delay before text starts
//       await new Promise((r) => setTimeout(r, 200));

//       // Stream the text word by word
//       const words = responseText.split(" ");
//       for (let i = 0; i < words.length; i++) {
//         const word = words[i] + (i < words.length - 1 ? " " : "");
//         const chunk = `data: ${JSON.stringify({ type: "text", content: word })}\n\n`;
//         controller.enqueue(encoder.encode(chunk));
//         await new Promise((r) => setTimeout(r, 18 + Math.random() * 12));
//       }

//       controller.enqueue(encoder.encode("data: [DONE]\n\n"));
//       controller.close();
//     },
//   });

//   return new Response(stream, {
//     headers: {
//       "Content-Type": "text/event-stream",
//       "Cache-Control": "no-cache",
//       Connection: "keep-alive",
//     },
//   });
// }

import { NextRequest } from "next/server";
import { spawn } from "child_process";
import path from "path";

export async function POST(req: NextRequest) {
  const { messages } = await req.json();

  const encoder = new TextEncoder();

  const stream = new ReadableStream({
    async start(controller) {
      // Spawn the Python script
      const scriptPath = path.join(process.cwd(), "backend", "chat.py");
      const python = spawn("python3", [scriptPath]);

      // Send messages to Python via stdin
      python.stdin.write(JSON.stringify(messages));
      python.stdin.end();

      // Forward Python's stdout (SSE chunks) to the browser
      python.stdout.on("data", (data: Buffer) => {
        const lines = data.toString().split("\n").filter(Boolean);
        for (const line of lines) {
          controller.enqueue(encoder.encode(line + "\n\n"));
        }
      });

      python.stderr.on("data", (err: Buffer) => {
        console.error("Python error:", err.toString());
      });

      python.on("close", () => {
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
        controller.close();
      });
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}
