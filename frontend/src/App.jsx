import { useState, useEffect, useRef, useMemo } from "react";
import {
  ArrowRight,
  Copy,
  Check,
  CornerDownLeft,
  ThumbsUp,
  ThumbsDown,
  Clock,
  X,
  ChevronDown,
  Sliders,
  Loader2,
  RotateCw,
  Trash2,
  Layers,
  Cpu,
  Zap,
  GitBranch,
  ShieldCheck,
  Search,
  Plus,
  Minus,
  Sparkles,
} from "lucide-react";

/* ------------------------------------------------------------------ *
 * Hone AI — a prompt optimizer.
 * Describe what you want the AI to do; Hone AI rewrites it into a clear,
 * structured prompt and streams it back token-by-token (SSE-style).
 *
 * Palette: pure-black, high-contrast monochrome with a white glow —
 * following the reference layout exactly.
 * Colors that Tailwind's core scale can't hit exactly are applied
 * inline so the whole thing renders without a Tailwind compiler.
 * ------------------------------------------------------------------ */

const C = {
  bg: "#000000",
  panel: "#0a0a0a",
  panel2: "#0d0d0d",
  line: "rgba(255,255,255,0.09)",
  lineStrong: "rgba(255,255,255,0.16)",
  text: "#ededed",
  mute: "#8b8b8b",
  dim: "#5f5f5f",
  white: "#fafafa",
};

const SANS = "'Inter', ui-sans-serif, system-ui, sans-serif";
const DISPLAY = "'Space Grotesk', 'Inter', sans-serif";
const MONO = "'JetBrains Mono', ui-monospace, monospace";

/* -------------------------- prompt builder -------------------------- */
/* Turns a plain goal into a structured, production-grade prompt.
   This is what the "analyze → draft → critique → refine" pipeline
   produces; here it runs client-side so the demo is fully interactive. */
function buildPrompt(goalRaw, { model, tone, format }) {
  const goal =
    (goalRaw || "").trim() ||
    "Explain a complex topic clearly to a beginner.";

  const clean = goal.replace(/\s+/g, " ").replace(/[.\s]+$/, "");
  const first = clean.charAt(0).toUpperCase() + clean.slice(1);

  const role = inferRole(clean);
  const formatBlock =
    format === "JSON"
      ? "Return a single JSON object. No prose outside the JSON. Use snake_case keys."
      : format === "Steps"
      ? "Return a numbered list of steps. Keep each step to one action."
      : format === "Table"
      ? "Return a Markdown table with a header row. One row per item."
      : "Return clean Markdown. Use short paragraphs and headings where helpful.";

  return `You are ${role}. You are careful, concrete, and never pad your answers.

## Task
${first}.

## Context
- Target model: ${model}
- The reader wants a result they can use immediately, not a lecture.
- Prefer real examples over abstract description.

## Requirements
- Voice: ${tone.toLowerCase()}, plain, and free of filler.
- Lead with the answer; put reasoning after, only if it helps.
- Be specific: use names, numbers, and concrete examples.
- If a key detail is missing, ask exactly one clarifying question first.

## Output format
${formatBlock}

## Guardrails
- Do not invent facts, sources, or statistics.
- If you are unsure, say so plainly instead of guessing.
- Stop when the task is done — no summaries of what you just wrote.`;
}

function inferRole(g) {
  const s = g.toLowerCase();
  if (/(code|function|bug|api|python|javascript|sql|regex|program)/.test(s))
    return "a senior software engineer who writes production-quality code";
  if (/(email|reply|message|outreach|cold|linkedin)/.test(s))
    return "a sharp communications lead who writes messages people actually reply to";
  if (/(market|ad|copy|landing|brand|slogan|tagline)/.test(s))
    return "a marketing strategist with a copywriter's instincts";
  if (/(essay|blog|article|story|write|draft|post)/.test(s))
    return "an experienced editor with a clear, unfussy writing style";
  if (/(plan|strategy|roadmap|business|analyze|research)/.test(s))
    return "an analyst who turns messy questions into decisions";
  if (/(teach|explain|learn|beginner|understand)/.test(s))
    return "a patient teacher who explains things in plain language";
  return "a domain expert who gives direct, well-organized answers";
}

/* Split into tokens (words + whitespace) for realistic streaming. */
function tokenize(str) {
  return str.match(/\s+|\S+/g) || [];
}

const est = (s) => Math.max(1, Math.ceil(s.length / 4)); // rough token count

/* ================================================================== *
 * API LAYER — talks to the FastAPI backend.
 *
 * Set your backend URL here (or via a build-time env in your real repo).
 * Assumed contract (adjust the paths to match your routes):
 *   POST   {BASE}/optimize   body {goal,model,tone,format}  → text/event-stream
 *            SSE events:  event: stage  data: {"index":0|1|2}
 *                         event: token  data: {"text":"..."}   (raw text also ok)
 *                         event: done   data: {}
 *   POST   {BASE}/auth/login   {email,password}      → {access_token, user}
 *   POST   {BASE}/auth/signup  {name,email,password} → {access_token, user}
 *   GET    {BASE}/prompts                            → [{id,goal,model,rating,created_at}]
 *   POST   {BASE}/prompts      {goal,model,prompt,rating} → {id,...}
 *   DELETE {BASE}/prompts/{id}
 * The refresh token is expected to live in an httpOnly cookie, so every
 * request sends credentials; the short-lived access token is passed as a
 * Bearer header and kept in memory (never localStorage).
 * ================================================================== */
const API_BASE = import.meta.env.VITE_API_BASE || "";

const authHeader = (t) => (t ? { Authorization: `Bearer ${t}` } : {});

function parseSSE(block, { onStage, onToken }) {
  let event = "message";
  const data = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
  }
  const payload = data.join("\n");
  if (!payload) return;
  if (event === "stage") {
    try { onStage(JSON.parse(payload).index); } catch {}
  } else if (event !== "done") {
    let text = payload;
    try { const j = JSON.parse(payload); text = j.text ?? j.token ?? payload; } catch {}
    onToken(text);
  }
}

const api = {
  // Streams the optimized prompt from FastAPI over SSE (fetch + ReadableStream,
  // so we can send a JSON body and an auth header — which EventSource can't).
  async optimize({ goal, opts, token, signal, onStage, onToken }) {
    const res = await fetch(`${API_BASE}/optimize`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeader(token) },
      credentials: "include",
      body: JSON.stringify({ goal, ...opts }),
      signal,
    });
    if (!res.ok || !res.body) throw new Error(`optimize ${res.status}`);
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      // The SSE spec allows CRLF, LF, or CR as the line terminator — some
      // servers (sse-starlette included) send CRLF, so normalize before
      // splitting on the blank-line block separator.
      buf += dec.decode(value, { stream: true }).replace(/\r\n/g, "\n");
      const blocks = buf.split("\n\n");
      buf = blocks.pop() || "";
      for (const b of blocks) parseSSE(b, { onStage, onToken });
    }
    if (buf.trim()) parseSSE(buf, { onStage, onToken });
  },

  async auth(mode, body) {
    const res = await fetch(`${API_BASE}/auth/${mode}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`auth ${res.status}`);
    return res.json(); // { access_token, user }
  },

  async listPrompts(token) {
    const res = await fetch(`${API_BASE}/prompts`, {
      headers: authHeader(token),
      credentials: "include",
    });
    if (!res.ok) throw new Error(`prompts ${res.status}`);
    return res.json();
  },

  async savePrompt(token, row) {
    const res = await fetch(`${API_BASE}/prompts`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeader(token) },
      credentials: "include",
      body: JSON.stringify(row),
    });
    if (!res.ok) throw new Error(`save ${res.status}`);
    return res.json();
  },

  async deletePrompt(token, id) {
    await fetch(`${API_BASE}/prompts/${id}`, {
      method: "DELETE",
      headers: authHeader(token),
      credentials: "include",
    });
  },

  // Rotates the refresh cookie into a fresh access token — used to restore
  // a session on page load without asking the user to log in again.
  async refresh() {
    const res = await fetch(`${API_BASE}/auth/refresh`, {
      method: "POST",
      credentials: "include",
    });
    if (!res.ok) throw new Error(`refresh ${res.status}`);
    return res.json(); // { access_token, user }
  },

  async logout() {
    await fetch(`${API_BASE}/auth/logout`, {
      method: "POST",
      credentials: "include",
    });
  },
};

/* ============================== APP ============================== */
export default function App() {
  const [view, setView] = useState("home"); // 'home' | 'app'
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [auth, setAuth] = useState(null); // null | 'login' | 'signup'
  const [history, setHistory] = useState([]);
  const [reduce, setReduce] = useState(false);
  const [exampleReq, setExampleReq] = useState(null); // {goal, nonce}

  // load fonts + reduced-motion pref
  useEffect(() => {
    const l = document.createElement("link");
    l.rel = "stylesheet";
    l.href =
      "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap";
    document.head.appendChild(l);
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduce(mq.matches);
    const fn = (e) => setReduce(e.matches);
    mq.addEventListener?.("change", fn);
    return () => mq.removeEventListener?.("change", fn);
  }, []);

  // On first load, try to turn the httpOnly refresh cookie (if any) into a
  // fresh access token, so a reload doesn't sign the user out.
  useEffect(() => {
    api
      .refresh()
      .then(({ access_token, user }) => {
        setToken(access_token);
        setUser(user);
      })
      .catch(() => {});
  }, []);

  // Once we know who's signed in, load their saved prompts from the backend.
  useEffect(() => {
    if (!user || !token) {
      setHistory([]);
      return;
    }
    api
      .listPrompts(token)
      .then((rows) =>
        setHistory(
          rows.map((r) => ({
            id: r.id,
            goal: r.goal,
            model: r.model,
            rating: r.rating,
            at: new Date(r.created_at).getTime(),
          }))
        )
      )
      .catch(() => {});
  }, [user, token]);

  const signOut = () => {
    api.logout().catch(() => {});
    setUser(null);
    setToken(null);
    setHistory([]);
  };

  return (
    <div
      style={{ background: C.bg, color: C.text, fontFamily: SANS }}
      className="flex min-h-screen w-full flex-col overflow-x-hidden antialiased"
    >
      <StyleTag />
      <TopNav
        view={view}
        setView={setView}
        user={user}
        onAuth={(m) => setAuth(m)}
        onSignOut={signOut}
      />

      <div className="flex-1">
        {view === "home" ? (
          <Home
            reduce={reduce}
            onStart={() => setView("app")}
            onExample={(g) => {
              setExampleReq({ goal: g, nonce: Date.now() });
              setView("app");
            }}
          />
        ) : (
          <Workspace
            reduce={reduce}
            user={user}
            token={token}
            history={history}
            setHistory={setHistory}
            requireAuth={() => setAuth("signup")}
            exampleReq={exampleReq}
          />
        )}
      </div>

      <Footer />

      {auth && (
        <AuthModal
          mode={auth}
          switchMode={(m) => setAuth(m)}
          onClose={() => setAuth(null)}
          onDone={({ access_token, user }) => {
            setToken(access_token);
            setUser(user);
            setAuth(null);
          }}
        />
      )}
    </div>
  );
}

/* ------------------------------ nav ------------------------------ */
function TopNav({ view, setView, user, onAuth, onSignOut }) {
  return (
    <header
      className="sticky top-0 z-40 w-full backdrop-blur"
      style={{
        borderBottom: `1px solid ${C.line}`,
        background: "rgba(0,0,0,0.6)",
      }}
    >
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-5 sm:px-8">
        <button
          onClick={() => setView("home")}
          className="flex items-center gap-2.5 outline-none"
        >
          <Mark size={22} />
          <span
            className="text-[15px] font-semibold tracking-tight"
            style={{ fontFamily: DISPLAY, color: C.white }}
          >
            Hone AI
          </span>
        </button>

        <div className="flex items-center gap-2">
          {user ? (
            <div className="flex items-center gap-3">
              <span className="hidden text-sm sm:block" style={{ color: C.mute }}>
                {user.name}
              </span>
              <div
                className="flex h-8 w-8 items-center justify-center rounded-full text-xs font-semibold"
                style={{ background: C.white, color: "#000" }}
              >
                {user.name.slice(0, 1).toUpperCase()}
              </div>
              <button
                onClick={onSignOut}
                className="text-sm transition-colors"
                style={{ color: C.dim }}
                onMouseEnter={(e) => (e.currentTarget.style.color = C.text)}
                onMouseLeave={(e) => (e.currentTarget.style.color = C.dim)}
              >
                Sign out
              </button>
            </div>
          ) : (
            <>
              <button
                onClick={() => onAuth("login")}
                className="rounded-full px-3.5 py-1.5 text-sm transition-colors"
                style={{ color: C.mute }}
                onMouseEnter={(e) => (e.currentTarget.style.color = C.text)}
                onMouseLeave={(e) => (e.currentTarget.style.color = C.mute)}
              >
                Log in
              </button>
              <button
                onClick={() => onAuth("signup")}
                className="rounded-full px-4 py-1.5 text-sm font-medium transition-transform active:scale-95"
                style={{ background: C.white, color: "#000" }}
              >
                Sign up
              </button>
            </>
          )}
        </div>
      </div>
    </header>
  );
}

/* ------------------------------ home ------------------------------ */
const EXAMPLE_GOAL =
  "Write a cold email to a startup founder asking for 15 minutes to demo our product.";

function Home({ onStart, onExample, reduce }) {
  return (
    <main className="relative">
      <section className="relative mx-auto max-w-7xl px-5 pb-16 pt-14 sm:px-8 sm:pt-20">
        <div className="grid items-center gap-10 lg:grid-cols-[1.05fr_0.95fr]">
          {/* left: headline */}
          <div className="relative z-10">
            <h1
              className="text-[clamp(2.75rem,7vw,5.25rem)] font-semibold leading-[0.95] tracking-tight"
              style={{ fontFamily: DISPLAY, color: C.white }}
            >
              Better prompts,
              <br />
              instantly.
            </h1>
            <p
              className="mt-6 max-w-md text-[15px] leading-relaxed sm:text-base"
              style={{ color: C.mute }}
            >
              Describe what you want the AI to do. Hone AI rewrites it into a
              clear, structured prompt that gets better results — streamed to
              you as it's written.
            </p>

            <div className="mt-8 flex flex-wrap items-center gap-3">
              <button
                onClick={onStart}
                className="group flex items-center gap-2 rounded-full px-5 py-2.5 text-sm font-medium transition-transform active:scale-95"
                style={{ background: C.white, color: "#000" }}
              >
                Optimize a prompt
                <ArrowRight
                  size={16}
                  className="transition-transform group-hover:translate-x-0.5"
                />
              </button>
              <button
                onClick={() => onExample(EXAMPLE_GOAL)}
                className="rounded-full px-5 py-2.5 text-sm font-medium transition-colors"
                style={{
                  border: `1px solid ${C.lineStrong}`,
                  color: C.text,
                }}
                onMouseEnter={(e) =>
                  (e.currentTarget.style.background = "rgba(255,255,255,0.04)")
                }
                onMouseLeave={(e) =>
                  (e.currentTarget.style.background = "transparent")
                }
              >
                See an example
              </button>
            </div>
          </div>

          {/* right: glowing prism + short value lines */}
          <div className="relative flex flex-col items-center justify-center">
            <PrismArt reduce={reduce} />
            <ul
              className="mt-1 flex flex-col items-center gap-1.5 text-center text-sm"
              style={{ color: C.mute, fontFamily: MONO }}
            >
              <li>Tuned for every model</li>
              <li>Structured, not guessed</li>
              <li>Streamed token by token</li>
            </ul>
          </div>
        </div>
      </section>

      <ModelStrip />
      <LiveDemo reduce={reduce} />
      <HowItWorks />
      <Features />
      <FAQ />
      <CtaBand onStart={onStart} />
    </main>
  );
}

function PrismArt({ reduce }) {
  return (
    <div className="relative h-[300px] w-full max-w-md sm:h-[380px]">
      {/* radial glow */}
      <div
        className="absolute left-1/2 top-1/2 h-[380px] w-[380px] -translate-x-1/2 -translate-y-1/2 rounded-full"
        style={{
          background:
            "radial-gradient(circle, rgba(255,255,255,0.55) 0%, rgba(255,255,255,0.10) 32%, rgba(255,255,255,0) 62%)",
          filter: "blur(4px)",
          animation: reduce ? "none" : "breathe 5s ease-in-out infinite",
        }}
      />
      <svg
        viewBox="0 0 200 200"
        className="absolute left-1/2 top-1/2 h-[210px] w-[210px] -translate-x-1/2 -translate-y-1/2"
      >
        <defs>
          <linearGradient id="edge" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#ffffff" />
            <stop offset="100%" stopColor="#9a9a9a" />
          </linearGradient>
        </defs>
        {/* black triangle with a bright edge — echoes the reference mark */}
        <polygon
          points="100,34 168,158 32,158"
          fill="#050505"
          stroke="url(#edge)"
          strokeWidth="1.4"
          strokeLinejoin="round"
        />
      </svg>
    </div>
  );
}

function ModelStrip() {
  const models = ["GPT-4o", "Claude", "Gemini", "Llama", "Mistral", "Grok"];
  return (
    <div style={{ borderTop: `1px solid ${C.line}` }}>
      <div className="mx-auto max-w-7xl px-5 py-8 sm:px-8">
        <p
          className="mb-5 text-center text-xs uppercase tracking-[0.2em]"
          style={{ color: C.dim }}
        >
          Optimized output for every major model
        </p>
        <div className="flex flex-wrap items-center justify-center gap-x-10 gap-y-4">
          {models.map((m) => (
            <span
              key={m}
              className="text-sm font-medium tracking-tight"
              style={{ color: C.mute, fontFamily: DISPLAY }}
            >
              {m}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ----------------------- section scaffolding ----------------------- */
function Section({ children, className = "", bordered = true }) {
  return (
    <section
      className={`relative ${className}`}
      style={bordered ? { borderTop: `1px solid ${C.line}` } : undefined}
    >
      <div className="mx-auto max-w-7xl px-5 py-16 sm:px-8 sm:py-20">
        {children}
      </div>
    </section>
  );
}

function Eyebrow({ children }) {
  return (
    <p
      className="mb-3 text-xs uppercase tracking-[0.22em]"
      style={{ color: C.dim }}
    >
      {children}
    </p>
  );
}

/* --------------- live before → after streaming demo --------------- */
const DEMO_SAMPLES = [
  "make me an email for a client who hasn't paid",
  "help me write a python function to dedupe a list",
  "explain vector databases to a beginner",
];

function LiveDemo({ reduce }) {
  const [before, setBefore] = useState(DEMO_SAMPLES[0]);
  const [out, setOut] = useState("");
  const [running, setRunning] = useState(false);
  const [played, setPlayed] = useState(false);
  const timers = useRef([]);
  const clear = () => {
    timers.current.forEach(clearTimeout);
    timers.current.forEach(clearInterval);
    timers.current = [];
  };
  useEffect(() => () => clear(), []);

  const play = (text) => {
    clear();
    setRunning(true);
    setPlayed(true);
    setOut("");
    const full = buildPrompt(text, {
      model: "GPT-4o",
      tone: "Direct",
      format: "Markdown",
    });
    const toks = tokenize(full);
    let i = 0;
    const iv = setInterval(
      () => {
        const take = reduce ? 8 : 2 + Math.floor(Math.random() * 3);
        setOut((p) => p + toks.slice(i, i + take).join(""));
        i += take;
        if (i >= toks.length) {
          clearInterval(iv);
          setRunning(false);
        }
      },
      reduce ? 8 : 22
    );
    timers.current.push(iv);
  };

  const pick = (s) => {
    setBefore(s);
    play(s);
  };

  return (
    <Section>
      <div className="mx-auto max-w-2xl text-center">
        <Eyebrow>Live demo</Eyebrow>
        <h2
          className="text-3xl font-semibold tracking-tight sm:text-4xl"
          style={{ fontFamily: DISPLAY, color: C.white }}
        >
          Watch a rough idea become a real prompt
        </h2>
        <p className="mx-auto mt-4 max-w-lg text-[15px]" style={{ color: C.mute }}>
          Pick a messy prompt below. Hone AI adds the role, context,
          constraints, and output format — and streams it back live.
        </p>
      </div>

      <div className="mt-8 flex flex-wrap justify-center gap-2">
        {DEMO_SAMPLES.map((s) => (
          <button
            key={s}
            onClick={() => pick(s)}
            className="rounded-full px-3.5 py-1.5 text-sm transition-colors"
            style={{
              border: `1px solid ${before === s ? C.lineStrong : C.line}`,
              background: before === s ? "rgba(255,255,255,0.06)" : "transparent",
              color: before === s ? C.text : C.mute,
            }}
          >
            {s.length > 42 ? s.slice(0, 40) + "…" : s}
          </button>
        ))}
      </div>

      <div className="mt-6 grid gap-4 lg:grid-cols-[0.85fr_1.15fr]">
        {/* before */}
        <div
          className="rounded-2xl p-5"
          style={{ background: C.panel, border: `1px solid ${C.line}` }}
        >
          <div className="mb-3 flex items-center gap-2 text-xs" style={{ color: C.dim }}>
            <span className="h-1.5 w-1.5 rounded-full" style={{ background: C.dim }} />
            Your rough prompt
          </div>
          <p className="text-[15px] leading-relaxed" style={{ color: C.mute }}>
            {before}
          </p>
          <button
            onClick={() => play(before)}
            disabled={running}
            className="mt-5 flex items-center gap-2 rounded-xl px-4 py-2 text-sm font-semibold transition-transform active:scale-[0.98] disabled:opacity-60"
            style={{ background: C.white, color: "#000" }}
          >
            {running ? (
              <>
                <Loader2 size={15} className="animate-spin" />
                Optimizing…
              </>
            ) : (
              <>
                <Sparkles size={15} />
                Optimize this
              </>
            )}
          </button>
        </div>

        {/* after */}
        <div
          className="rounded-2xl"
          style={{ background: C.panel, border: `1px solid ${C.line}` }}
        >
          <div
            className="flex items-center gap-2 px-5 py-3 text-xs"
            style={{ borderBottom: `1px solid ${C.line}`, color: C.mute }}
          >
            <span className="h-1.5 w-1.5 rounded-full" style={{ background: C.white }} />
            Honed prompt
          </div>
          <div className="p-5">
            {!played ? (
              <div
                className="flex h-[220px] items-center justify-center text-center text-sm"
                style={{ color: C.dim }}
              >
                Hit “Optimize this” to see it stream.
              </div>
            ) : (
              <pre
                className="min-h-[220px] whitespace-pre-wrap break-words text-[12.5px] leading-relaxed"
                style={{ color: C.text, fontFamily: MONO }}
              >
                {out}
                {running && (
                  <span
                    className="ml-0.5 inline-block h-3.5 w-2 translate-y-0.5"
                    style={{
                      background: C.white,
                      animation: reduce ? "none" : "blink 1s step-end infinite",
                    }}
                  />
                )}
              </pre>
            )}
          </div>
        </div>
      </div>
    </Section>
  );
}

/* ------------------------- how it works ------------------------- */
function HowItWorks() {
  const steps = [
    {
      n: "01",
      t: "Understand",
      d: "Reads your goal and target model to work out what you actually want.",
    },
    {
      n: "02",
      t: "Draft",
      d: "Writes a first structured prompt with a clear role, task, and format.",
    },
    {
      n: "03",
      t: "Critique",
      d: "Checks it against a quality bar — vague wording, missing constraints, ambiguity.",
    },
    {
      n: "04",
      t: "Refine",
      d: "Rewrites the weak parts and streams the final prompt back to you.",
    },
  ];
  return (
    <Section>
      <div className="max-w-2xl">
        <Eyebrow>How it works</Eyebrow>
        <h2
          className="text-3xl font-semibold tracking-tight sm:text-4xl"
          style={{ fontFamily: DISPLAY, color: C.white }}
        >
          A four-step pipeline, not a single guess
        </h2>
        <p className="mt-4 text-[15px]" style={{ color: C.mute }}>
          Every prompt runs through the same loop a careful engineer would —
          draft, critique, refine — so you don't have to.
        </p>
      </div>

      <div className="mt-12 grid gap-px overflow-hidden rounded-2xl sm:grid-cols-2 lg:grid-cols-4"
        style={{ background: C.line }}>
        {steps.map((s) => (
          <div
            key={s.n}
            className="p-6"
            style={{ background: C.panel }}
          >
            <div
              className="text-sm font-medium tabular-nums"
              style={{ color: C.dim, fontFamily: MONO }}
            >
              {s.n}
            </div>
            <div
              className="mt-4 text-lg font-semibold"
              style={{ color: C.white, fontFamily: DISPLAY }}
            >
              {s.t}
            </div>
            <p className="mt-2 text-sm leading-relaxed" style={{ color: C.mute }}>
              {s.d}
            </p>
          </div>
        ))}
      </div>
    </Section>
  );
}

/* --------------------------- features --------------------------- */
function Features() {
  const feats = [
    {
      icon: Layers,
      t: "Structured, not guessed",
      d: "Every prompt gets a role, task, context, constraints, and an explicit output format.",
    },
    {
      icon: Cpu,
      t: "Tuned per model",
      d: "Output shaped for GPT-4o, Claude, Gemini, Llama and more — pick your target.",
    },
    {
      icon: Zap,
      t: "Streamed live",
      d: "See the prompt build token by token over SSE, the moment it's generated.",
    },
    {
      icon: GitBranch,
      t: "Versioned & rated",
      d: "Save versions, rate what works, and come back to your best prompts anytime.",
    },
    {
      icon: Search,
      t: "Semantic history",
      d: "Find past prompts by meaning, not exact words — powered by embeddings.",
    },
    {
      icon: ShieldCheck,
      t: "Private by default",
      d: "Your account, your prompts. Secure sign-in, nothing shared without you.",
    },
  ];
  return (
    <Section>
      <div className="max-w-2xl">
        <Eyebrow>What you get</Eyebrow>
        <h2
          className="text-3xl font-semibold tracking-tight sm:text-4xl"
          style={{ fontFamily: DISPLAY, color: C.white }}
        >
          Everything a good prompt needs, built in
        </h2>
      </div>

      <div className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {feats.map((f) => (
          <FeatureCard key={f.t} {...f} />
        ))}
      </div>
    </Section>
  );
}

function FeatureCard({ icon: Icon, t, d }) {
  return (
    <div
      className="group rounded-2xl p-6 transition-colors"
      style={{ background: C.panel, border: `1px solid ${C.line}` }}
      onMouseEnter={(e) => (e.currentTarget.style.borderColor = C.lineStrong)}
      onMouseLeave={(e) => (e.currentTarget.style.borderColor = C.line)}
    >
      <div
        className="flex h-10 w-10 items-center justify-center rounded-xl"
        style={{ border: `1px solid ${C.line}`, color: C.white }}
      >
        <Icon size={18} />
      </div>
      <h3
        className="mt-4 text-base font-semibold"
        style={{ color: C.white, fontFamily: DISPLAY }}
      >
        {t}
      </h3>
      <p className="mt-2 text-sm leading-relaxed" style={{ color: C.mute }}>
        {d}
      </p>
    </div>
  );
}

/* ------------------------------ faq ------------------------------ */
function FAQ() {
  const items = [
    {
      q: "What does Hone AI actually do?",
      a: "You describe what you want the AI to do in plain words. Hone AI rewrites that into a structured, production-grade prompt — with a role, task, context, constraints, and output format — and streams it back to you.",
    },
    {
      q: "Which models does it support?",
      a: "The prompt is tuned for whichever model you pick — GPT-4o, Claude, Gemini, Llama, Mistral and more. You choose the target under Options before optimizing.",
    },
    {
      q: "Do I need an account to try it?",
      a: "No. You can optimize prompts as a guest. Signing up lets you save, version, and rate your prompts so you can reuse the ones that work.",
    },
    {
      q: "Is my data private?",
      a: "Yes. Your prompts live under your account with secure sign-in, and nothing is shared without you.",
    },
    {
      q: "Can I use it from my own code?",
      a: "Yes — Hone AI exposes a REST API with streaming, so you can drop prompt optimization into your own app or pipeline.",
    },
  ];
  const [open, setOpen] = useState(0);
  return (
    <Section>
      <div className="grid gap-10 lg:grid-cols-[0.7fr_1.3fr]">
        <div>
          <Eyebrow>FAQ</Eyebrow>
          <h2
            className="text-3xl font-semibold tracking-tight sm:text-4xl"
            style={{ fontFamily: DISPLAY, color: C.white }}
          >
            Questions, answered
          </h2>
        </div>
        <div className="space-y-2">
          {items.map((it, i) => {
            const isOpen = open === i;
            return (
              <div
                key={i}
                className="rounded-xl"
                style={{ background: C.panel, border: `1px solid ${C.line}` }}
              >
                <button
                  onClick={() => setOpen(isOpen ? -1 : i)}
                  className="flex w-full items-center justify-between gap-4 px-5 py-4 text-left"
                >
                  <span
                    className="text-[15px] font-medium"
                    style={{ color: C.white }}
                  >
                    {it.q}
                  </span>
                  <span style={{ color: C.mute }} className="shrink-0">
                    {isOpen ? <Minus size={16} /> : <Plus size={16} />}
                  </span>
                </button>
                {isOpen && (
                  <p
                    className="px-5 pb-4 text-sm leading-relaxed"
                    style={{ color: C.mute }}
                  >
                    {it.a}
                  </p>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </Section>
  );
}

/* ---------------------------- cta band ---------------------------- */
function CtaBand({ onStart }) {
  return (
    <Section>
      <div className="relative overflow-hidden rounded-3xl px-6 py-16 text-center sm:py-20"
        style={{ background: C.panel, border: `1px solid ${C.line}` }}>
        <div
          className="pointer-events-none absolute left-1/2 top-0 h-64 w-64 -translate-x-1/2 -translate-y-1/2 rounded-full"
          style={{
            background:
              "radial-gradient(circle, rgba(255,255,255,0.22) 0%, rgba(255,255,255,0) 70%)",
          }}
        />
        <h2
          className="relative text-3xl font-semibold tracking-tight sm:text-5xl"
          style={{ fontFamily: DISPLAY, color: C.white }}
        >
          Stop wrestling with prompts.
        </h2>
        <p className="relative mx-auto mt-4 max-w-md text-[15px]" style={{ color: C.mute }}>
          Describe the goal once. Let Hone AI do the structuring — and get a
          better result every time.
        </p>
        <button
          onClick={onStart}
          className="group relative mt-8 inline-flex items-center gap-2 rounded-full px-6 py-3 text-sm font-semibold transition-transform active:scale-95"
          style={{ background: C.white, color: "#000" }}
        >
          Optimize a prompt
          <ArrowRight
            size={16}
            className="transition-transform group-hover:translate-x-0.5"
          />
        </button>
      </div>
    </Section>
  );
}

/* --------------------------- workspace --------------------------- */
function Workspace({ user, token, history, setHistory, requireAuth, reduce, exampleReq }) {
  const [goal, setGoal] = useState("");
  const [showOpts, setShowOpts] = useState(false);
  const [opts, setOpts] = useState({
    model: "GPT-4o",
    tone: "Direct",
    format: "Markdown",
  });

  const [phase, setPhase] = useState("idle"); // idle | running | done
  const [step, setStep] = useState(0); // 0..2 during run
  const [out, setOut] = useState("");
  const [copied, setCopied] = useState(false);
  const [rating, setRating] = useState(0);
  const [tab, setTab] = useState("optimize"); // optimize | history
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);

  const abortRef = useRef(null);
  useEffect(() => () => abortRef.current?.abort(), []);

  // "See an example" from the landing page: fill the input and run it.
  useEffect(() => {
    if (!exampleReq) return;
    setTab("optimize");
    setGoal(exampleReq.goal);
    const t = setTimeout(() => run(exampleReq.goal), reduce ? 60 : 280);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [exampleReq && exampleReq.nonce]);

  const tokensOut = useMemo(() => (out ? est(out) : 0), [out]);

  const run = async (overrideGoal) => {
    if (phase === "running") return;
    const g = typeof overrideGoal === "string" ? overrideGoal : goal;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setError("");
    setCopied(false);
    setRating(0);
    setOut("");
    setPhase("running");
    setStep(0);

    try {
      await api.optimize({
        goal: g,
        opts,
        token,
        signal: controller.signal,
        onStage: (index) => setStep(index),
        onToken: (text) => setOut((p) => p + text),
      });
      setPhase("done");
    } catch (e) {
      if (e.name === "AbortError") return;
      setError("Couldn't reach Hone AI. Check the backend is running and try again.");
      setPhase("idle");
    }
  };

  const copy = () => {
    const ta = document.createElement("textarea");
    ta.value = out;
    document.body.appendChild(ta);
    ta.select();
    try {
      document.execCommand("copy");
    } catch (e) {}
    document.body.removeChild(ta);
    setCopied(true);
    setTimeout(() => setCopied(false), 1600);
  };

  const save = async () => {
    if (!user) return requireAuth();
    if (!out || saving) return;
    setSaving(true);
    setError("");
    try {
      const row = await api.savePrompt(token, {
        goal: goal.trim() || "Untitled prompt",
        model: opts.model,
        tone: opts.tone,
        format: opts.format,
        prompt: out,
        rating: rating >= 4 ? 1 : rating > 0 && rating < 3 ? -1 : 0,
        meta: {},
      });
      setHistory((h) => [
        {
          id: row.id,
          goal: row.goal,
          model: row.model,
          rating: row.rating,
          at: new Date(row.created_at).getTime(),
        },
        ...h,
      ]);
    } catch (e) {
      setError("Couldn't save that prompt. Try again in a moment.");
    } finally {
      setSaving(false);
    }
  };

  const onKey = (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") run();
  };

  return (
    <main className="mx-auto max-w-6xl px-5 py-8 sm:px-8 sm:py-10">
      {/* view tabs */}
      <div className="mb-6 flex items-center gap-1">
        {["optimize", "history"].map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className="rounded-full px-4 py-1.5 text-sm font-medium capitalize transition-colors"
            style={{
              color: tab === t ? "#000" : C.mute,
              background: tab === t ? C.white : "transparent",
            }}
          >
            {t}
          </button>
        ))}
      </div>

      {tab === "history" ? (
        <History
          history={history}
          setHistory={setHistory}
          token={token}
          user={user}
          requireAuth={requireAuth}
        />
      ) : (
        <div className="grid gap-5 lg:grid-cols-2">
          {/* ---- input ---- */}
          <div
            className="rounded-2xl p-5 sm:p-6"
            style={{ background: C.panel, border: `1px solid ${C.line}` }}
          >
            <label
              className="mb-1.5 block text-sm font-medium"
              style={{ color: C.text }}
            >
              Write down anything that comes to mind
            </label>
            <p className="mb-3 text-[13px] leading-relaxed" style={{ color: C.dim }}>
              It doesn't have to be perfect — Hone AI will optimize it into a
              prompt built to get the best results from any model out there.
            </p>
            <textarea
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              onKeyDown={onKey}
              rows={6}
              placeholder="e.g. Write a friendly reminder email to a client who missed a payment"
              className="w-full resize-none rounded-xl p-4 text-[15px] leading-relaxed outline-none transition-colors"
              style={{
                background: C.bg,
                border: `1px solid ${C.line}`,
                color: C.text,
                fontFamily: SANS,
              }}
              onFocus={(e) =>
                (e.currentTarget.style.borderColor = C.lineStrong)
              }
              onBlur={(e) => (e.currentTarget.style.borderColor = C.line)}
            />

            {/* optional, tucked-away controls */}
            <button
              onClick={() => setShowOpts((s) => !s)}
              className="mt-3 flex items-center gap-1.5 text-sm transition-colors"
              style={{ color: C.dim }}
              onMouseEnter={(e) => (e.currentTarget.style.color = C.mute)}
              onMouseLeave={(e) => (e.currentTarget.style.color = C.dim)}
            >
              <Sliders size={14} />
              Options
              <ChevronDown
                size={14}
                className="transition-transform"
                style={{ transform: showOpts ? "rotate(180deg)" : "none" }}
              />
            </button>

            {showOpts && (
              <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-3">
                <Select
                  label="Model"
                  value={opts.model}
                  onChange={(v) => setOpts((o) => ({ ...o, model: v }))}
                  options={["GPT-4o", "Claude", "Gemini", "Llama", "Mistral"]}
                />
                <Select
                  label="Tone"
                  value={opts.tone}
                  onChange={(v) => setOpts((o) => ({ ...o, tone: v }))}
                  options={["Direct", "Friendly", "Formal", "Playful"]}
                />
                <Select
                  label="Format"
                  value={opts.format}
                  onChange={(v) => setOpts((o) => ({ ...o, format: v }))}
                  options={["Markdown", "Steps", "JSON", "Table"]}
                />
              </div>
            )}

            <button
              onClick={run}
              disabled={phase === "running"}
              className="group mt-5 flex w-full items-center justify-center gap-2 rounded-xl py-3 text-sm font-semibold transition-transform active:scale-[0.99] disabled:opacity-60"
              style={{ background: C.white, color: "#000" }}
            >
              {phase === "running" ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  Optimizing…
                </>
              ) : (
                <>
                  Optimize
                  <span
                    className="ml-1 hidden items-center gap-0.5 rounded px-1.5 py-0.5 text-[11px] sm:flex"
                    style={{ background: "rgba(0,0,0,0.12)" }}
                  >
                    ⌘<CornerDownLeft size={11} />
                  </span>
                </>
              )}
            </button>

            {error && (
              <p className="mt-3 text-sm" style={{ color: "#f87171" }}>
                {error}
              </p>
            )}
          </div>

          {/* ---- output ---- */}
          <div
            className="flex flex-col rounded-2xl"
            style={{ background: C.panel, border: `1px solid ${C.line}` }}
          >
            <div
              className="flex items-center justify-between px-5 py-3.5"
              style={{ borderBottom: `1px solid ${C.line}` }}
            >
              <StepBar phase={phase} step={step} />
              {phase !== "idle" && (
                <span
                  className="text-xs tabular-nums"
                  style={{ color: C.dim, fontFamily: MONO }}
                >
                  ≈{tokensOut} tokens
                </span>
              )}
            </div>

            <div className="relative flex-1 p-5">
              {phase === "idle" ? (
                <div
                  className="flex h-full min-h-[240px] items-center justify-center text-center text-sm"
                  style={{ color: C.dim }}
                >
                  Your optimized prompt will appear here.
                </div>
              ) : (
                <pre
                  className="min-h-[240px] whitespace-pre-wrap break-words text-[13px] leading-relaxed"
                  style={{ color: C.text, fontFamily: MONO }}
                >
                  {out}
                  {phase === "running" && (
                    <span
                      className="ml-0.5 inline-block h-4 w-2 translate-y-0.5"
                      style={{
                        background: C.white,
                        animation: reduce ? "none" : "blink 1s step-end infinite",
                      }}
                    />
                  )}
                </pre>
              )}
            </div>

            {phase === "done" && (
              <div
                className="flex flex-wrap items-center gap-2 px-5 py-3.5"
                style={{ borderTop: `1px solid ${C.line}` }}
              >
                <GhostBtn onClick={copy}>
                  {copied ? <Check size={14} /> : <Copy size={14} />}
                  {copied ? "Copied" : "Copy"}
                </GhostBtn>
                <GhostBtn onClick={run}>
                  <RotateCw size={14} />
                  Regenerate
                </GhostBtn>
                <GhostBtn onClick={save} disabled={saving}>
                  {saving ? "Saving…" : "Save"}
                </GhostBtn>

                <div className="ml-auto flex items-center gap-1">
                  <RateBtn
                    active={rating >= 4}
                    onClick={() => setRating(rating >= 4 ? 0 : 5)}
                  >
                    <ThumbsUp size={14} />
                  </RateBtn>
                  <RateBtn
                    active={rating > 0 && rating < 3}
                    onClick={() => setRating(rating > 0 && rating < 3 ? 0 : 2)}
                  >
                    <ThumbsDown size={14} />
                  </RateBtn>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </main>
  );
}

function StepBar({ phase, step }) {
  const steps = ["Understanding", "Drafting", "Refining"];
  const activeIdx = phase === "done" ? 3 : phase === "running" ? step : -1;
  return (
    <div className="flex items-center gap-2">
      {steps.map((s, i) => {
        const done = activeIdx > i || phase === "done";
        const active = activeIdx === i && phase === "running";
        return (
          <div key={s} className="flex items-center gap-2">
            <div className="flex items-center gap-1.5">
              <span
                className="h-1.5 w-1.5 rounded-full transition-colors"
                style={{
                  background: done ? C.white : active ? C.mute : C.line,
                  animation: active ? "pulseDot 1s ease-in-out infinite" : "none",
                }}
              />
              <span
                className="text-xs"
                style={{ color: done || active ? C.text : C.dim }}
              >
                {s}
              </span>
            </div>
            {i < steps.length - 1 && (
              <span style={{ color: C.line }} className="text-xs">
                ·
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}

function Select({ label, value, onChange, options }) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-xs" style={{ color: C.dim }}>
        {label}
      </span>
      <div className="relative">
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-full appearance-none rounded-lg px-3 py-2 text-sm outline-none"
          style={{
            background: C.bg,
            border: `1px solid ${C.line}`,
            color: C.text,
          }}
        >
          {options.map((o) => (
            <option key={o} value={o} style={{ background: "#111" }}>
              {o}
            </option>
          ))}
        </select>
        <ChevronDown
          size={14}
          className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2"
          style={{ color: C.dim }}
        />
      </div>
    </label>
  );
}

function GhostBtn({ children, onClick, disabled }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm transition-colors disabled:opacity-50"
      style={{ border: `1px solid ${C.line}`, color: C.text }}
      onMouseEnter={(e) =>
        (e.currentTarget.style.background = "rgba(255,255,255,0.05)")
      }
      onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
    >
      {children}
    </button>
  );
}

function RateBtn({ children, active, onClick }) {
  return (
    <button
      onClick={onClick}
      className="flex h-8 w-8 items-center justify-center rounded-lg transition-colors"
      style={{
        border: `1px solid ${active ? C.lineStrong : C.line}`,
        background: active ? "rgba(255,255,255,0.08)" : "transparent",
        color: active ? C.white : C.mute,
      }}
    >
      {children}
    </button>
  );
}

/* ----------------------------- history ----------------------------- */
function History({ history, setHistory, token, user, requireAuth }) {
  if (!user) {
    return (
      <div
        className="rounded-2xl p-12 text-center"
        style={{ background: C.panel, border: `1px solid ${C.line}` }}
      >
        <p style={{ color: C.mute }}>Log in to see your history.</p>
        <button
          onClick={requireAuth}
          className="mt-4 rounded-full px-4 py-1.5 text-sm font-medium transition-transform active:scale-95"
          style={{ background: C.white, color: "#000" }}
        >
          Log in
        </button>
      </div>
    );
  }
  if (!history.length) {
    return (
      <div
        className="rounded-2xl p-12 text-center"
        style={{ background: C.panel, border: `1px solid ${C.line}` }}
      >
        <p style={{ color: C.mute }}>
          Nothing saved yet. Optimize a prompt and hit Save to keep it here.
        </p>
      </div>
    );
  }
  return (
    <div className="space-y-2.5">
      {history.map((h) => (
        <div
          key={h.id}
          className="group flex items-center gap-4 rounded-xl px-4 py-3.5 transition-colors"
          style={{ background: C.panel, border: `1px solid ${C.line}` }}
          onMouseEnter={(e) =>
            (e.currentTarget.style.borderColor = C.lineStrong)
          }
          onMouseLeave={(e) => (e.currentTarget.style.borderColor = C.line)}
        >
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm" style={{ color: C.text }}>
              {h.goal}
            </p>
            <div
              className="mt-1 flex items-center gap-3 text-xs"
              style={{ color: C.dim }}
            >
              <span style={{ fontFamily: MONO }}>{h.model}</span>
              <span className="flex items-center gap-1">
                <Clock size={11} />
                {ago(h.at)}
              </span>
              {h.rating === 1 && (
                <ThumbsUp size={12} style={{ color: C.mute }} />
              )}
              {h.rating === -1 && (
                <ThumbsDown size={12} style={{ color: C.mute }} />
              )}
            </div>
          </div>
          <button
            onClick={() => {
              setHistory((list) => list.filter((x) => x.id !== h.id));
              if (token) api.deletePrompt(token, h.id).catch(() => {});
            }}
            className="opacity-0 transition-opacity group-hover:opacity-100"
            style={{ color: C.dim }}
            onMouseEnter={(e) => (e.currentTarget.style.color = C.text)}
            onMouseLeave={(e) => (e.currentTarget.style.color = C.dim)}
            aria-label="Delete"
          >
            <Trash2 size={15} />
          </button>
        </div>
      ))}
    </div>
  );
}

function ago(t) {
  const m = Math.floor((Date.now() - t) / 60000);
  if (m < 1) return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

/* ------------------------------ auth ------------------------------ */
function AuthModal({ mode, switchMode, onClose, onDone }) {
  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  const [pw, setPw] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const signup = mode === "signup";

  const submit = async () => {
    if (busy) return;
    setError("");
    setBusy(true);
    try {
      const body = signup ? { name, email, password: pw } : { email, password: pw };
      const data = await api.auth(signup ? "signup" : "login", body);
      onDone(data);
    } catch (e) {
      setError(
        signup
          ? "Couldn't create that account — the email may already be registered."
          : "Invalid email or password."
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: "rgba(0,0,0,0.7)", backdropFilter: "blur(4px)" }}
      onClick={onClose}
    >
      <div
        className="w-full max-w-sm rounded-2xl p-6 sm:p-7"
        style={{ background: C.panel2, border: `1px solid ${C.lineStrong}` }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-6 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Mark size={20} />
            <span
              className="font-semibold"
              style={{ fontFamily: DISPLAY, color: C.white }}
            >
              Hone AI
            </span>
          </div>
          <button onClick={onClose} style={{ color: C.dim }} aria-label="Close">
            <X size={18} />
          </button>
        </div>

        <h2
          className="text-xl font-semibold tracking-tight"
          style={{ fontFamily: DISPLAY, color: C.white }}
        >
          {signup ? "Create your account" : "Welcome back"}
        </h2>
        <p className="mt-1.5 text-sm" style={{ color: C.mute }}>
          {signup
            ? "Save and version every prompt you optimize."
            : "Sign in to pick up where you left off."}
        </p>

        <div className="space-y-3">
          {signup && (
            <Field label="Name" value={name} onChange={setName} placeholder="Ada Lovelace" />
          )}
          <Field
            label="Email"
            value={email}
            onChange={setEmail}
            placeholder="you@work.com"
            type="email"
          />
          <Field
            label="Password"
            value={pw}
            onChange={setPw}
            placeholder="••••••••"
            type="password"
          />
        </div>

        {error && (
          <p className="mt-3 text-sm" style={{ color: "#f87171" }}>
            {error}
          </p>
        )}

        <button
          onClick={submit}
          disabled={busy}
          className="mt-5 w-full rounded-xl py-2.5 text-sm font-semibold transition-transform active:scale-[0.99] disabled:opacity-60"
          style={{ border: `1px solid ${C.lineStrong}`, color: C.text }}
        >
          {busy ? "Please wait…" : signup ? "Create account" : "Log in"}
        </button>

        <p className="mt-5 text-center text-sm" style={{ color: C.mute }}>
          {signup ? "Already have an account?" : "New to Hone AI?"}{" "}
          <button
            onClick={() => switchMode(signup ? "login" : "signup")}
            className="font-medium underline-offset-2 hover:underline"
            style={{ color: C.white }}
          >
            {signup ? "Log in" : "Sign up"}
          </button>
        </p>
      </div>
    </div>
  );
}

function Field({ label, value, onChange, placeholder, type = "text" }) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-xs" style={{ color: C.mute }}>
        {label}
      </span>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full rounded-lg px-3 py-2.5 text-sm outline-none transition-colors"
        style={{ background: C.bg, border: `1px solid ${C.line}`, color: C.text }}
        onFocus={(e) => (e.currentTarget.style.borderColor = C.lineStrong)}
        onBlur={(e) => (e.currentTarget.style.borderColor = C.line)}
      />
    </label>
  );
}

/* ----------------------------- footer ----------------------------- */
function Footer() {
  return (
    <footer style={{ borderTop: `1px solid ${C.line}` }} className="mt-8">
      <div className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-4 px-5 py-8 sm:flex-row sm:px-8">
        <div className="flex items-center gap-2">
          <Mark size={18} />
          <span className="text-sm" style={{ color: C.mute }}>
            Hone AI — the prompt optimization layer
          </span>
        </div>
        <div className="flex items-center gap-6 text-sm" style={{ color: C.dim }}>
          <span>Privacy</span>
          <span>© 2026</span>
        </div>
      </div>
    </footer>
  );
}

/* ------------------------------ bits ------------------------------ */
function Mark({ size = 22 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden>
      <polygon
        points="12,3 21,20 3,20"
        fill="none"
        stroke={C.white}
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function StyleTag() {
  return (
    <style>{`
      *{ -webkit-tap-highlight-color: transparent; }
      ::selection{ background: rgba(255,255,255,0.18); }
      *:focus-visible{ outline: 2px solid rgba(255,255,255,0.55); outline-offset: 2px; border-radius: 6px; }
      @keyframes breathe { 0%,100%{ opacity:.75; transform:translate(-50%,-50%) scale(1);} 50%{ opacity:1; transform:translate(-50%,-50%) scale(1.06);} }
      @keyframes blink { 50%{ opacity:0; } }
      @keyframes pulseDot { 0%,100%{ opacity:.4; } 50%{ opacity:1; } }
      textarea::placeholder, input::placeholder{ color:#4a4a4a; }
      select{ color-scheme: dark; }
    `}</style>
  );
}
