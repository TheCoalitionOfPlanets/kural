/* Inline SVG rather than an icon package: six glyphs is not worth a dependency,
 * and `currentColor` is what makes them work in both themes for free.
 */
const base = {
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.7,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
};

export const MicIcon = (p: { className?: string }) => (
  <svg {...base} className={p.className}>
    <rect x="9" y="2.5" width="6" height="11" rx="3" />
    <path d="M5.5 11a6.5 6.5 0 0 0 13 0" />
    <path d="M12 17.5V21" />
  </svg>
);

export const MicOffIcon = (p: { className?: string }) => (
  <svg {...base} className={p.className}>
    <path d="M9 5a3 3 0 0 1 6 0v5" />
    <path d="M15 13.2A3 3 0 0 1 9 12V9.8" />
    <path d="M5.5 11a6.5 6.5 0 0 0 10.2 5.3M18.5 11v.6" />
    <path d="M12 17.5V21" />
    <path d="M3.5 3.5l17 17" />
  </svg>
);

export const StopIcon = (p: { className?: string }) => (
  <svg {...base} className={p.className}>
    <rect x="6.5" y="6.5" width="11" height="11" rx="2.5" />
  </svg>
);

export const ListIcon = (p: { className?: string }) => (
  <svg {...base} className={p.className}>
    <path d="M4 7h16M4 12h16M4 17h10" />
  </svg>
);

export const SunIcon = (p: { className?: string }) => (
  <svg {...base} className={p.className}>
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
  </svg>
);

export const MoonIcon = (p: { className?: string }) => (
  <svg {...base} className={p.className}>
    <path d="M20 14.5A8.5 8.5 0 1 1 9.5 4a6.8 6.8 0 0 0 10.5 10.5Z" />
  </svg>
);
