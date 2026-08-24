import type { Metadata, Viewport } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Kural — Voice",
  description:
    "Always-listening speech pipeline. Indian languages run on local models; " +
    "everything else is routed to ElevenLabs.",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#ffffff" },
    { media: "(prefers-color-scheme: dark)", color: "#000000" },
  ],
  width: "device-width",
  initialScale: 1,
  // The orb sits dead centre; letting it zoom on a double-tap during a call is
  // only ever an accident.
  maximumScale: 1,
};

/* The theme is read and applied before first paint. Doing it in an effect
 * instead means one frame of white on a black-themed page, which on a design
 * built out of pure black and pure white is a flash, not a nuance. */
const THEME_BOOT = `
(function () {
  try {
    var t = localStorage.getItem("kural-theme");
    if (t === "light" || t === "dark") {
      document.documentElement.setAttribute("data-theme", t);
    }
  } catch (e) {}
})();
`;

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_BOOT }} />
      </head>
      <body>{children}</body>
    </html>
  );
}
