/** @type {import('tailwindcss').Config} */
export default {
  // Dark mode is a class on <html>, toggled by useTheme + the inline script
  // in index.html (which also prevents a light-flash on load).
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        // All theme colors are defined as CSS variables (see src/index.css)
        // so a single `.dark` block re-skins the whole app: every component
        // keeps its semantic class (bg-paper-50, text-ink-600, …) and only
        // the underlying values flip.
        // Muted indigo: calm enough to read like a reference tool rather
        // than a marketing page.
        brand: {
          50: 'rgb(var(--c-brand-50) / <alpha-value>)',
          100: 'rgb(var(--c-brand-100) / <alpha-value>)',
          200: 'rgb(var(--c-brand-200) / <alpha-value>)',
          300: 'rgb(var(--c-brand-300) / <alpha-value>)',
          400: 'rgb(var(--c-brand-400) / <alpha-value>)',
          500: 'rgb(var(--c-brand-500) / <alpha-value>)',
          600: 'rgb(var(--c-brand-600) / <alpha-value>)',
          700: 'rgb(var(--c-brand-700) / <alpha-value>)',
          800: 'rgb(var(--c-brand-800) / <alpha-value>)',
          900: 'rgb(var(--c-brand-900) / <alpha-value>)',
        },
        // Warm neutrals: "ink on paper" instead of the default cool greys.
        // The scales flip roles between modes: paper-* is always the surface
        // ramp, ink-* is always the text ramp.
        ink: {
          50: 'rgb(var(--c-ink-50) / <alpha-value>)',
          100: 'rgb(var(--c-ink-100) / <alpha-value>)',
          200: 'rgb(var(--c-ink-200) / <alpha-value>)',
          300: 'rgb(var(--c-ink-300) / <alpha-value>)',
          400: 'rgb(var(--c-ink-400) / <alpha-value>)',
          500: 'rgb(var(--c-ink-500) / <alpha-value>)',
          600: 'rgb(var(--c-ink-600) / <alpha-value>)',
          700: 'rgb(var(--c-ink-700) / <alpha-value>)',
          800: 'rgb(var(--c-ink-800) / <alpha-value>)',
          900: 'rgb(var(--c-ink-900) / <alpha-value>)',
        },
        paper: {
          50: 'rgb(var(--c-paper-50) / <alpha-value>)',
          100: 'rgb(var(--c-paper-100) / <alpha-value>)',
          200: 'rgb(var(--c-paper-200) / <alpha-value>)',
          300: 'rgb(var(--c-paper-300) / <alpha-value>)',
        },
        // Reserved for citations. Warm ochre keeps source markers visually
        // distinct from interactive elements, which are indigo.
        mark: {
          50: 'rgb(var(--c-mark-50) / <alpha-value>)',
          100: 'rgb(var(--c-mark-100) / <alpha-value>)',
          500: 'rgb(var(--c-mark-500) / <alpha-value>)',
          600: 'rgb(var(--c-mark-600) / <alpha-value>)',
          700: 'rgb(var(--c-mark-700) / <alpha-value>)',
        },
        // Code blocks intentionally do NOT flip with the theme: code should
        // read as "terminal" in both modes.
        'code-bg': 'rgb(var(--c-code-bg) / <alpha-value>)',
        'code-text': 'rgb(var(--c-code-text) / <alpha-value>)',
      },
      fontFamily: {
        // Serif for headings only: it reads as "paper" and, with a Songti
        // fallback, gives Chinese titles the same register as an academic
        // journal. Body text stays sans for screen legibility.
        serif: [
          'ui-serif',
          'Georgia',
          'Cambria',
          '"Songti SC"',
          '"Source Han Serif SC"',
          '"Noto Serif SC"',
          'SimSun',
          'serif',
        ],
      },
      typography: {
        DEFAULT: {
          css: {
            maxWidth: 'none',
            // Inline code follows the ochre mark ramp so it stays a
            // "highlighter" in both themes.
            code: { color: 'rgb(var(--c-mark-600))' },
          },
        },
      },
    },
  },
  plugins: [],
}
