/** @type {import('tailwindcss').Config} */
// ASU Learning Engineering Institute palette, applied with Apple's craft rules.
// See DESIGN_SYSTEM.md. Colour marks meaning; neutrals carry the interface.
//
// Two hard rules:
//   1. Never gold text on white (about 1.7:1, fails WCAG). Gold is a background
//      with black on top, or a rule. That is how the LEI site uses it.
//   2. Maroon is identity and primary action, not a large fill behind body text.
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        asu: {
          maroon: "#8C1D40",
          "maroon-dark": "#6B1531",
          "maroon-tint": "#F6EBEF",
          gold: "#FFC627",
          "gold-tint": "#FFF6DC",
          black: "#000000",
        },
        border: "#E8E8EA",
        input: "#E8E8EA",
        ring: "#8C1D40",
        background: "#FFFFFF",
        foreground: "#18181B",
        primary: { DEFAULT: "#8C1D40", foreground: "#FFFFFF" },
        secondary: { DEFAULT: "#F4F4F5", foreground: "#18181B" },
        muted: { DEFAULT: "#F4F4F5", foreground: "#71717A" },
        accent: { DEFAULT: "#FFC627", foreground: "#000000" },
        card: { DEFAULT: "#FFFFFF", foreground: "#18181B" },
      },
      borderRadius: {
        sm: "0.375rem",
        md: "0.625rem",
        lg: "0.875rem",
        xl: "1.125rem",
      },
      boxShadow: {
        // Bigger surfaces read as thicker: more blur, deeper shadow.
        chip: "0 1px 2px rgba(24,24,27,0.06)",
        card: "0 1px 2px rgba(24,24,27,0.05), 0 4px 16px -4px rgba(24,24,27,0.07)",
        float: "0 2px 6px rgba(24,24,27,0.07), 0 12px 32px -8px rgba(24,24,27,0.14)",
      },
      transitionTimingFunction: {
        // Spring approximations. Not interruptible; see DESIGN_SYSTEM.md.
        standard: "cubic-bezier(0.32, 0.72, 0, 1)",
        momentum: "cubic-bezier(0.34, 1.36, 0.64, 1)",
      },
    },
  },
  plugins: [],
};
