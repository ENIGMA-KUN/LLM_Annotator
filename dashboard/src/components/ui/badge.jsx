import React from "react";
import { cva } from "class-variance-authority";
import { cn } from "@/lib/utils";

// Colour marks meaning, not decoration.
//   maroon / gold  = ASU brand, identity and emphasis
//   emerald/amber/rose = result quality, a semantic scale independent of brand
//
// Gold is a background with black text on it, never gold text on white
// (roughly 1.7:1 contrast, fails WCAG). See DESIGN_SYSTEM.md.
const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2.5 py-0.5 type-caption font-semibold",
  {
    variants: {
      variant: {
        default: "border-transparent bg-asu-maroon text-white",
        maroon: "border-asu-maroon/20 bg-asu-maroon-tint text-asu-maroon",
        gold: "border-transparent bg-asu-gold text-black",
        secondary: "border-transparent bg-gray-100 text-gray-700",
        outline: "border-border text-foreground",
        emerald: "border-emerald-200 bg-emerald-50 text-emerald-700",
        amber: "border-amber-200 bg-amber-50 text-amber-800",
        indigo: "border-indigo-200 bg-indigo-50 text-indigo-700",
        rose: "border-rose-200 bg-rose-50 text-rose-700",
        slate: "border-gray-200 bg-gray-100 text-gray-600",
      },
    },
    defaultVariants: { variant: "default" },
  }
);

function Badge({ className, variant, ...props }) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}

export { Badge, badgeVariants };
