"use client"

import { motion } from "framer-motion"
import { useInView } from "framer-motion"
import { useRef } from "react"
import { Check, X, Target, Zap, Globe } from "lucide-react"

const competitors = [
  {
    name: "Siloed Giants (CRM, ERP)",
    weakness:
      "Master of one domain, create data silos, expensive integration afterthought",
  },
  {
    name: "iPaaS (Plumbing)",
    weakness:
      "Provide pipes but don&apos;t solve fragmented UI/UX, requires dozens of licenses",
  },
  {
    name: "Low-Code (LC/NC)",
    weakness: "Good for new apps, not designed to unify legacy systems at scale",
  },
  {
    name: '"All-In-One" Suites',
    weakness: "Jack of all trades, master of none, lack depth and scalability",
  },
]

const airviewxAdvantages = [
  "Zero touch automation",
  "Cross Functional workflows",
  "Physical to digital integration",
  "Single unified platform",
  "Agentic AI at the core",
]

const marketStats = [
  { value: "$47B+", label: "Market Size" },
  { value: "40%", label: "CAGR through 2030" },
  { value: "40-45%", label: "Global Demand Share" },
]

export function Competitive() {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: true, margin: "-100px" })

  return (
    <section className="relative py-24 lg:py-32 overflow-hidden">
      <div ref={ref} className="relative mx-auto max-w-7xl px-6 lg:px-8">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5 }}
          className="text-center mb-16"
        >
          <span className="inline-block text-xs uppercase tracking-wider text-primary font-medium mb-4">
            Competitive Landscape
          </span>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-balance">
            Why AirviewX Wins
          </h2>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-8 lg:gap-12">
          {/* Competitors Column */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={isInView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="space-y-4"
          >
            <h3 className="text-lg font-semibold text-muted-foreground mb-6 flex items-center gap-2">
              <X className="h-5 w-5 text-red-400" />
              Traditional Solutions
            </h3>
            {competitors.map((competitor, index) => (
              <div
                key={competitor.name}
                className="p-5 rounded-xl border border-border/50 bg-card/30 hover:bg-card/50 transition-colors"
              >
                <div className="flex items-start gap-3">
                  <div className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-red-400/10">
                    <X className="h-3.5 w-3.5 text-red-400" />
                  </div>
                  <div>
                    <h4 className="font-medium text-foreground">
                      {competitor.name}
                    </h4>
                    <p className="text-sm text-muted-foreground mt-1">
                      {competitor.weakness}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </motion.div>

          {/* AirviewX Column */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={isInView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <div className="glass rounded-2xl p-6 lg:p-8 glow h-full">
              <h3 className="text-lg font-semibold text-foreground mb-6 flex items-center gap-2">
                <Check className="h-5 w-5 text-emerald-400" />
                AirviewX Advantage
              </h3>

              <div className="space-y-4 mb-8">
                {airviewxAdvantages.map((advantage, index) => (
                  <div key={advantage} className="flex items-center gap-3">
                    <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-emerald-400/10">
                      <Check className="h-3.5 w-3.5 text-emerald-400" />
                    </div>
                    <span className="text-foreground">{advantage}</span>
                  </div>
                ))}
              </div>

              {/* Target Markets */}
              <div className="border-t border-border/50 pt-6 mb-6">
                <h4 className="text-sm font-medium text-muted-foreground mb-4">
                  Target Markets
                </h4>
                <div className="flex flex-wrap gap-2">
                  {[
                    "Telecom",
                    "Utilities",
                    "Construction",
                    "Public Sector",
                  ].map((market) => (
                    <span
                      key={market}
                      className="inline-flex items-center gap-1.5 rounded-full bg-primary/10 px-3 py-1 text-xs font-medium text-primary"
                    >
                      <Target className="h-3 w-3" />
                      {market}
                    </span>
                  ))}
                </div>
              </div>

              {/* Market Stats */}
              <div className="grid grid-cols-3 gap-4">
                {marketStats.map((stat) => (
                  <div key={stat.label} className="text-center">
                    <div className="text-2xl font-bold text-gradient">
                      {stat.value}
                    </div>
                    <div className="text-xs text-muted-foreground mt-1">
                      {stat.label}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  )
}
