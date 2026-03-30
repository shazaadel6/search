"use client"

import { motion } from "framer-motion"
import { useInView } from "framer-motion"
import { useRef } from "react"
import {
  Lightbulb,
  Building,
  Rocket,
  Crown,
  ArrowRight,
  Database,
  Bot,
  Target,
  Zap,
} from "lucide-react"

const stages = [
  {
    icon: Lightbulb,
    title: "Concept and R&D",
    status: "completed",
    items: [
      "Telecom, Utilities, Renewable energy, smart cities background",
      "Gaps of industry analysis",
      "UBOS vision development",
      "BPMN, RPA, AI/ML & BI core architecture",
    ],
  },
  {
    icon: Building,
    title: "Foundation Building",
    status: "completed",
    items: [
      "Decision points & data flows mapping",
      "Pilot orchestrations across consulting enterprise",
      "Real-time data infrastructure",
      "AI-ready workflow execution plans",
    ],
  },
  {
    icon: Rocket,
    title: "Soft Launch",
    status: "current",
    items: [
      "Build composable architecture",
      "Deploy Agentic workflows",
      "Human in loop controls",
      "Orchestration Layer deployment",
    ],
  },
  {
    icon: Crown,
    title: "Full Autonomy",
    status: "future",
    items: [
      "Deploy industry-specific packs",
      "Regulator Compliance integration",
      "Advanced governance & security",
      "End-to-end automation",
      "Strategic human oversight only",
      "Market leading differentiation",
    ],
  },
]

const pillars = [
  { icon: Database, label: "Data Infrastructure" },
  { icon: Bot, label: "AI Agent Capability" },
  { icon: Target, label: "Industry Specialization" },
  { icon: Zap, label: "Full Autonomy" },
]

export function Roadmap() {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: true, margin: "-100px" })

  return (
    <section id="roadmap" className="relative py-24 lg:py-32 overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 grid-background opacity-20" />

      <div ref={ref} className="relative mx-auto max-w-7xl px-6 lg:px-8">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5 }}
          className="text-center mb-16"
        >
          <span className="inline-block text-xs uppercase tracking-wider text-primary font-medium mb-4">
            Roadmap
          </span>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-balance mb-6">
            Progressive Journey to
            <span className="text-gradient"> Full Autonomy</span>
          </h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            First Movers Capture 60% Market Share
          </p>
        </motion.div>

        {/* Timeline */}
        <div className="relative mb-16">
          {/* Connection Line */}
          <div className="absolute left-8 lg:left-1/2 top-0 bottom-0 w-px bg-border lg:-translate-x-px" />

          <div className="space-y-12">
            {stages.map((stage, index) => (
              <motion.div
                key={stage.title}
                initial={{ opacity: 0, y: 20 }}
                animate={isInView ? { opacity: 1, y: 0 } : {}}
                transition={{ duration: 0.5, delay: index * 0.15 }}
                className={`relative flex flex-col lg:flex-row gap-8 ${
                  index % 2 === 0 ? "lg:flex-row-reverse" : ""
                }`}
              >
                {/* Timeline Node */}
                <div className="absolute left-8 lg:left-1/2 -translate-x-1/2 z-10">
                  <div
                    className={`flex h-16 w-16 items-center justify-center rounded-full border-4 ${
                      stage.status === "current"
                        ? "border-primary bg-primary/20"
                        : stage.status === "completed"
                        ? "border-emerald-500 bg-emerald-500/20"
                        : "border-border bg-card"
                    }`}
                  >
                    <stage.icon
                      className={`h-7 w-7 ${
                        stage.status === "current"
                          ? "text-primary"
                          : stage.status === "completed"
                          ? "text-emerald-500"
                          : "text-muted-foreground"
                      }`}
                    />
                  </div>
                </div>

                {/* Content */}
                <div className="ml-24 lg:ml-0 lg:w-[calc(50%-4rem)]">
                  <div
                    className={`p-6 rounded-2xl border ${
                      stage.status === "current"
                        ? "border-primary/30 bg-primary/5 glow"
                        : "border-border/50 bg-card/30"
                    }`}
                  >
                    <div className="flex items-center gap-3 mb-4">
                      <h3 className="text-xl font-bold text-foreground">
                        {stage.title}
                      </h3>
                      {stage.status === "current" && (
                        <span className="inline-flex items-center rounded-full bg-primary/10 px-2.5 py-0.5 text-xs font-medium text-primary">
                          Current
                        </span>
                      )}
                    </div>
                    <ul className="space-y-2">
                      {stage.items.map((item) => (
                        <li
                          key={item}
                          className="flex items-start gap-2 text-sm text-muted-foreground"
                        >
                          <div
                            className={`mt-1.5 h-1.5 w-1.5 rounded-full shrink-0 ${
                              stage.status === "current"
                                ? "bg-primary"
                                : stage.status === "completed"
                                ? "bg-emerald-500"
                                : "bg-muted-foreground"
                            }`}
                          />
                          {item}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>

                {/* Spacer for alternating layout */}
                <div className="hidden lg:block lg:w-[calc(50%-4rem)]" />
              </motion.div>
            ))}
          </div>
        </div>

        {/* Pillars */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5, delay: 0.6 }}
          className="glass rounded-2xl p-6 lg:p-8"
        >
          <h3 className="text-lg font-semibold text-foreground mb-6 text-center">
            Strategic Pillars
          </h3>
          <div className="flex flex-wrap items-center justify-center gap-4">
            {pillars.map((pillar, index) => (
              <div key={pillar.label} className="flex items-center">
                <div className="flex items-center gap-3 px-4 py-2 rounded-lg bg-secondary/30">
                  <pillar.icon className="h-5 w-5 text-primary" />
                  <span className="text-sm font-medium text-foreground">
                    {pillar.label}
                  </span>
                </div>
                {index < pillars.length - 1 && (
                  <ArrowRight className="h-4 w-4 text-muted-foreground mx-2" />
                )}
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  )
}
