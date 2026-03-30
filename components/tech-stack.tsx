"use client"

import { motion } from "framer-motion"
import { useInView } from "framer-motion"
import { useRef } from "react"
import {
  Cpu,
  Bot,
  BarChart,
  Workflow,
  Smartphone,
  Cloud,
  Zap,
  Shield,
  CheckCircle2,
} from "lucide-react"

const techLayers = [
  {
    layer: "CORE",
    tech: "BPMN & RPA",
    function: "Process execution & last-mile automation",
    icon: Workflow,
    color: "bg-blue-500",
  },
  {
    layer: "BPMN",
    tech: "Process as System of Execution",
    function:
      "End-to-end, event-driven orchestration across CRM, catalog, fulfillment, field operations, billing",
    icon: Cpu,
    color: "bg-primary",
  },
  {
    layer: "RPA",
    tech: "Last-Mile & Legacy Automation",
    function:
      "Non-intrusive automation for legacy, human-dependent, non-API systems",
    icon: Bot,
    color: "bg-accent",
  },
  {
    layer: "AI/ML",
    tech: "Agentic Intelligence Layer",
    function:
      "Embedded AI agents for decisioning, prediction, autonomous actions",
    icon: Bot,
    color: "bg-emerald-500",
  },
  {
    layer: "BI/Analytics",
    tech: "Real-Time Operational Intelligence",
    function:
      "Unified, role-based analytics across orders, assets, workforce, incidents",
    icon: BarChart,
    color: "bg-amber-500",
  },
]

const advantages = [
  {
    icon: CheckCircle2,
    title: "Single Low-Code BOS",
    description: "ITSM + BPM + FSM + EAM primitives in one product",
  },
  {
    icon: Zap,
    title: "Auto-Provisioning",
    description: "Niche strength vs. horizontal suites",
  },
  {
    icon: Shield,
    title: "Purpose-Built",
    description: "For complex service orchestration",
  },
  {
    icon: Workflow,
    title: "Project Orchestration",
    description: "Bridging digital planning to physical execution",
  },
]

const benefits = [
  { icon: Smartphone, label: "Mobile-First Design" },
  { icon: Cloud, label: "Scalable Architecture" },
  { icon: Zap, label: "Faster Rollout" },
  { icon: Shield, label: "Lower TCO" },
]

export function TechStack() {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: true, margin: "-100px" })

  return (
    <section
      id="technology"
      className="relative py-24 lg:py-32 overflow-hidden"
    >
      {/* Background */}
      <div className="absolute inset-0 grid-background opacity-30" />
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-primary/5 rounded-full blur-3xl" />

      <div ref={ref} className="relative mx-auto max-w-7xl px-6 lg:px-8">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5 }}
          className="text-center mb-16"
        >
          <span className="inline-block text-xs uppercase tracking-wider text-primary font-medium mb-4">
            Technology Stack
          </span>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-balance mb-6">
            AirviewX Architecture
          </h2>
          <p className="text-lg text-muted-foreground max-w-3xl mx-auto text-pretty">
            Where Processes Execute, Agents Decide, and Intelligence Compounds
          </p>
        </motion.div>

        {/* Tech Layers */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="mb-16"
        >
          <div className="space-y-3">
            {techLayers.map((layer, index) => (
              <motion.div
                key={layer.layer}
                initial={{ opacity: 0, x: -20 }}
                animate={isInView ? { opacity: 1, x: 0 } : {}}
                transition={{ duration: 0.5, delay: 0.2 + index * 0.1 }}
                className="group flex flex-col md:flex-row md:items-center gap-4 p-5 rounded-xl border border-border/50 bg-card/30 hover:bg-card/60 hover:border-primary/30 transition-all"
              >
                <div className="flex items-center gap-4 md:w-1/4">
                  <div
                    className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-lg ${layer.color}`}
                  >
                    <layer.icon className="h-5 w-5 text-white" />
                  </div>
                  <div>
                    <span className="text-xs font-mono text-primary uppercase">
                      {layer.layer}
                    </span>
                    <h4 className="font-semibold text-foreground">
                      {layer.tech}
                    </h4>
                  </div>
                </div>
                <div className="md:w-3/4 md:border-l md:border-border/50 md:pl-6">
                  <p className="text-sm text-muted-foreground">
                    {layer.function}
                  </p>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Advantages & Benefits */}
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Advantages */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={isInView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.5, delay: 0.5 }}
            className="glass rounded-2xl p-6 lg:p-8"
          >
            <h3 className="text-lg font-semibold text-foreground mb-6">
              The AirviewX Advantage
            </h3>
            <div className="space-y-4">
              {advantages.map((advantage) => (
                <div key={advantage.title} className="flex items-start gap-3">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary/10">
                    <advantage.icon className="h-4 w-4 text-primary" />
                  </div>
                  <div>
                    <h4 className="font-medium text-foreground">
                      {advantage.title}
                    </h4>
                    <p className="text-sm text-muted-foreground">
                      {advantage.description}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Benefits */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={isInView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.5, delay: 0.6 }}
            className="glass rounded-2xl p-6 lg:p-8"
          >
            <h3 className="text-lg font-semibold text-foreground mb-6">
              Key Benefits
            </h3>
            <div className="grid grid-cols-2 gap-4">
              {benefits.map((benefit) => (
                <div
                  key={benefit.label}
                  className="flex flex-col items-center justify-center p-4 rounded-xl bg-secondary/30 text-center"
                >
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary/10 mb-3">
                    <benefit.icon className="h-6 w-6 text-primary" />
                  </div>
                  <span className="text-sm font-medium text-foreground">
                    {benefit.label}
                  </span>
                </div>
              ))}
            </div>

            {/* Additional Benefits List */}
            <div className="mt-6 pt-6 border-t border-border/50">
              <div className="flex flex-wrap gap-2">
                {[
                  "Fewer integrations",
                  "Complete visibility",
                  "Real-time sync",
                  "AI-ready platform",
                  "Microservices",
                  "Auto-scaling",
                ].map((tag) => (
                  <span
                    key={tag}
                    className="inline-block rounded-full bg-secondary/50 px-3 py-1 text-xs text-muted-foreground"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  )
}
