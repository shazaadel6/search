"use client"

import { motion } from "framer-motion"
import { useInView } from "framer-motion"
import { useRef } from "react"
import {
  Layers,
  Clock,
  Database,
  DollarSign,
  Calendar,
  Hourglass,
  Gauge,
} from "lucide-react"

const painPoints = [
  {
    icon: Layers,
    stat: "130+",
    label: "SaaS Apps Per Company",
    description: "Average enterprise software stack",
  },
  {
    icon: Clock,
    stat: "25%",
    label: "Context Switching",
    description: "Wasted productivity time",
  },
  {
    icon: Database,
    stat: "Majority",
    label: "Unused Data",
    description: "Trapped in silos",
  },
  {
    icon: DollarSign,
    stat: "3-5x",
    label: "Higher Costs",
    description: "Due to fragmentation",
  },
  {
    icon: Calendar,
    stat: "6-12",
    label: "Months Deployment",
    description: "Average rollout time",
  },
  {
    icon: Hourglass,
    stat: "40%",
    label: "Time Waste",
    description: "On manual processes",
  },
  {
    icon: Gauge,
    stat: "<10%",
    label: "Automated Processes",
    description: "In most enterprises",
  },
]

export function Problems() {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: true, margin: "-100px" })

  return (
    <section id="problems" className="relative py-24 lg:py-32 overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 bg-gradient-to-b from-background via-secondary/20 to-background" />

      <div ref={ref} className="relative mx-auto max-w-7xl px-6 lg:px-8">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5 }}
          className="text-center mb-16"
        >
          <span className="inline-block text-xs uppercase tracking-wider text-primary font-medium mb-4">
            Market Problems
          </span>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-balance">
            AI without orchestration
            <br />
            <span className="text-muted-foreground">and authority is theater</span>
          </h2>
        </motion.div>

        {/* Pain Points Grid */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 lg:gap-6">
          {painPoints.map((point, index) => (
            <motion.div
              key={point.label}
              initial={{ opacity: 0, y: 20 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className={`group relative rounded-2xl border border-border/50 bg-card/50 p-6 transition-all hover:border-primary/30 hover:bg-card ${
                index === painPoints.length - 1 ? "col-span-2 md:col-span-1" : ""
              }`}
            >
              <div className="flex flex-col h-full">
                <div className="mb-4 inline-flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
                  <point.icon className="h-5 w-5" />
                </div>
                <div className="text-3xl font-bold text-foreground mb-1">
                  {point.stat}
                </div>
                <div className="text-sm font-medium text-foreground mb-1">
                  {point.label}
                </div>
                <div className="text-xs text-muted-foreground">
                  {point.description}
                </div>
              </div>

              {/* Hover glow effect */}
              <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-primary/5 to-accent/5 opacity-0 group-hover:opacity-100 transition-opacity -z-10" />
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}
