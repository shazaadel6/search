"use client"

import { motion } from "framer-motion"
import { useInView } from "framer-motion"
import { useRef } from "react"
import { TrendingUp, Clock, DollarSign, BarChart3 } from "lucide-react"

const roiMetrics = [
  {
    icon: Clock,
    value: "18",
    unit: "months",
    label: "Full Return",
    description: "Complete ROI realization",
  },
  {
    icon: TrendingUp,
    value: "7.2",
    unit: "months",
    label: "Payback Period",
    description: "Break-even point",
  },
]

const gainAreas = [
  { label: "OPEX Reduction", percentage: 35 },
  { label: "Productivity Gains", percentage: 45 },
  { label: "Intelligence ROI", percentage: 55 },
]

export function ROI() {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: true, margin: "-100px" })

  return (
    <section className="relative py-24 lg:py-32 overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 bg-gradient-to-b from-background via-primary/5 to-background" />

      <div ref={ref} className="relative mx-auto max-w-7xl px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5 }}
          className="glass rounded-3xl p-8 lg:p-12 glow"
        >
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            {/* Left - Main Metrics */}
            <div>
              <span className="inline-block text-xs uppercase tracking-wider text-primary font-medium mb-4">
                ROI & Payback
              </span>
              <h2 className="text-3xl sm:text-4xl font-bold text-balance mb-8">
                Measurable gains across
                <span className="text-gradient"> OPEX, productivity, and intelligence</span>
              </h2>

              <div className="grid sm:grid-cols-2 gap-6">
                {roiMetrics.map((metric, index) => (
                  <motion.div
                    key={metric.label}
                    initial={{ opacity: 0, y: 20 }}
                    animate={isInView ? { opacity: 1, y: 0 } : {}}
                    transition={{ duration: 0.5, delay: 0.2 + index * 0.1 }}
                    className="p-6 rounded-2xl bg-secondary/30 border border-border/30"
                  >
                    <div className="flex items-center gap-3 mb-4">
                      <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                        <metric.icon className="h-5 w-5 text-primary" />
                      </div>
                    </div>
                    <div className="flex items-baseline gap-1">
                      <span className="text-4xl font-bold text-foreground">
                        {metric.value}
                      </span>
                      <span className="text-lg text-muted-foreground">
                        {metric.unit}
                      </span>
                    </div>
                    <div className="mt-2">
                      <div className="text-sm font-medium text-foreground">
                        {metric.label}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {metric.description}
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>

            {/* Right - Gain Areas */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={isInView ? { opacity: 1, x: 0 } : {}}
              transition={{ duration: 0.5, delay: 0.4 }}
            >
              <h3 className="text-lg font-semibold text-foreground mb-6">
                Performance Gains
              </h3>
              <div className="space-y-6">
                {gainAreas.map((area, index) => (
                  <div key={area.label}>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-foreground">
                        {area.label}
                      </span>
                      <span className="text-sm font-mono text-primary">
                        +{area.percentage}%
                      </span>
                    </div>
                    <div className="h-3 rounded-full bg-secondary/50 overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={isInView ? { width: `${area.percentage}%` } : {}}
                        transition={{
                          duration: 1,
                          delay: 0.5 + index * 0.15,
                          ease: "easeOut",
                        }}
                        className="h-full rounded-full bg-gradient-to-r from-primary to-accent"
                      />
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-8 p-4 rounded-xl bg-primary/5 border border-primary/20">
                <div className="flex items-center gap-3">
                  <BarChart3 className="h-5 w-5 text-primary" />
                  <p className="text-sm text-muted-foreground">
                    Based on enterprise deployments across telecom, utilities, and field-service environments
                  </p>
                </div>
              </div>
            </motion.div>
          </div>
        </motion.div>
      </div>
    </section>
  )
}
