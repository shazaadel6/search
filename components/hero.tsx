"use client"

import { motion } from "framer-motion"
import { ArrowRight, TrendingUp, TrendingDown, Play } from "lucide-react"
import Link from "next/link"

const timelineSteps = [
  { year: "2008", label: "On-Premise", active: false },
  { year: "2013", label: "Cloud Era", active: false },
  { year: "2020", label: "Connected", active: false },
  { year: "2030", label: "Intelligent", active: true },
]

const metrics = [
  {
    label: "Faster Delivery",
    value: "+35%",
    trend: "up",
    description: "Customer NPS improvement",
  },
  {
    label: "Legacy Systems",
    value: "-20%",
    trend: "down",
    description: "Slower delivery reduction",
  },
]

export function Hero() {
  return (
    <section className="relative min-h-screen overflow-hidden pt-16">
      {/* Background Effects */}
      <div className="absolute inset-0 grid-background opacity-50" />
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/20 rounded-full blur-3xl" />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-accent/20 rounded-full blur-3xl" />

      <div className="relative mx-auto max-w-7xl px-6 lg:px-8 py-24 lg:py-32">
        <div className="flex flex-col lg:flex-row gap-16 items-center">
          {/* Left Content */}
          <div className="flex-1 text-center lg:text-left">
            {/* Badge */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="inline-flex items-center gap-2 rounded-full border border-border/50 bg-secondary/50 px-4 py-1.5 mb-6"
            >
              <span className="relative flex h-2 w-2">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary opacity-75" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-primary" />
              </span>
              <span className="text-sm text-muted-foreground">
                Introducing AirviewX
              </span>
            </motion.div>

            {/* Headline */}
            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-balance"
            >
              Build faster, operate smarter, and{" "}
              <span className="text-gradient animate-gradient">
                scale effortlessly
              </span>{" "}
              with agentic AI at the core
            </motion.h1>

            {/* Subheadline */}
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="mt-6 text-lg text-muted-foreground max-w-2xl mx-auto lg:mx-0 text-pretty"
            >
              From fragmented legacy to autonomous intelligence - AirviewX is
              the execution layer for the next phase of enterprise evolution
            </motion.p>

            {/* CTAs */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.3 }}
              className="mt-10 flex flex-col sm:flex-row gap-4 justify-center lg:justify-start"
            >
              <Link
                href="#contact"
                className="group inline-flex items-center justify-center gap-2 rounded-full bg-primary px-6 py-3 text-sm font-medium text-primary-foreground transition-all hover:bg-primary/90 hover:shadow-lg hover:shadow-primary/25"
              >
                Request Demo
                <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
              </Link>
              <Link
                href="#technology"
                className="group inline-flex items-center justify-center gap-2 rounded-full border border-border bg-secondary/50 px-6 py-3 text-sm font-medium text-foreground transition-all hover:bg-secondary hover:border-border/80"
              >
                <Play className="h-4 w-4" />
                Learn More
              </Link>
            </motion.div>

            {/* Evolution Timeline */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.4 }}
              className="mt-16"
            >
              <p className="text-xs uppercase tracking-wider text-muted-foreground mb-4">
                Enterprise Evolution
              </p>
              <div className="flex items-center gap-2 justify-center lg:justify-start overflow-x-auto pb-2">
                {timelineSteps.map((step, index) => (
                  <div key={step.year} className="flex items-center">
                    <div
                      className={`flex flex-col items-center px-4 py-2 rounded-lg transition-colors ${
                        step.active
                          ? "bg-primary/10 border border-primary/30"
                          : "bg-secondary/30"
                      }`}
                    >
                      <span
                        className={`text-sm font-mono font-semibold ${
                          step.active ? "text-primary" : "text-muted-foreground"
                        }`}
                      >
                        {step.year}
                      </span>
                      <span className="text-xs text-muted-foreground whitespace-nowrap">
                        {step.label}
                      </span>
                    </div>
                    {index < timelineSteps.length - 1 && (
                      <div className="w-8 h-px bg-border mx-1" />
                    )}
                  </div>
                ))}
              </div>
            </motion.div>
          </div>

          {/* Right Content - Metrics Card */}
          <motion.div
            initial={{ opacity: 0, x: 40 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            className="flex-1 w-full max-w-lg"
          >
            <div className="glass rounded-2xl p-6 glow">
              <h3 className="text-sm font-medium text-muted-foreground mb-6">
                Enterprise Transformation Metrics
              </h3>

              <div className="space-y-6">
                {metrics.map((metric) => (
                  <div
                    key={metric.label}
                    className="flex items-center justify-between p-4 rounded-xl bg-secondary/30"
                  >
                    <div>
                      <p className="text-sm text-muted-foreground">
                        {metric.label}
                      </p>
                      <p className="text-xs text-muted-foreground/70 mt-1">
                        {metric.description}
                      </p>
                    </div>
                    <div
                      className={`flex items-center gap-2 ${
                        metric.trend === "up"
                          ? "text-emerald-400"
                          : "text-red-400"
                      }`}
                    >
                      {metric.trend === "up" ? (
                        <TrendingUp className="h-5 w-5" />
                      ) : (
                        <TrendingDown className="h-5 w-5" />
                      )}
                      <span className="text-2xl font-bold">{metric.value}</span>
                    </div>
                  </div>
                ))}
              </div>

              <p className="mt-6 text-xs text-muted-foreground/70 text-center">
                Based on observed enterprise transformations across telecom,
                utilities, and field-service environments
              </p>

              {/* Visual Element */}
              <div className="mt-6 relative h-32 rounded-xl bg-gradient-to-br from-primary/5 to-accent/5 border border-border/30 overflow-hidden">
                <div className="absolute inset-0 flex items-end justify-around p-4">
                  {[40, 65, 55, 80, 70, 90, 85].map((height, i) => (
                    <motion.div
                      key={i}
                      initial={{ height: 0 }}
                      animate={{ height: `${height}%` }}
                      transition={{ duration: 0.8, delay: 0.5 + i * 0.1 }}
                      className="w-4 rounded-t bg-gradient-to-t from-primary/40 to-primary"
                    />
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>

      {/* Bottom Gradient */}
      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-background to-transparent" />
    </section>
  )
}
