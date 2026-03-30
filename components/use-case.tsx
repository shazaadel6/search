"use client"

import { motion } from "framer-motion"
import { useInView } from "framer-motion"
import { useRef } from "react"
import {
  AlertCircle,
  Cpu,
  CheckCircle2,
  Clock,
  Zap,
  TrendingUp,
  Shield,
  MessageSquare,
  Search,
  FileCheck,
  Send,
  Users,
} from "lucide-react"

const phases = [
  {
    number: "01",
    title: "Detect & Classify",
    subtitle: "Intelligent Alert Capture",
    icon: AlertCircle,
    color: "from-amber-500 to-orange-500",
    sections: [
      {
        title: "Detection Sources",
        items: [
          { icon: Cpu, label: "Monitoring Alerts", desc: "Automated system & network alerts" },
          { icon: Users, label: "User Reports", desc: "Portal, email, phone submissions" },
        ],
      },
      {
        title: "Classification",
        items: [
          { icon: AlertCircle, label: "Severity Assessment", desc: "P1-P4 priority classification" },
          { icon: Search, label: "Impact Analysis", desc: "Service & business impact scope" },
        ],
      },
    ],
  },
  {
    number: "02",
    title: "Triage & Orchestrate",
    subtitle: "AI-Powered Response Engine",
    icon: Cpu,
    color: "from-primary to-accent",
    sections: [
      {
        title: "Incident Templates",
        items: [
          { icon: AlertCircle, label: "Critical Incidents", desc: "P1 major outage workflows" },
          { icon: Zap, label: "Service Outages", desc: "Degradation responses" },
          { icon: Shield, label: "Security Events", desc: "Breach & threat protocols" },
        ],
      },
      {
        title: "Response Modules",
        items: [
          { icon: TrendingUp, label: "Escalation Paths", desc: "Multi-tier routing rules" },
          { icon: FileCheck, label: "Runbooks", desc: "Automated remediation steps" },
          { icon: MessageSquare, label: "Communications", desc: "Stakeholder notifications" },
        ],
      },
    ],
    metrics: [
      { label: "MTTR", value: "4.2 hrs" },
      { label: "Auto-resolved", value: "68%" },
      { label: "Escalation", value: "12%" },
    ],
  },
  {
    number: "03",
    title: "Resolve & Recover",
    subtitle: "Full Lifecycle Execution",
    icon: CheckCircle2,
    color: "from-emerald-500 to-teal-500",
  },
]

const workflowSteps = [
  "Incident Logging",
  "Triage & Classification",
  "Assignment & Escalation",
  "Investigation",
  "Resolution Actions",
  "Communication",
  "Verification & Testing",
  "Closure & Documentation",
  "Post-Incident Review",
]

const slaMetrics = [
  { icon: Clock, label: "24/7 Monitoring", value: "Always On" },
  { icon: Zap, label: "Response SLA", value: "< 15m" },
  { icon: Shield, label: "Uptime Target", value: "99.9%" },
]

export function UseCase() {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: true, margin: "-100px" })

  return (
    <section className="relative py-24 lg:py-32 overflow-hidden">
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
            Use Case
          </span>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-balance mb-6">
            Incident Management
          </h2>
          <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
            End-to-end workflow from detection to resolution
          </p>
        </motion.div>

        {/* Phases */}
        <div className="space-y-8 mb-16">
          {phases.map((phase, index) => (
            <motion.div
              key={phase.number}
              initial={{ opacity: 0, y: 20 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.5, delay: index * 0.15 }}
              className="glass rounded-2xl p-6 lg:p-8"
            >
              <div className="flex flex-col lg:flex-row gap-6">
                {/* Phase Header */}
                <div className="lg:w-1/4 flex flex-row lg:flex-col items-center lg:items-start gap-4">
                  <div
                    className={`flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br ${phase.color}`}
                  >
                    <phase.icon className="h-7 w-7 text-white" />
                  </div>
                  <div>
                    <span className="text-xs font-mono text-primary">
                      Phase {phase.number}
                    </span>
                    <h3 className="text-xl font-bold text-foreground">
                      {phase.title}
                    </h3>
                    <p className="text-sm text-muted-foreground">
                      {phase.subtitle}
                    </p>
                  </div>
                </div>

                {/* Phase Content */}
                <div className="lg:w-3/4 lg:border-l lg:border-border/50 lg:pl-8">
                  {phase.sections ? (
                    <div className="grid md:grid-cols-2 gap-6">
                      {phase.sections.map((section) => (
                        <div key={section.title}>
                          <h4 className="text-sm font-medium text-muted-foreground mb-4">
                            {section.title}
                          </h4>
                          <div className="space-y-3">
                            {section.items.map((item) => (
                              <div
                                key={item.label}
                                className="flex items-start gap-3 p-3 rounded-lg bg-secondary/30"
                              >
                                <item.icon className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                                <div>
                                  <div className="text-sm font-medium text-foreground">
                                    {item.label}
                                  </div>
                                  <div className="text-xs text-muted-foreground">
                                    {item.desc}
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div>
                      {/* Workflow Steps for Phase 03 */}
                      <div className="flex flex-wrap gap-2">
                        {workflowSteps.map((step, stepIndex) => (
                          <div key={step} className="flex items-center">
                            <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-secondary/50 text-xs font-medium text-foreground">
                              <span className="text-primary font-mono">
                                {String(stepIndex + 1).padStart(2, "0")}
                              </span>
                              {step}
                            </span>
                            {stepIndex < workflowSteps.length - 1 && (
                              <Send className="h-3 w-3 text-muted-foreground mx-1 rotate-0" />
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Metrics */}
                  {phase.metrics && (
                    <div className="mt-6 pt-6 border-t border-border/50">
                      <div className="flex flex-wrap gap-6">
                        {phase.metrics.map((metric) => (
                          <div key={metric.label} className="text-center">
                            <div className="text-2xl font-bold text-gradient">
                              {metric.value}
                            </div>
                            <div className="text-xs text-muted-foreground">
                              {metric.label}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* SLA Metrics */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5, delay: 0.5 }}
          className="grid sm:grid-cols-3 gap-4"
        >
          {slaMetrics.map((metric) => (
            <div
              key={metric.label}
              className="flex items-center justify-center gap-4 p-6 rounded-2xl border border-primary/20 bg-primary/5"
            >
              <metric.icon className="h-8 w-8 text-primary" />
              <div>
                <div className="text-sm text-muted-foreground">
                  {metric.label}
                </div>
                <div className="text-xl font-bold text-foreground">
                  {metric.value}
                </div>
              </div>
            </div>
          ))}
        </motion.div>
      </div>
    </section>
  )
}
