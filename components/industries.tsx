"use client"

import { motion } from "framer-motion"
import { useInView } from "framer-motion"
import { useRef, useState } from "react"
import {
  Radio,
  Building2,
  HardHat,
  Zap,
  ArrowRight,
  AlertTriangle,
  Target,
  DollarSign,
} from "lucide-react"

const industries = [
  {
    id: "telecom",
    icon: Radio,
    name: "Telecom & Managed Services",
    marketSize: "$15B",
    painPoints: [
      "Delayed activations",
      "Manual provisioning",
      "Gig truck rolls",
    ],
    needs: [
      "Zero-touch workflows",
      "Deployment automation",
      "Asset lineage",
    ],
    color: "from-blue-500 to-cyan-500",
  },
  {
    id: "public",
    icon: Building2,
    name: "Public Sector",
    marketSize: "$12B",
    painPoints: ["Fragmented systems", "Slow service delivery"],
    needs: ["Integrated citizen services", "Compliance tracking"],
    color: "from-primary to-accent",
  },
  {
    id: "construction",
    icon: HardHat,
    name: "Construction / Facilities",
    marketSize: "$11B",
    painPoints: [
      "Disjointed work orders",
      "Inspections",
      "Vendor coordination",
    ],
    needs: ["Scheduling", "Site handovers", "CAFM-style reporting"],
    color: "from-amber-500 to-orange-500",
  },
  {
    id: "utilities",
    icon: Zap,
    name: "Utilities / Energy",
    marketSize: "$9B",
    painPoints: ["Legacy EAM + separate ITSM", "Compliance-heavy changes"],
    needs: [
      "Unified change field maintenance",
      "Mobile safety checks",
    ],
    color: "from-emerald-500 to-teal-500",
  },
]

const fragmentationCosts = [
  { icon: AlertTriangle, label: "Zero transparency across systems" },
  { icon: AlertTriangle, label: "Manual data entry & duplication" },
  { icon: AlertTriangle, label: "Delayed decision-making" },
  { icon: DollarSign, label: "High operational costs (3-5x)" },
]

export function Industries() {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: true, margin: "-100px" })
  const [activeIndustry, setActiveIndustry] = useState(industries[0])

  return (
    <section
      id="industries"
      className="relative py-24 lg:py-32 overflow-hidden"
    >
      {/* Background */}
      <div className="absolute inset-0 bg-gradient-to-b from-background via-secondary/10 to-background" />

      <div ref={ref} className="relative mx-auto max-w-7xl px-6 lg:px-8">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5 }}
          className="text-center mb-16"
        >
          <span className="inline-block text-xs uppercase tracking-wider text-primary font-medium mb-4">
            Target Industries
          </span>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-balance mb-6">
            Beachhead Verticals
          </h2>
          <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
            Focused on industries representing 40-45% of global demand
          </p>
        </motion.div>

        {/* Industry Tabs */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="mb-8"
        >
          <div className="flex flex-wrap justify-center gap-2">
            {industries.map((industry) => (
              <button
                key={industry.id}
                onClick={() => setActiveIndustry(industry)}
                className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all ${
                  activeIndustry.id === industry.id
                    ? "bg-primary text-primary-foreground"
                    : "bg-secondary/50 text-muted-foreground hover:bg-secondary hover:text-foreground"
                }`}
              >
                <industry.icon className="h-4 w-4" />
                <span className="hidden sm:inline">{industry.name}</span>
                <span className="sm:hidden">{industry.name.split(" ")[0]}</span>
              </button>
            ))}
          </div>
        </motion.div>

        {/* Active Industry Details */}
        <motion.div
          key={activeIndustry.id}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="glass rounded-2xl p-6 lg:p-8 mb-12"
        >
          <div className="grid lg:grid-cols-3 gap-8">
            {/* Industry Info */}
            <div className="lg:col-span-1">
              <div
                className={`inline-flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br ${activeIndustry.color} mb-4`}
              >
                <activeIndustry.icon className="h-8 w-8 text-white" />
              </div>
              <h3 className="text-2xl font-bold text-foreground mb-2">
                {activeIndustry.name}
              </h3>
              <div className="flex items-center gap-2 text-primary">
                <DollarSign className="h-5 w-5" />
                <span className="text-xl font-semibold">
                  {activeIndustry.marketSize}
                </span>
                <span className="text-sm text-muted-foreground">
                  Market Size
                </span>
              </div>
            </div>

            {/* Pain Points */}
            <div>
              <h4 className="text-sm font-medium text-muted-foreground uppercase tracking-wider mb-4 flex items-center gap-2">
                <AlertTriangle className="h-4 w-4 text-red-400" />
                Pain Points
              </h4>
              <ul className="space-y-3">
                {activeIndustry.painPoints.map((point) => (
                  <li key={point} className="flex items-start gap-3">
                    <div className="mt-1.5 h-1.5 w-1.5 rounded-full bg-red-400 shrink-0" />
                    <span className="text-foreground">{point}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Needs */}
            <div>
              <h4 className="text-sm font-medium text-muted-foreground uppercase tracking-wider mb-4 flex items-center gap-2">
                <Target className="h-4 w-4 text-emerald-400" />
                Industry Needs
              </h4>
              <ul className="space-y-3">
                {activeIndustry.needs.map((need) => (
                  <li key={need} className="flex items-start gap-3">
                    <div className="mt-1.5 h-1.5 w-1.5 rounded-full bg-emerald-400 shrink-0" />
                    <span className="text-foreground">{need}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </motion.div>

        {/* Cost of Fragmentation */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="rounded-2xl border border-red-500/20 bg-red-500/5 p-6 lg:p-8"
        >
          <h3 className="text-lg font-semibold text-foreground mb-6 text-center">
            Structural Cost of Fragmentation
          </h3>
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {fragmentationCosts.map((cost) => (
              <div
                key={cost.label}
                className="flex items-center gap-3 p-4 rounded-xl bg-background/50"
              >
                <cost.icon className="h-5 w-5 text-red-400 shrink-0" />
                <span className="text-sm text-foreground">{cost.label}</span>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  )
}
