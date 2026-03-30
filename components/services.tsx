"use client"

import { motion } from "framer-motion"
import { useInView } from "framer-motion"
import { useRef } from "react"
import {
  Code2,
  Brain,
  BarChart3,
  Workflow,
  Server,
  ArrowRight,
} from "lucide-react"

const services = [
  {
    icon: Code2,
    title: "Software Development",
    description:
      "Custom enterprise solutions built for scale and performance",
    items: ["Web applications", "Mobile apps", "SaaS platforms", "Enterprise systems"],
    color: "from-blue-500 to-cyan-500",
  },
  {
    icon: Brain,
    title: "AI / Machine Learning",
    description: "Intelligent systems that learn and adapt to your business",
    items: [
      "Predictive analytics",
      "Computer vision",
      "NLP & chatbots",
      "Generative AI integrations",
    ],
    color: "from-primary to-accent",
  },
  {
    icon: BarChart3,
    title: "Data & Analytics",
    description: "Transform raw data into actionable business intelligence",
    items: ["Data warehousing", "BI dashboards", "Data engineering"],
    color: "from-emerald-500 to-teal-500",
  },
  {
    icon: Workflow,
    title: "Digital Transformation",
    description: "Modernize operations and accelerate business growth",
    items: ["Process automation", "Cloud migration", "IT consulting"],
    color: "from-orange-500 to-amber-500",
  },
  {
    icon: Server,
    title: "Managed Services",
    description: "Reliable infrastructure management and support",
    items: ["DevOps", "Cloud hosting", "Technical support"],
    color: "from-rose-500 to-pink-500",
  },
]

export function Services() {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: true, margin: "-100px" })

  return (
    <section
      id="services"
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
            Services Portfolio
          </span>
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-balance">
            End-to-end solutions for
            <br />
            <span className="text-muted-foreground">enterprise excellence</span>
          </h2>
        </motion.div>

        {/* Services Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {services.map((service, index) => (
            <motion.div
              key={service.title}
              initial={{ opacity: 0, y: 20 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className={`group relative rounded-2xl border border-border/50 bg-card/50 p-6 transition-all hover:border-primary/30 hover:bg-card ${
                index === services.length - 1 ? "md:col-span-2 lg:col-span-1" : ""
              }`}
            >
              {/* Icon */}
              <div
                className={`mb-4 inline-flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br ${service.color}`}
              >
                <service.icon className="h-6 w-6 text-white" />
              </div>

              {/* Content */}
              <h3 className="text-xl font-semibold text-foreground mb-2">
                {service.title}
              </h3>
              <p className="text-sm text-muted-foreground mb-4">
                {service.description}
              </p>

              {/* Items */}
              <ul className="space-y-2 mb-6">
                {service.items.map((item) => (
                  <li
                    key={item}
                    className="flex items-center gap-2 text-sm text-muted-foreground"
                  >
                    <div className="h-1 w-1 rounded-full bg-primary" />
                    {item}
                  </li>
                ))}
              </ul>

              {/* Learn More Link */}
              <a
                href="#contact"
                className="inline-flex items-center gap-1 text-sm font-medium text-primary hover:underline"
              >
                Learn more
                <ArrowRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-1" />
              </a>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}
