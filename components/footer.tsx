"use client"

import Link from "next/link"
import { Linkedin, Twitter, Github } from "lucide-react"

const footerLinks = {
  Products: [
    { label: "AirviewX Platform", href: "#technology" },
    { label: "BPMN Engine", href: "#technology" },
    { label: "RPA Automation", href: "#technology" },
    { label: "AI/ML Layer", href: "#technology" },
    { label: "Analytics", href: "#technology" },
  ],
  Solutions: [
    { label: "Telecom", href: "#industries" },
    { label: "Public Sector", href: "#industries" },
    { label: "Construction", href: "#industries" },
    { label: "Utilities", href: "#industries" },
  ],
  Services: [
    { label: "Software Development", href: "#services" },
    { label: "AI/ML Services", href: "#services" },
    { label: "Data & Analytics", href: "#services" },
    { label: "Digital Transformation", href: "#services" },
    { label: "Managed Services", href: "#services" },
  ],
  Company: [
    { label: "About Us", href: "#" },
    { label: "Careers", href: "#" },
    { label: "Blog", href: "#" },
    { label: "Contact", href: "#contact" },
  ],
}

const socialLinks = [
  { icon: Linkedin, href: "#", label: "LinkedIn" },
  { icon: Twitter, href: "#", label: "Twitter" },
  { icon: Github, href: "#", label: "GitHub" },
]

export function Footer() {
  return (
    <footer className="relative border-t border-border/50 bg-card/30">
      <div className="mx-auto max-w-7xl px-6 lg:px-8 py-16">
        {/* Main Footer Content */}
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-8 lg:gap-12 mb-12">
          {/* Brand Column */}
          <div className="col-span-2 md:col-span-4 lg:col-span-1">
            <Link href="/" className="flex items-center gap-3 mb-4">
              <div className="relative flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-accent">
                <span className="font-mono text-lg font-bold text-white">T</span>
              </div>
              <div className="flex flex-col">
                <span className="text-lg font-semibold text-foreground">
                  Tarkibat
                </span>
                <span className="text-[10px] text-muted-foreground -mt-1">
                  Trust to Innovate
                </span>
              </div>
            </Link>
            <p className="text-sm text-muted-foreground mb-6 max-w-xs">
              The Enterprise Swiss-Knife for automation, orchestration, and
              agentic AI. Building the new business operation system.
            </p>
            <div className="flex items-center gap-3">
              {socialLinks.map((social) => (
                <a
                  key={social.label}
                  href={social.href}
                  aria-label={social.label}
                  className="flex h-9 w-9 items-center justify-center rounded-lg bg-secondary/50 text-muted-foreground hover:bg-secondary hover:text-foreground transition-colors"
                >
                  <social.icon className="h-4 w-4" />
                </a>
              ))}
            </div>
          </div>

          {/* Link Columns */}
          {Object.entries(footerLinks).map(([title, links]) => (
            <div key={title}>
              <h4 className="text-sm font-semibold text-foreground mb-4">
                {title}
              </h4>
              <ul className="space-y-3">
                {links.map((link) => (
                  <li key={link.label}>
                    <Link
                      href={link.href}
                      className="text-sm text-muted-foreground hover:text-foreground transition-colors"
                    >
                      {link.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* Bottom Bar */}
        <div className="pt-8 border-t border-border/50">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <p className="text-sm text-muted-foreground">
              &copy; {new Date().getFullYear()} Tarkibat. All rights reserved.
              Confidential with Copyrights.
            </p>
            <div className="flex items-center gap-6">
              <Link
                href="#"
                className="text-sm text-muted-foreground hover:text-foreground transition-colors"
              >
                Privacy Policy
              </Link>
              <Link
                href="#"
                className="text-sm text-muted-foreground hover:text-foreground transition-colors"
              >
                Terms of Service
              </Link>
              <Link
                href="#"
                className="text-sm text-muted-foreground hover:text-foreground transition-colors"
              >
                Cookie Policy
              </Link>
            </div>
          </div>
        </div>
      </div>
    </footer>
  )
}
