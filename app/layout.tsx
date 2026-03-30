import type { Metadata, Viewport } from "next"
import { Inter, JetBrains_Mono } from "next/font/google"
import "./globals.css"

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
})

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains-mono",
})

export const metadata: Metadata = {
  title: "Tarkibat | AirviewX - The Enterprise Swiss-Knife for AI Automation",
  description:
    "Build faster, operate smarter, and scale effortlessly with agentic AI at the core. AirviewX is the execution layer for the next phase of enterprise evolution.",
  keywords: [
    "AirviewX",
    "Tarkibat",
    "Enterprise AI",
    "Business Automation",
    "BPMN",
    "RPA",
    "Digital Transformation",
    "Agentic AI",
  ],
  authors: [{ name: "Tarkibat" }],
  openGraph: {
    type: "website",
    locale: "en_US",
    url: "https://airviewx.com",
    siteName: "Tarkibat - AirviewX",
    title: "Tarkibat | AirviewX - Enterprise AI Automation Platform",
    description:
      "From fragmented legacy to autonomous intelligence. AirviewX is the new business operation system.",
  },
  twitter: {
    card: "summary_large_image",
    title: "Tarkibat | AirviewX",
    description:
      "The Enterprise Swiss-Knife for automation, orchestration, and agentic AI",
  },
}

export const viewport: Viewport = {
  themeColor: "#0a0a0f",
  width: "device-width",
  initialScale: 1,
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.variable} ${jetbrainsMono.variable} font-sans`}>
        {children}
      </body>
    </html>
  )
}
