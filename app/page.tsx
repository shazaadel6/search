import { Header } from "@/components/header"
import { Hero } from "@/components/hero"
import { Problems } from "@/components/problems"
import { Competitive } from "@/components/competitive"
import { Services } from "@/components/services"
import { TechStack } from "@/components/tech-stack"
import { Industries } from "@/components/industries"
import { UseCase } from "@/components/use-case"
import { ROI } from "@/components/roi"
import { Roadmap } from "@/components/roadmap"
import { Contact } from "@/components/contact"
import { Footer } from "@/components/footer"

export default function Home() {
  return (
    <main className="relative min-h-screen overflow-x-hidden">
      <Header />
      <Hero />
      <Problems />
      <Competitive />
      <Services />
      <TechStack />
      <Industries />
      <UseCase />
      <ROI />
      <Roadmap />
      <Contact />
      <Footer />
    </main>
  )
}
