import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LoRA Chat & Train",
  description: "Private chat application with automatic LoRA fine-tuning",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
