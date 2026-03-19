# 🛒 Smart Retail Assistant: Vision-Agentic Checkout System

An end-to-end "Smart Checkout" solution that combines **Computer Vision** (Object Detection) with **Semantic Reasoning** (Local LLM Agent). This system identifies retail products, calculates prices via a custom ontology, and applies business logic using a local AI agent.

---

## 🌟 Key Features
* **Dual-Model Vision:** Supports benchmarking between **YOLOv8n** and **YOLO11n** architectures.
* **Semantic Reasoning Agent:** Powered by **Ollama (Llama 3.2:3b)** to handle age verification and promotional logic.
* **Dynamic Ontology:** Product metadata (prices, categories, promos) managed via a structured `ontology.json`.
* **Professional UI:** A reactive **Streamlit** dashboard for image uploads and digital receipt generation.
* **Privacy-First:** 100% local execution—no data leaves the machine.

---

## 🛠️ Technology Stack
* **Language:** Python 3.13
* **Vision:** Ultralytics (YOLOv8/11), OpenCV, PIL
* **Reasoning:** Ollama (Llama 3.2 3B)
* **Interface:** Streamlit
* **Data:** JSON-based Product Ontology

---

## 🚀 Getting Started

### 1. Prerequisites
* Install [Ollama](https://ollama.com/)
* Install [Python 3.10+](https://www.python.org/downloads/)
* Core Computer Vision
* ultralytics>=8.3.0
opencv-python>=4.8.0
pillow>=10.0.0
numpy>=1.24.0

* AI Reasoning Agent
ollama>=0.3.3

* Web Interface & Dashboard
streamlit>=1.30.0

* Utilities
protobuf>=3.20.3
requests>=2.31.0
