# Lecture 195: Qwen 3.8 27B with DeepSeek Harness Agent is Awesome

This project demonstrates running an autonomous local desktop agent powered entirely by **Qwen 3.8 27B** via **Ollama**, orchestrated by the **DeepSeek Harness** (`dsh`) agent framework, without sending any data to external APIs.

---

## Architecture Overview

```
┌────────────────────────────┐
│   DeepSeek Harness (dsh)   │   Autonomous task planner, shell tool execution,
│    CLI / Web UI Interface  │   file management, and multi-turn verification
└─────────────┬──────────────┘
              │ OpenAI-compatible API (http://localhost:11434/v1)
              ▼
┌────────────────────────────┐
│       Ollama Engine        │   Runs Qwen 3.8 27B locally on GPU/CPU
│     (qwen3.8:27b-q4_k_m)   │   High-accuracy tool calling & structured planning
└─────────────┬──────────────┘
              │ Direct file system operations
              ▼
┌────────────────────────────┐
│     Target Workspace       │   Manipulates files, writes Markdown reports,
│   (sample-workspace/)      │   and performs tabular data aggregation
└────────────────────────────┘
```

---

## The Three Agent Tasks

The demonstration tests three sequential agent tasks located in [`tasks/`](tasks/):

### Task 1: Autonomous File Organization ([`tasks/task1-sort-files.txt`](tasks/task1-sort-files.txt))
> *"This folder is a mess. Sort the files into subfolders by type: Images, Documents and Data. Then tell me what you moved."*
- **Agent Behavior**: Inspects the folder contents using `ls` / `dir`, creates subfolders (`Images/`, `Documents/`, `Data/`), moves corresponding files into each directory, and produces a clean summary table of relocated files.

### Task 2: Data Analysis & Report Generation ([`tasks/task2-workout-report.txt`](tasks/task2-workout-report.txt))
> *"Now read workout-log.csv in the Data folder. Total my minutes for each activity, and save a short report called report.md in the Documents folder, with a table and the activity I did most."*
- **Agent Behavior**: Reads and parses `Data/workout-log.csv`, aggregates durations by exercise (Running: 135 mins, Cycling: 95 mins, Weightlifting: 60 mins, Yoga: 40 mins), identifies Running as the top activity, and creates formatted Markdown at `Documents/report.md`.

### Task 3: Iterative Refinement ([`tasks/task3-add-total.txt`](tasks/task3-add-total.txt))
> *"Nice. Add one line at the end of report.md with my total time for the week in hours."*
- **Agent Behavior**: Reads `report.md`, calculates the sum ($330 \text{ minutes} = 5.5 \text{ hours}$), and appends the calculated weekly total without breaking the existing report layout.

---

## Step-by-Step Walkthrough

### 1. Launch Ollama with Qwen 3.8 27B
Make sure Ollama is running and has Qwen 3.8 pulled:
```bash
ollama pull qwen3.8:27b
ollama run qwen3.8:27b
```

### 2. Launch DeepSeek Harness
Launch DeepSeek Harness via npx or global installation:
```bash
npx @deepseek-ai/dsh web
```
Or start in terminal CLI mode targeting `sample-workspace`:
```bash
npx @deepseek-ai/dsh --provider ollama --model qwen3.8:27b --cwd ./sample-workspace
```

### 3. Execute the Prompts
Feed the prompts sequentially from `tasks/task1-sort-files.txt`, `tasks/task2-workout-report.txt`, and `tasks/task3-add-total.txt` into the harness interface.

---

## Why Qwen 3.8 27B Excels Here
1. **Instruction Following**: Strictly adheres to negative constraints and multi-part requirements without skipping steps.
2. **Tool Reliability**: Uses bash/PowerShell commands cleanly with correct syntax, checking results before concluding.
3. **Local Privacy**: All reasoning and file manipulation occurs 100% on the local host with zero network egress.
