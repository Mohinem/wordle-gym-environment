# 🧠 Wordle Environment — Pure Strategy Agent

This repository contains a **custom Wordle environment** built using [Gymnasium](https://gymnasium.farama.org/) and a **rule-based strategy agent** for solving the game.

The agent uses **action masking, letter constraints, and heuristics** to achieve a win rate of around **~93%** on a 542‑word dataset.  
No machine learning or training is required.

The environment supports **human‑mode rendering** so you can watch the agent play Wordle in real‑time.

---

## 🚀 Features

- **Custom `gymnasium` environment for Wordle**:
  - Implements correct Wordle rules and letter feedback
  - Step‑by‑step feedback after each guess
  - `render_mode="human"` for visual gameplay

- **Pure Strategy Agent** (no training needed):
  - High‑information starting words (`ARISE`, `SOARE`, etc.)
  - Action masking to only select guesses consistent with discovered constraints
  - Letter frequency and pattern‑fitting heuristics for optimal guessing

- **Statistics Tracking**:
  - Win rate, average guesses, recent rolling win rate
  - Win distribution by attempt number
  - Rolling performance plots with Matplotlib

---

## 📦 Installation

**Python 3.10+ recommended**

### Install dependencies:

pip install gymnasium numpy matplotlib


---

## 🕹 Usage

### Run the Pure Strategy Agent
Runs instantly without any training and plays multiple games in sequence.

cd examples

python3 strategic_wordle_agent.py


What it does:
- Plays multiple Wordle episodes
- Uses human‑mode rendering to display the board after each guess
- Tracks and prints performance statistics
- Optionally saves results and generates performance plots


## ⚙️ Customization

- **Change opening words** → edit `self.opening_words` in `StrategyAgent`
- **Change dataset** → update `word_list` in `wordle_env.py`
- **Adjust display speed** → modify `time.sleep()` delays in `strategic_wordle_agent.py`
- **Change number of episodes** → edit `num_episodes` in `test_strategy_agent()`

---

## 📜 License
MIT License — free to use, modify, and distribute with attribution.
