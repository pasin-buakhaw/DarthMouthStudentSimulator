Here is your **clean, production-ready `README.md`** generated from your provided notes, formatted for clarity, professionalism, and immediate use:

---

```markdown
# Student Life Analysis Pipeline

This project analyzes student life data using a modular pipeline and integrates with various Large Language Model (LLM) providers (OpenAI, Anthropic) for advanced analysis and summarization.

## ✨ Features

- Aggregates and analyzes student data (emotions, academics, activities, etc.)
- Supports multiple LLM providers: **OpenAI**, **Anthropic**
- Configurable via environment variables and JSON/CSV files
- Outputs weekly summaries for a given student

## 📁 Project Structure

```

.
├── all\_pipeline.py
├── agents/
├── dataset/
│   ├── education/
│   │   ├── class.csv
│   │   ├── class\_info.json
│   │   └── deadlines.csv
│   └── \*.csv (ignored by .gitignore)
├── BigFive.csv
├── lab\_assignment.csv
├── .env.example
├── requirements.txt
└── README.md

````

## ✅ Requirements

- Python **3.8+**
- `pip`

## 💻 Installation



1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Set up environment variables:**

* Copy `.env.example` to `.env` (if provided), or create a `.env` file with your API keys:

```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

3. **Ensure required data files are present:**

* `BigFive.csv`
* `lab_assignment.csv`
* `dataset/education/class.csv`
* `dataset/education/class_info.json`
* `dataset/education/deadlines.csv`

## 🚀 Usage

Run the main pipeline script:

```bash
python all_pipeline.py
```

By default, the script is configured to analyze data for **user `u58`** and **weeks 1-10**, You can change these settings in `all_pipeline.py`.

## ⚙️ Configuration

Edit the `config` dictionary in `all_pipeline.py` to change:

* Data file paths
* User ID (`uid`)
* LLM client type (`client_type`: `"openai"` or `"anthropic"`)

## 📤 Output

The script prints a summary of the analysis for each week, including:

* Emotion scores
* Academic scores

