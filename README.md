# Student Life Analysis Pipeline

A modular pipeline for analyzing student life data using Large Language Models (LLMs). This project aggregates student behavioral data and generates insightful weekly summaries using OpenAI and Anthropic APIs.

## ✨ Features

- **Multi-source data analysis**: Processes emotions, academics, activities, and behavioral patterns
- **LLM integration**: Supports OpenAI and Anthropic for advanced analysis and summarization
- **Flexible configuration**: Environment variables and JSON/CSV configuration files
- **Weekly insights**: Generates comprehensive weekly summaries for individual students
- **Modular architecture**: Clean, extensible pipeline design

## 📁 Project Structure

```
.
├── all_pipeline.py          # Main pipeline script
├── agents/                  # Analysis agents and modules
├── dataset/
│   ├── education/
│   │   ├── class.csv       # Class enrollment data
│   │   ├── class_info.json # Course information
│   │   └── deadlines.csv   # Assignment deadlines
│   └── *.csv               # Additional student data (gitignored)
├── BigFive.csv             # Personality assessment data
├── lab_assignment.csv      # Laboratory assignment records
├── .env.example            # Environment variables template
├── requirements.txt        # Python dependencies
└── README.md
```

## 🛠️ Requirements

- **Python 3.8+**
- pip package manager
- API keys for OpenAI and/or Anthropic

## 💻 Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd student-life-analysis-pipeline
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` with your API keys:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

4. **Verify data files:**
   Ensure the following files are present:
   - `BigFive.csv`
   - `lab_assignment.csv`
   - `dataset/education/class.csv`
   - `dataset/education/class_info.json`
   - `dataset/education/deadlines.csv`

## 🚀 Usage

### Basic Usage

Run the main pipeline with default settings:

```bash
python all_pipeline.py
```

**Default configuration:**
- **User ID**: `u58`
- **Analysis period**: Weeks 1-10
- **LLM provider**: OpenAI (configurable)

### Custom Configuration

Modify the `config` dictionary in `all_pipeline.py` to customize:

```python
config = {
    'uid': 'u58',              # Target user ID
    'weeks': range(1, 11),     # Analysis period
    'client_type': 'openai',   # 'openai' or 'anthropic'
    # ... other settings
}
```

## ⚙️ Configuration Options

| Parameter | Description | Options |
|-----------|-------------|---------|
| `uid` | Student user ID to analyze | Any valid user ID in dataset |
| `client_type` | LLM provider | `"openai"` or `"anthropic"` |
| `weeks` | Analysis time period | Range object (e.g., `range(1, 11)`) |

## 📊 Output

The pipeline generates comprehensive weekly analysis including:

- **Emotional patterns**: Mood trends and emotional state analysis
- **Academic performance**: Grades, assignment completion, study habits
- **Behavioral insights**: Activity patterns and engagement metrics
- **Personalized recommendations**: Based on Big Five personality assessment

### Sample Output

```
Week 1 Analysis for User u58:
- Emotion Score: 7.2/10 (Positive trend)
- Academic Score: 8.5/10 (High engagement)
- Key Insights: Strong start to semester, consistent study patterns
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 📚 Citation
If you use this code in your research, please cite our paper:
```
bibtex@misc{sommuang2025llm,
  title={LLM Agent-Based Simulation of Student Activities and Mental Health Using Smartphone Sensing Data},
  author={Sommuang, Wayupuk and Kerdthaisong, Kun and Buakhaw, Pasin and Wong, Aslan B. and Yongsatianchot, Nutchanon},
  series={UbiComp Companion '25},
  year={2025},
  month={June},
  publisher={ACM},
  address={New York, NY, USA},
  location={Espoo, Finland},
  doi={10.1145/3714394.3750585},
  url={https://doi.org/10.1145/3714394.3750585}
}
```

## 🙏 Acknowledgments

- PreceptorAI Powered by Cariva for compute resource
- Contributors to the student life dataset

---

**Need help?** Open an issue or contact the maintainers.
