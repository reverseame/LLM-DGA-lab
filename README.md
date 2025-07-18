# The Machines are Watching: Exploring the Potential of Large Language Models for Detecting Algorithmically Generated Domains

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A comprehensive framework for evaluating Large Language Models (LLMs) in detecting Algorithmically Generated Domains (AGDs) used by malware for Command and Control communication. This repository provides the complete implementation of experiments described in our research paper, enabling reproducible evaluation across multiple LLM providers and experimental setups.

## Features

* **Multi-experiment framework** supporting 5 different experimental configurations (0-4)
* **Zero-shot and few-shot evaluation** of 9 LLMs from 4 major vendors (OpenAI, Anthropic, Google, Mistral)
* **Binary classification** framework for distinguishing AGDs from legitimate domains
* **Multiclass family classification** with 10-shot learning for malware family identification
* **Comprehensive metrics** including Accuracy, Precision, Recall, F1-score, FPR, TPR, MCC, and Cohen's κ
* **Automatic retry mechanism** for failed classifications with batch processing optimization
* **Reproducible experiments** aligned with the published research methodology
* **CSV export** with detailed analysis and confidence intervals for statistical validation

## Supported Models

| Provider | Model | Type | Context Window | API Tag |
|----------|-------|------|----------------|---------|
| OpenAI | GPT-4o | Large | 128,000 | `gpt-4o-2024-11-20` |
| OpenAI | GPT-4o-mini | Small | 128,000 | `gpt-4o-mini-2024-07-18` |
| Anthropic | Claude 3.5 Sonnet | Large | 200,000 | `claude-3-5-sonnet-20241022` |
| Anthropic | Claude 3.5 Haiku | Small | 200,000 | `claude-3-5-haiku-20241022` |
| Google | Gemini 1.5 Pro | Large | 2,097,152 | `gemini-1.5-pro-002` |
| Google | Gemini 1.5 Flash | Small | 1,048,576 | `gemini-1.5-flash-002` |
| Google | Gemini 1.5 Flash-8B | Small | 1,048,576 | `gemini-1.5-flash-8b-001` |
| Mistral | Mistral Large | Large | 131,000 | `mistral-large-2411` |
| Mistral | Mistral Small | Small | 32,000 | `mistral-small-2409` |

## Installation

The framework runs on Python 3.11+. To use the evaluation framework, follow these installation steps:

### Requirements

It is necessary to install the required packages. To do this, execute the following command (it is recommended to use a virtualized Python environment):

```bash
pip3 install -r requirements.txt
```

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/reverseame/LLM-DGA-lab
   cd LLM-DGA-lab
   ```

2. **Set up API keys:**
   Create a `.secret` file in the root directory with your API keys using the exact format below:
   ```
   API_KEY_OPENAI="your_openai_api_key_here"
   API_KEY_ANTHROPIC="your_anthropic_api_key_here"
   API_KEY_MISTRALAI="your_mistralai_api_key_here"
   API_KEY_GEMINI="your_gemini_api_key_here"
   ```
   **Important**: Use the exact key names shown above and include the quotes around your API keys.

3. **Prepare the datasets:**
   - **Malicious domains**: Place domain family files in `prompts/datasetAGDFamilies/`
     - Each malware family should have its own CSV file (e.g., `conficker.csv`, `cryptolocker.csv`)
     - Each CSV file must contain one domain per line (no headers, no commas between domains)
     - Example format for `family_name.csv`:
       ```
       abc123domain.com
       xyz456domain.org
       random789domain.net
       ```
   - **Legitimate domains**: Place legitimate domains in `prompts/legitimateDomains/domains.csv`
     - One domain per line format (same as malicious domain files)
     - Example format:
       ```
       google.com
       facebook.com
       amazon.com
       ```

## Usage

### Experiment Configuration

The framework supports five different experimental configurations:

| Experiment | Purpose | Dataset | Approach | Samples |
|------------|---------|---------|----------|---------|
| **0** | Development/Testing | 1 domain + 25 benign | Minimal test | Single domain validation |
| **1** | Binary Classification (Zero-shot) | 1000 malicious per family + 25000 benign | P1 prompt | Zero-shot detection |
| **2** | Binary Classification (Enhanced) | 1000 malicious per family + 25000 benign | P2 prompt | Enhanced with lexical features |
| **3** | Multiclass Family Classification | 2000 per family (25 families) | P3 prompt | 10-shot learning |
| **4** | Real-World Domain Analysis | Custom real-world dataset | P4 prompt | Real DNS traffic analysis |

### Configuration

The `main.py` file contains the primary execution code. Before running experiments, configure the following parameters:

```python
# Execution configuration
SECOND_TRY = False        # Set to True for retry failed classifications
EXPERIMENT = 2           # Set to 0, 1, 2, 3, or 4 for different experiments
BATCH_SIZE = 125         # Batch size for processing domains (optimized for API limits)
SEND_REQUEST = True      # Set to False to skip LLM requests (analysis only)

# Select models to test (comment/uncomment as needed)
LLMS = [
    OpenAI("gpt-4o-2024-11-20"),
    OpenAI("gpt-4o-mini-2024-07-18"),
    Anthropic("claude-3-5-sonnet-20241022"),
    Anthropic("claude-3-5-haiku-20241022"),
    Gemini("gemini-1.5-pro-002"),
    Gemini("gemini-1.5-flash-002"),
    Gemini("gemini-1.5-flash-8b-001"),
    MistralAI("mistral-large-2411"),
    MistralAI("mistral-small-2409"),
]
```

### Running Experiments

**Experiment 0 (Development Testing):**
```bash
# Set EXPERIMENT = 0 in main.py
python main.py
```

**Experiment 1 (P1 - Zero-Shot Binary Classification):**
```bash
# Set EXPERIMENT = 1 in main.py
python main.py
```

**Experiment 2 (P2 - Enhanced Binary Classification):**
```bash
# Set EXPERIMENT = 2 in main.py
python main.py
```

**Experiment 3 (P3 - Multiclass Family Classification):**
```bash
# Set EXPERIMENT = 3 in main.py
python main.py
```

**Experiment 4 (P4 - Real-World Domain Analysis):**
```bash
# Set EXPERIMENT = 4 in main.py
# Note: Requires manual dataset preparation
python main.py
```

### Workflow Execution

The framework follows a structured workflow:

1. **First Run (`SEND_REQUEST = True` and `SECOND_TRY = False`):**
   - Generates or loads prompts based on experiment configuration
   - Sends classification requests to selected LLMs in batches
   - Saves responses to `output/` directory
   - Analyzes results and generates comprehensive metrics
   - Calculates confidence intervals for binary classification experiments

2. **Retry Run (`SEND_REQUEST = True` and `SECOND_TRY = True`):**
   - Identifies domains that were not properly classified
   - Automatically retries classification for missing domains
   - Continues processing until all domains are classified

3. **Analysis Only (`SEND_REQUEST = False`):**
   - Skips LLM API requests
   - Performs analysis on existing results in `output/` directory

### Experiment Details

#### Experiment 0 (Development Testing)
```python
0: {
    "middle_prompts": [],                    # Minimal prompt
    "final_prompt": "EndBinary.txt",         # Binary classification format
    "num_train_samples": 0,                  # Zero-shot
    "num_test_domains": 1,                   # Single domain test
    "num_legitimate_domains": 25             # Small benign set
}
```

#### Experiment 1 (Zero-Shot Binary Classification)
```python
1: {
    "middle_prompts": [],                    # No additional context
    "final_prompt": "EndBinary.txt",         # Binary classification format
    "num_train_samples": 0,                  # Zero-shot learning
    "num_test_domains": 1000,                # 1000 malicious domains per family
    "num_legitimate_domains": 25000          # 25000 legitimate domains
}
```

#### Experiment 2 (Enhanced Binary Classification)
```python
2: {
    "middle_prompts": ["Prompt1.txt"],       # Lexical feature analysis
    "final_prompt": "EndBinary.txt",         # Binary classification format
    "num_train_samples": 0,                  # Zero-shot learning
    "num_test_domains": 1000,                # 1000 malicious domains per family
    "num_legitimate_domains": 25000          # 25000 legitimate domains
}
```

#### Experiment 3 (Multiclass Family Classification)
```python
3: {
    "middle_prompts": ["Prompt1.txt", "Prompt2.txt"],  # Enhanced context + examples
    "final_prompt": "EndMulticlass.txt",     # Multiclass classification format
    "num_train_samples": 10,                 # 10-shot learning
    "num_test_domains": 2000,                # 2000 malicious domains per family
    "num_legitimate_domains": 0              # Only malicious domains
}
```

#### Experiment 4 (Real-World Analysis)
```python
4: {
    "middle_prompts": [],                    # Custom prompt required
    "final_prompt": "",                      # Custom format
    "num_train_samples": 0,                  # Zero-shot
    "num_test_domains": 0,                   # Custom dataset
    "num_legitimate_domains": 0              # Custom dataset
}
```

### Results

After executing the experiments, the framework generates comprehensive results:

#### Output Files

- **LLM Responses**: `output/{model_name}_EXP{experiment_number}.out`
- **Global Metrics**: `metrics/GLOBAL_EXP{experiment_number}.csv`
- **Malicious Domain Metrics**: `metrics/MALICIOUS_EXP{experiment_number}.csv`
- **Benign Domain Metrics**: `metrics/BENIGN_EXP{experiment_number}.csv`
- **Family-Specific Metrics**: `metrics/families/{FAMILY_NAME}_EXP{experiment_number}.csv` (Experiment 3)
- **Retry Domains**: `try_again_domains/{model_name}_EXP{experiment_number}.json`

#### Metrics Interpretation

The CSV files contain the following performance metrics:

**Standard Metrics:**
- `accuracy`: Overall classification accuracy
- `precision`: Precision score for malicious domain detection
- `recall`: Recall score (True Positive Rate)
- `f1_score`: F1-score (harmonic mean of precision and recall)
- `fpr`: False Positive Rate (critical for deployment)
- `tpr`: True Positive Rate
- `mcc`: Matthews Correlation Coefficient
- `kappa`: Cohen's Kappa Coefficient

## Project Structure

```
├── main.py                    # Main execution script with 5-experiment framework
├── utils/
│   ├── analyzer.py            # Comprehensive analysis with confidence intervals
│   ├── generatePrompt.py      # Prompt generation and domain management
│   ├── metrics.py             # Standard metrics class definition
│   ├── confidence_metrics.py  # Extended metrics with confidence intervals
│   ├── file_utils.py          # File handling utilities
│   └── config.py              # Configuration management
├── models/
│   ├── LLM.py                 # Abstract base class for LLM implementations
│   ├── OpenAI/                # OpenAI API implementation
│   ├── Anthropic/             # Anthropic API implementation
│   ├── Gemini/                # Google Gemini API implementation
│   └── MistralAI/             # Mistral API implementation
├── prompts/                   # Prompt templates and instructions
│   ├── StartingPoints/StartBase.txt          # Base prompt template
│   ├── Prompt4Experiments/Prompt1.txt        # Lexical feature analysis component
│   ├── Prompt4Experiments/Prompt2.txt        # Family classification examples
│   ├── EndingPoints/EndBinary.txt            # Binary classification instructions
│   ├── EndingPoints/EndMulticlass.txt        # Multiclass classification instructions
│   ├── datasetAGDFamilies/    # Malicious domain families (25 families)
│   └── legitimateDomains/     # Legitimate domain datasets
├── dataset/                   # Generated datasets per experiment (auto-created)
├── output/                    # LLM responses (auto-created)
├── metrics/                   # Calculated metrics (auto-created)
├── try_again_domains/         # Domains needing reclassification (auto-created)
├── requirements.txt           # Python dependencies
└── .secret                    # API keys configuration
```

## Troubleshooting

### Common Issues

1. **API Rate Limits**: The script includes automatic retry with 15-second delays for rate limit handling
2. **Missing Domain Classifications**: Use `SECOND_TRY = True` to automatically retry unclassified domains
3. **File Permissions**: Ensure write permissions for `dataset/`, `output/`, and `metrics/` directories
4. **API Key Issues**: Verify your `.secret` file format and key validity

### Error Files

- `format_error.txt`: Contains malformed LLM responses that couldn't be parsed
- `format_error_multiclass.txt`: Multiclass-specific format errors
- `try_again_domains/`: Directory containing domains that need reclassification

## License

Licensed under the [GNU GPLv3](LICENSE) license.

## How to cite

If you are using this software or find our research useful, please cite it as follows:

```bibtex
TBD
```

## Funding support

This research was supported by grant PID2023-151467OA-I00 (CRAPER), funded by MICIU/AEI/10.13039/501100011033 and by ERDF/EU, by grant TED2021-131115A-I00 (MIMFA), funded by MICIU/AEI/10.13039/501100011033 and by the European Union NextGenerationEU/PRTR, by grant Proyecto Estratégico Ciberseguridad EINA UNIZAR, funded by the Spanish National Cybersecurity Institute (INCIBE) and the European Union NextGenerationEU/PRTR, by grant Programa de Proyectos Estratégicos de Grupos de Investigación (DisCo research group, ref. T21-23R), funded by the University, Industry and Innovation Department of the Aragonese Government, and by the RAPID project (Grant No. CS.007) financed by the Dutch Research Council (NWO).

We extend our gratitude to the DGArchive team for providing the current dataset in advance, allowing us to begin experimentation sooner.

![INCIBE_logos](misc/img/INCIBE_logos.jpg)