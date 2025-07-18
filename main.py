"""
Main script for LLM evaluation experiments on Algorithmically Generated Domain (AGD) detection.

This program manages prompt generation, sending requests to different language models,
and analyzing their responses for domain classification tasks as described in the paper:
"The Machines are Watching: Exploring the Potential of Large Language Models for 
Detecting Algorithmically Generated Domains"

Supported Experiments:
- Experiment 0: Single domain test (development/testing)
- Experiment 1: Binary classification (zero-shot) - 1000 malicious + 25000 benign
- Experiment 2: Binary classification with lexical features - 1000 malicious + 25000 benign  
- Experiment 3: Multiclass family classification (10-shot) - 2000 malicious per family
- Experiment 4: Real-world domain classification - Custom dataset required
"""

import os
import math

# Models - Import different LLM implementations
from models.Gemini.Gemini import Gemini
from models.OpenAI.OpenAI import OpenAI
from models.Anthropic.Anthropic import Anthropic
from models.MistralAI.MistralAI import MistralAI

# Personal libraries - Import utility modules
from utils.generatePrompt import PromptGenerator
from utils.file_utils import save_to_text_file, load_from_text_file
from utils.analyzer import Analyzer

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================
SECOND_TRY = False        # Flag to indicate if this is a second classification attempt
EXPERIMENT = 2           # ID of the experiment to run (0-4)
BATCH_SIZE = 125         # Batch size for processing domains (optimized for API limits)
SEND_REQUEST = True      # Flag to control whether to send requests to LLMs

# ============================================================================
# LLM SELECTION
# ============================================================================
# List of available LLMs - Comment/uncomment to select models for testing
# Models are grouped by vendor for easy selection
LLMS = [
    # OpenAI Models
    OpenAI("gpt-4o-2024-11-20"),           # GPT-4o (larger model)
    OpenAI("gpt-4o-mini-2024-07-18"),      # GPT-4o-mini (smaller, cost-effective)
    
    # Anthropic Models  
    Anthropic("claude-3-5-sonnet-20241022"),   # Claude Sonnet 3.5 (larger model)
    Anthropic("claude-3-5-haiku-20241022"),    # Claude Haiku 3.5 (smaller model)
    
    # Google Gemini Models
    Gemini("gemini-1.5-pro-002"),         # Gemini Pro (larger model)
    Gemini("gemini-1.5-flash-002"),       # Gemini Flash (balanced)
    Gemini("gemini-1.5-flash-8b-001"),    # Gemini Flash-8B (smallest)
    
    # Mistral AI Models
    MistralAI("mistral-large-2411"),      # Mistral Large
    MistralAI("mistral-small-2409"),      # Mistral Small
]

# ============================================================================
# DIRECTORY CONFIGURATION
# ============================================================================
DATASET_DIR = "dataset/"                  # Directory for storing datasets
SECOND_TRY_DOMAINS = "try_again_domains"  # Directory for domains needing reclassification
METRICS_DIR = "metrics/"                  # Directory for storing metrics
OUTPUT_DIR = "output"                     # Directory for results

# ============================================================================
# CORE COMPONENTS INITIALIZATION
# ============================================================================
GENERATOR = PromptGenerator(
    "prompts/",                           # Base prompts directory
    "prompts/datasetAGDFamilies",        # Malicious domain families dataset
    "prompts/legitimateDomains/domains.csv"  # Legitimate domains dataset
)

ANALYZER = Analyzer(
    "prompts/datasetAGDFamilies",        # Malicious domain families dataset
    "prompts/legitimateDomains/domains.csv"  # Legitimate domains dataset
)

def readPrompt(experiment: int) -> tuple:
    """
    Reads or generates prompts based on the experiment number.
    
    This function handles the configuration for each of the supported experiments
    and either loads existing prompts from disk or generates new ones based on
    the experiment parameters.

    Parameters:
    -----------
    experiment : int
        The experiment number (0-4), which determines the prompt configuration:
        - 0: Single domain test (development)
        - 1: Binary classification (zero-shot, P1 prompt)
        - 2: Binary classification with lexical features (P2 prompt) 
        - 3: Multiclass family classification (P3 prompt with 10-shot learning)
        - 4: Real-world domain classification (custom dataset required)

    Returns:
    --------
    tuple
        A tuple containing:
        - explanationPrompt (str): The explanation prompt for the LLM
        - samplesPromptList (list): List of sample domains to classify

    Raises:
    -------
    ValueError
        If the experiment number is not in the range 0-4
    """
    
    # Validate experiment number range
    if not (0 <= experiment <= 4):
        raise ValueError(f"'experiment' with value {experiment} must be between 0 and 4")

    # Create necessary directories
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(SECOND_TRY_DOMAINS, exist_ok=True)

    # ========================================================================
    # EXPERIMENT CONFIGURATION MAPPING
    # ========================================================================
    # Configuration dictionaries for different experiments based on the paper methodology
    
    # Middle prompts define the contextual information provided to LLMs
    middle_prompts_options = {
        0: [],                    # Experiment 0: Minimal prompt for testing
        1: [],                    # Experiment 1: Zero-shot binary classification (P1)
        2: ["Prompt1.txt"],       # Experiment 2: Binary with lexical features (P2)
        3: ["Prompt1.txt", "Prompt2.txt"],  # Experiment 3: Multiclass (P3)
        4: [],                    # Experiment 4: Real-world domains
    }

    # Final prompts determine the output format and task specification
    final_prompt_options = {
        0: "EndBinary.txt",       # Binary classification format
        1: "EndBinary.txt",       # Binary classification format
        2: "EndBinary.txt",       # Binary classification format
        3: "EndMulticlass.txt",   # Multiclass classification format
        4: "",                    # Custom format for real-world data
    }

    # Number of training samples for few-shot learning
    num_train_samples_options = {
        0: 0,    # No training samples (zero-shot)
        1: 0,    # No training samples (zero-shot)  
        2: 0,    # No training samples (zero-shot)
        3: 10,   # 10-shot learning for family classification
        4: 0,    # No training samples
    }

    # Number of test domains to generate/use
    num_test_domains_options = {
        0: 1,      # Single domain for testing
        1: 1000,   # 1000 malicious domains for binary classification
        2: 1000,   # 1000 malicious domains for binary classification
        3: 2000,   # 2000 malicious domains for multiclass (per family)
        4: 0,      # Custom dataset
    }

    # Number of legitimate domains to include
    num_legitimate_domains_options = {
        0: 25,     # Small set for testing
        1: 25000,  # Large set for robust binary classification
        2: 25000,  # Large set for robust binary classification  
        3: 0,      # Only malicious domains for family classification
        4: 0,      # Custom dataset
    }

    # Get configuration for the specified experiment
    middle_prompts = middle_prompts_options.get(experiment)
    final_prompt = final_prompt_options.get(experiment)
    num_train_samples = num_train_samples_options.get(experiment)
    num_test_domains = num_test_domains_options.get(experiment)
    num_legitimate_domains = num_legitimate_domains_options.get(experiment)

    # ========================================================================
    # PROMPT LOADING/GENERATION
    # ========================================================================
    
    # Define paths for saving/loading prompts
    explanation_prompt_path = os.path.join(DATASET_DIR, str(experiment), "prompt.json")
    samples_prompt_list_path = os.path.join(DATASET_DIR, str(experiment), "samples.json")

    # Create experiment-specific directory
    os.makedirs(os.path.join(DATASET_DIR, str(experiment)), exist_ok=True)

    # Modify path for second try scenarios (reclassification of failed domains)
    if SECOND_TRY:
        samples_prompt_list_path = os.path.join(SECOND_TRY_DOMAINS, f"{LLMS[0].model}_EXP{str(EXPERIMENT)}.json")

    # Try to load existing prompts from disk
    explanationPrompt = load_from_text_file(explanation_prompt_path)
    samplesPromptList = load_from_text_file(samples_prompt_list_path)

    # Special handling for Experiment 4 (real-world domains)
    if (explanationPrompt is None or samplesPromptList is None) and EXPERIMENT == 4:
        print("ERROR: Experiment 4 requires manual dataset preparation!")
        print("Please add the real-world domain dataset manually to the appropriate directory.")
        exit(1)

    # Generate new prompts if they don't exist and it's not experiment 4
    if explanationPrompt is None or samplesPromptList is None:
        print(f"Generating prompts for Experiment {experiment}...")
        explanationPrompt, samplesPromptList = GENERATOR.generate_prompt(
            starting_prompt="StartBase.txt",
            middle_prompts=middle_prompts,
            final_prompt=final_prompt,
            num_train_samples=num_train_samples,
            num_test_domains=num_test_domains,
            num_legitimate_domains=num_legitimate_domains
        )
        
        # Save generated prompts for future use
        save_to_text_file(explanation_prompt_path, explanationPrompt)
        save_to_text_file(samples_prompt_list_path, samplesPromptList)
        print(f"Prompts saved for Experiment {experiment}")
    
    return explanationPrompt, samplesPromptList


def main():
    """
    Main function that executes the complete experiment flow.
    
    This function orchestrates the entire experimental pipeline:
    1. Validates the experiment configuration
    2. Creates necessary directories
    3. Reads or generates prompts based on selected experiment
    4. Executes requests on configured LLMs (if SEND_REQUEST is True)
    5. Analyzes results and generates metrics
    
    The function handles different experiment types:
    - Binary classification (Experiments 1-2, 4)
    - Multiclass classification (Experiment 3)
    - Development testing (Experiment 0)
    """
    
    # ========================================================================
    # EXPERIMENT VALIDATION AND SETUP
    # ========================================================================
    
    # Validate experiment number
    if not (0 <= EXPERIMENT <= 4):
        print(f"ERROR: EXPERIMENT must be between 0 and 4, got {EXPERIMENT}")
        print("Available experiments:")
        print("  0: Single domain test (development)")
        print("  1: Binary classification (zero-shot)")
        print("  2: Binary classification with lexical features") 
        print("  3: Multiclass family classification (10-shot)")
        print("  4: Real-world domain classification")
        return

    # Initialize directory structure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    print(f"Starting Experiment {EXPERIMENT}")
    print(f"Batch size: {BATCH_SIZE} domains per request")
    print(f"Models to evaluate: {len(LLMS)}")

    # ========================================================================
    # PROMPT GENERATION/LOADING
    # ========================================================================
    
    # Get prompts for the selected experiment
    try:
        (explanationPrompt, samplesPromptList) = readPrompt(experiment=EXPERIMENT)
        print(f"Successfully loaded prompts for Experiment {EXPERIMENT}")
        if samplesPromptList:
            print(f"Total domains to process: {len(samplesPromptList)}")
    except Exception as e:
        print(f"ERROR loading prompts: {e}")
        return

    # ========================================================================
    # LLM PROCESSING (if enabled)
    # ========================================================================
    
    if SEND_REQUEST:
        print("\n" + "="*60)
        print("STARTING LLM PROCESSING")
        print("="*60)
        
        # Process each configured LLM
        for llm_idx, llm in enumerate(LLMS, 1):
            model_name = llm.model
            print(f"\n[{llm_idx}/{len(LLMS)}] Processing model: {model_name}")
            print("-" * 50)
            
            # Create conversation history with the explanation prompt
            conversation_history = llm.craftConversationHistory(explanationPrompt, "yes")

            if SECOND_TRY:
                # ============================================================
                # SECOND TRY PROCESSING (Reclassification of failed domains)
                # ============================================================
                print(f"Running second-try classification for {model_name}")
                
                output_file = os.path.join(OUTPUT_DIR, f"{model_name}_EXP{str(EXPERIMENT)}.out")
                try_again_path = os.path.join(SECOND_TRY_DOMAINS, f"{model_name}_EXP{str(EXPERIMENT)}.json")
                
                # Check which domains need reclassification
                all_classified = ANALYZER.check_domains(file_path=output_file, output_path=try_again_path)
                
                # Continue processing until all domains are classified
                iteration = 1
                while not all_classified:
                    print(f"  Reclassification iteration {iteration}")
                    
                    # Load domains that need reclassification
                    samples_prompt_list_path = os.path.join(SECOND_TRY_DOMAINS, f"{LLMS[0].model}_EXP{str(EXPERIMENT)}.json")
                    samplesPromptList = load_from_text_file(samples_prompt_list_path)
                    
                    if not samplesPromptList:
                        print("  No domains to reclassify")
                        break
                        
                    print(f"  Domains to reclassify: {len(samplesPromptList)}")
                    
                    # Process domains in batches
                    total_batches = math.ceil(len(samplesPromptList) / BATCH_SIZE)
                    for i in range(0, len(samplesPromptList), BATCH_SIZE):
                        batch_num = i // BATCH_SIZE + 1
                        chunk = samplesPromptList[i:i + BATCH_SIZE]
                        samplesPrompt = GENERATOR.create_prompt_from_domain_list(chunk)

                        # Calculate progress
                        domains_in_batch = len(samplesPrompt.split(","))
                        print(f"    Batch {batch_num}/{total_batches}: {domains_in_batch} domains")

                        # Get LLM response and save results
                        try:
                            response, _ = llm.chat(samplesPrompt, conversation_history)
                            with open(output_file, 'a', encoding='utf-8') as f:
                                f.write(samplesPrompt + "\n")
                                f.write("-" * 15 + "\n")
                                f.write(response + "\n")
                                f.write("*" * 15 + "\n")
                        except Exception as e:
                            print(f"    ERROR in batch {batch_num}: {e}")
                            continue
                        
                    # Check if all domains are now classified
                    all_classified = ANALYZER.check_domains(file_path=output_file, output_path=try_again_path)
                    iteration += 1
                    
                    if iteration > 5:  # Safety limit
                        print("  WARNING: Maximum reclassification iterations reached")
                        break
                    
            else:  
                # ============================================================
                # FIRST-TIME PROCESSING
                # ============================================================
                print(f"Running first-time classification for {model_name}")
                
                if not samplesPromptList:
                    print("  No domains to process")
                    continue
                    
                print(f"  Total domains: {len(samplesPromptList)}")
                
                # Process domains in batches
                total_batches = math.ceil(len(samplesPromptList) / BATCH_SIZE)
                successful_batches = 0
                
                for i in range(0, len(samplesPromptList), BATCH_SIZE):
                    batch_num = i // BATCH_SIZE + 1
                    chunk = samplesPromptList[i:i + BATCH_SIZE]
                    samplesPrompt = GENERATOR.create_prompt_from_domain_list(chunk)

                    # Calculate progress
                    domains_in_batch = len(samplesPrompt.split(","))
                    print(f"    Batch {batch_num}/{total_batches}: {domains_in_batch} domains")

                    # Get LLM response
                    try:
                        response, _ = llm.chat(samplesPrompt, conversation_history)

                        # Save results to output file
                        output_file = os.path.join(OUTPUT_DIR, f"{model_name}_EXP{str(EXPERIMENT)}.out")
                        with open(output_file, 'a', encoding='utf-8') as f:
                            f.write(samplesPrompt + "\n")
                            f.write("-" * 15 + "\n")
                            f.write(response + "\n")
                            f.write("*" * 15 + "\n")
                        
                        successful_batches += 1
                        
                    except Exception as e:
                        print(f"    ERROR in batch {batch_num}: {e}")
                        continue
                
                print(f"  Completed: {successful_batches}/{total_batches} batches")
        
        print("\n" + "="*60)
        print("LLM PROCESSING COMPLETED")
        print("="*60)
    
    else:
        print("\nSKIPPING LLM PROCESSING (SEND_REQUEST = False)")
        print("Only analysis will be performed on existing output files.")

    # ========================================================================
    # RESULTS ANALYSIS
    # ========================================================================
    
    print("\n" + "="*60)
    print("STARTING RESULTS ANALYSIS")
    print("="*60)
    
    for llm_idx, llm in enumerate(LLMS, 1):
        model_name = llm.model
        print(f"\n[{llm_idx}/{len(LLMS)}] Analyzing results for: {model_name}")
        print("-" * 50)
        
        # Define paths for analysis
        file_path = os.path.join(OUTPUT_DIR, f"{model_name}_EXP{str(EXPERIMENT)}.out")
        try_again_path = os.path.join(SECOND_TRY_DOMAINS, f"{model_name}_EXP{str(EXPERIMENT)}.json")

        # Check if output file exists
        if not os.path.exists(file_path):
            print(f"  WARNING: Output file not found: {file_path}")
            continue

        # Only analyze if all domains were correctly processed
        domains_classified = ANALYZER.check_domains(file_path=file_path, output_path=try_again_path)
        
        if domains_classified:
            print(f"  All domains classified successfully")
            
            # Determine analysis type based on experiment
            if EXPERIMENT == 3:
                # ========================================================
                # MULTICLASS FAMILY CLASSIFICATION ANALYSIS
                # ========================================================
                print("  Running multiclass family classification analysis...")
                
                try:
                    family_stats, global_stats = ANALYZER.analyze_multiclass(file_path=file_path, size=100000)

                    # Create metrics directory for family-specific results
                    family_metrics_dir = os.path.join(METRICS_DIR, "families/")
                    os.makedirs(family_metrics_dir, exist_ok=True)

                    # Save family-specific metrics (excluding UNKNOWN families)
                    families_saved = 0
                    for family, stats in family_stats.items():
                        if 'UNKNOWN' in family.upper():
                            continue
                        
                        metrics_path = os.path.join(family_metrics_dir, f"{family.upper()}_EXP{str(EXPERIMENT)}.csv")
                        file_exists = os.path.isfile(metrics_path)
                        
                        with open(metrics_path, mode='a') as file:
                            if not file_exists:
                                file.write("model,accuracy,precision,recall,f1,fpr,tpr,mcc,kappa\n")
                            file.write(f"{model_name},{stats['accuracy']:.3f},{stats['precision']:.3f},"
                                    f"{stats['recall']:.3f},{stats['f1']:.3f},{stats['fpr']:.3f},"
                                    f"{stats['tpr']:.3f},{stats['mcc']:.3f},{stats['kappa']:.3f}\n")
                        families_saved += 1

                    # Save global metrics
                    metrics_path = os.path.join(METRICS_DIR, f"GLOBAL_EXP{str(EXPERIMENT)}.csv")
                    file_exists = os.path.isfile(metrics_path)

                    with open(metrics_path, mode='a') as file:
                        if not file_exists:
                            file.write("model,accuracy,precision,recall,f1,fpr,tpr,mcc,kappa\n")
                        file.write(f"{model_name},{global_stats['total_accuracy']:.3f},"
                                f"{global_stats['total_precision']:.3f},{global_stats['total_recall']:.3f},"
                                f"{global_stats['total_f1']:.3f},{global_stats['total_fpr']:.3f},"
                                f"{global_stats['total_tpr']:.3f},{global_stats['total_mcc']:.3f},"
                                f"{global_stats['total_kappa']:.3f}\n")
                    
                    print(f"  Saved metrics for {families_saved} families")
                    print(f"  Global accuracy: {global_stats['total_accuracy']:.3f}")
                    
                except Exception as e:
                    print(f"  ERROR in multiclass analysis: {e}")
                    
            elif EXPERIMENT == 4:
                # ========================================================
                # REAL-WORLD DOMAIN CLASSIFICATION ANALYSIS  
                # ========================================================
                print("  Running real-world domain classification analysis...")
                
                try:
                    metrics = ANALYZER.analyze_real_world_results(file_path=file_path, size=100000)

                    # Save real-world domain metrics
                    metrics_path = os.path.join(METRICS_DIR, f"REAL_WORLD_EXP{str(EXPERIMENT)}.csv")
                    file_exists = os.path.isfile(metrics_path)
                    
                    with open(metrics_path, mode='a') as file:
                        if not file_exists:
                            file.write("model," + metrics.csv_header() + '\n')
                        file.write(model_name + "," + metrics.to_csv() + '\n')
                    
                    print(f"  Real-world classification completed")
                    
                except Exception as e:
                    print(f"  ERROR in real-world analysis: {e}")
                    
            else:
                # ========================================================
                # BINARY CLASSIFICATION ANALYSIS (Experiments 0, 1, 2)
                # ========================================================
                print("  Running binary classification analysis...")
                
                try:
                    (malicious_metrics, benign_metrics, overall_metrics) = ANALYZER.analyze(file_path=file_path, size=100000)
                    
                    # Save overall metrics
                    metrics_path = os.path.join(METRICS_DIR, f"GLOBAL_EXP{str(EXPERIMENT)}.csv")
                    file_exists = os.path.isfile(metrics_path)
                    with open(metrics_path, mode='a') as file:
                        if not file_exists:
                            file.write("model," + overall_metrics.csv_header() + '\n')
                        file.write(model_name + "," + overall_metrics.to_csv() + '\n')
                    
                    # Save malicious domain metrics
                    metrics_path = os.path.join(METRICS_DIR, f"MALICIOUS_EXP{str(EXPERIMENT)}.csv")
                    file_exists = os.path.isfile(metrics_path)
                    with open(metrics_path, mode='a') as file:
                        if not file_exists:
                            file.write("model," + malicious_metrics.csv_header() + '\n')
                        file.write(model_name + "," + malicious_metrics.to_csv() + '\n')
                    
                    # Save benign domain metrics
                    metrics_path = os.path.join(METRICS_DIR, f"BENIGN_EXP{str(EXPERIMENT)}.csv")
                    file_exists = os.path.isfile(metrics_path)
                    with open(metrics_path, mode='a') as file:
                        if not file_exists:
                            file.write("model," + benign_metrics.csv_header() + '\n')
                        file.write(model_name + "," + benign_metrics.to_csv() + '\n')
                    
                    print(f"  Overall accuracy: {overall_metrics.accuracy:.3f}")
                    print(f"  Overall F1-score: {overall_metrics.f1_score:.3f}")
                    
                except Exception as e:
                    print(f"  ERROR in binary analysis: {e}")
                    
        else:
            print(f"  WARNING: Some domains were not classified")
            print(f"  Check: {try_again_path}")
            print(f"  Consider running with SECOND_TRY = True")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETED")
    print("="*60)
    print(f"Results saved in: {METRICS_DIR}")
    print(f"Raw outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()