import os
from typing import List, Set, Tuple
from pathlib import Path
import random

class PromptGenerator:
    """
    A class to generate prompts by combining different sections and managing domains.
    """

    def __init__(self, base_directory: str, families_directory: str, legitimate_domains_file: str):
        """
        Initialize a new PromptGenerator instance.

        Args:
            base_directory (str): Path to the directory containing prompt sections
            families_directory (str): Path to the directory containing domain families
            legitimate_domains_file (str): Path to the file containing legitimate domains
        """
        self.base_dir = Path(base_directory)
        self.families_dir = Path(families_directory)
        self.legitimate_domains_file = Path(legitimate_domains_file)
        self.sections = {
            'starting': self.base_dir / 'StartingPoints',
            'middle': self.base_dir / 'Prompt4Experiments',
            'final': self.base_dir / 'EndingPoints'
        }
        self.validate_directory_structure()
        self.used_domains: Set[str] = set()
        self.legitimate_domains: List[str] = self.load_legitimate_domains()

    def validate_directory_structure(self):
        """
        Validate that all required directories and files exist.

        Raises:
            ValueError: If any required directory or file is missing
        """
        for section, path in self.sections.items():
            if not path.exists() or not path.is_dir():
                raise ValueError(f"Subdirectory '{section}' not found")
        if not self.families_dir.exists() or not self.families_dir.is_dir():
            raise ValueError(f"Families directory not found")
        if not self.legitimate_domains_file.exists() or not self.legitimate_domains_file.is_file():
            raise ValueError(f"Legitimate domains file not found")

    def load_legitimate_domains(self) -> List[str]:
        """
        Load legitimate domains from the specified file.

        Returns:
            List[str]: List of legitimate domains
        """
        with open(self.legitimate_domains_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def read_family_domains(self):
        """
        Read all domain family files and return their contents.

        Returns:
            dict: Dictionary mapping family names to lists of domains
        """
        families = {}
        for file_path in self.families_dir.glob("*.csv"):
            family_name = file_path.stem
            with open(file_path, 'r', encoding='utf-8') as f:
                domains = [line.strip() for line in f if line.strip()]
            families[family_name] = domains
        return families

    def generate_test_domains(self, num_test_domains_per_family: int, num_legitimate_domains: int) -> List[str]:
        """
        Generate test domains by interleaving domains from families and legitimate domains.

        Args:
            num_test_domains_per_family (int): Number of test domains to select per family
            num_legitimate_domains (int): Number of legitimate domains to include

        Returns:
            List[str]: List of interleaved test domains
        """
        families = self.read_family_domains()
        family_domains = []

        # Select test domains from each family
        for domains in families.values():
            available_domains = [d for d in domains if d not in self.used_domains]
            selected = random.sample(available_domains, num_test_domains_per_family)
            family_domains.append(selected)
            self.used_domains.update(selected)

        # Select legitimate domains
        available_legitimate_domains = [d for d in self.legitimate_domains if d not in self.used_domains]
        selected_legitimate = random.sample(available_legitimate_domains, num_legitimate_domains)

        # Calculate legitimate domains per iteration
        total_iterations = num_test_domains_per_family
        legitimate_per_iteration = num_legitimate_domains // total_iterations
        leftover_legitimate = num_legitimate_domains % total_iterations

        interleaved_domains = []

        # Interleave family domains and legitimate domains
        for i in range(total_iterations):
            # Add one domain from each family
            for family_list in family_domains:
                interleaved_domains.append(family_list[i])

            # Add legitimate domains for this iteration
            for _ in range(legitimate_per_iteration):
                if selected_legitimate:
                    interleaved_domains.append(selected_legitimate.pop(0))

            # Distribute leftover legitimate domains
            if leftover_legitimate > 0 and selected_legitimate:
                interleaved_domains.append(selected_legitimate.pop(0))
                leftover_legitimate -= 1

        return interleaved_domains

    def read_prompt_file(self, section: str, filename: str) -> str:
        """
        Read and return the content of a prompt file.

        Args:
            section (str): The section where the file is located
            filename (str): The name of the file to read

        Returns:
            str: The content of the prompt file

        Raises:
            FileNotFoundError: If the specified file doesn't exist
        """
        file_path = self.sections[section] / filename
        if not file_path.exists():
            raise FileNotFoundError(f"File {filename} not found")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def create_prompt_from_domain_list(self, domains: List[str]) -> str:
        """
        Converts a list of domains into a comma-separated string.

        Args:
            domains (list): List of strings representing domains

        Returns:
            str: String with all domains separated by commas
        """
        return ", ".join(domains)
    
    def generate_training_samples(self, num_train_samples: int) -> str:
        """
        Generate training samples by randomly selecting domains from each family.
        Note: This function is included for compatibility but not used in experiments 1 and 2.

        Args:
            num_train_samples (int): Number of training samples to select per family

        Returns:
            str: Formatted string containing selected domains grouped by family
        """
        families = self.read_family_domains()
        family_samples = []
        
        for family_name, domains in families.items():
            available_domains = [d for d in domains if d not in self.used_domains]
            selected_domains = random.sample(available_domains, min(num_train_samples, len(available_domains)))
            self.used_domains.update(selected_domains)
            family_samples.append(f"{family_name}: {', '.join(selected_domains)};")
        
        return '\n'.join(family_samples)

    def generate_prompt(self, starting_prompt: str, middle_prompts: List[str], 
                        final_prompt: str, num_train_samples: int, 
                        num_test_domains: int, num_legitimate_domains: int) -> Tuple[str, List[str]]:
        """
        Generate a complete prompt by combining sections and domain samples.

        Args:
            starting_prompt (str): Filename of the starting prompt
            middle_prompts (List[str]): List of filenames for middle section prompts
            final_prompt (str): Filename of the final prompt
            num_train_samples (int): Number of training samples per family (0 for experiments 1 and 2)
            num_test_domains (int): Number of test domains per family
            num_legitimate_domains (int): Number of legitimate domains to include

        Returns:
            Tuple[str, List[str]]: The complete prompt and list of test domains
        """
        self.used_domains.clear()  # Reset used domains at the start
        combined_prompt = []
        
        combined_prompt.append(self.read_prompt_file('starting', starting_prompt))
        
        for mid_prompt in middle_prompts:
            combined_prompt.append(self.read_prompt_file('middle', mid_prompt))
        
        if num_train_samples > 0:
            combined_prompt.append(self.generate_training_samples(num_train_samples))
            
        combined_prompt.append(self.read_prompt_file('final', final_prompt))
        
        # Generate test domains
        test_domains = self.generate_test_domains(num_test_domains, num_legitimate_domains)
        
        return '\n\n'.join(combined_prompt), test_domains