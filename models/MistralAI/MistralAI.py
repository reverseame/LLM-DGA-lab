import requests
import os
from time import sleep
import time
from utils.config import Config
from models.LLM import LLM

class MistralAI(LLM):
    """
    A class to interact with the MistralAI chat API.
    
    This class provides methods to send messages to a MistralAI model and maintain
    conversation history.
    
    Attributes:
        url (str): The endpoint URL for the MistralAI API
        headers (dict): HTTP headers for API requests
        model (str): The name of the MistralAI model to use
    """

    def __init__(self, model="ministral-3b-latest",url="https://api.mistral.ai/v1/chat/completions"):
        """
        Initialize a new MistralAI chat instance.
        
        Args:
            model (str, optional): The model name to use. Defaults to "gpt-4o-mini"
        """
        self.url = url
        config = Config()
        api_key = config.get_value("API_KEY_MISTRALAI")
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': 'Bearer '+ api_key
        }
        self.model = model
    
    def chat(self, prompt: str, conversation_history: dict = None) -> tuple:
        """
        Send a message to the MistralAI model and get a response.
        
        This method handles the conversation with the MistralAI model, maintaining the
        conversation history and formatting messages appropriately.
        
        Args:
            prompt (str): The user's message to send to the model
            conversation_history (dict, optional): Previous conversation containing
                a "messages" key with the message list. Defaults to None.
        
        Returns:
            tuple: A tuple containing (response_content, updated_messages), where:
                - response_content (str): The model's response text
                - updated_messages (dict): The full conversation history including
                  the new messages
        """
        # If no conversation history is provided, initialize an empty dictionary with messages list
        if conversation_history is None:
            conversation_history = {"messages": []}
        
        # Create a copy of the existing messages and add the new prompt
        messages = conversation_history["messages"].copy()
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Prepare the request data
        data = {
            "model": self.model,
            "temperature": 0.0,
            "messages": messages
        }

        ok = False
        while not ok:
            # Send request to the API and get response
            first = time.time()
            response = requests.post(self.url, headers=self.headers, json=data)
            second = time.time()
            duration = second - first

            script_dir = os.path.dirname(os.path.abspath(__file__))
            target_dir = os.path.join(script_dir, "logger")
            os.makedirs(target_dir, exist_ok=True)

            try:
                # Log
                log_name = "responses_"+self.model+".log"
                log_path = os.path.join(target_dir, log_name)
                with open(log_path, "a") as f:
                    f.write(str(response.json()))
                    f.write("\n"+("-"*15)+"\n")
                # Get answer to the prompt
                response_content = response.json()['choices'][0]['message']['content']
                                # Log time
                log_name_time = "responses_time_"+self.model+".log"
                log_path_time = os.path.join(target_dir, log_name_time)
                with open(log_path_time, "a") as f:
                    f.write(f"{duration:.6f}\n")
                ok = True
            except Exception as e:
                print(e)
                print("Trying again in 15 seconds to send the request...")
                sleep(15)

        # Add the assistant's response to the messages
        messages.append({
            "role": "assistant",
            "content": response_content
        })
        
        # Create updated conversation history
        updated_conversation = {
            "messages": messages
        }
        
        # Return both the response and the updated conversation history
        return response_content, updated_conversation
    
    def craftConversationHistory(self, user_message: str, assistant_message: str) -> dict:
        """
        Formats user and assistant messages into a structured dictionary.
        
        Args:
            user_message (str): The message sent by the user
            assistant_message (str): The response from the assistant
            
        Returns:
            dict: A dictionary containing messages array and metadata
                {
                    "messages": [
                        {"role": "user", "content": "user message"},
                        {"role": "assistant", "content": "assistant response"}
                    ]
                }
        """
        conversation = {
            "messages": [
                {
                    "role": "user",
                    "content": user_message
                },
                {
                    "role": "assistant",
                    "content": assistant_message
                }
            ]
        }
        
        return conversation