from abc import ABC, abstractmethod

class LLM(ABC):
    
    @abstractmethod
    def chat(self, prompt: str, conversation_history: dict = None) -> tuple:
        """
        Send a message to the LLM model and get a response.
        
        This method handles the conversation with the LLM model, maintaining the
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
        pass
    
    @abstractmethod
    def craftConversationHistory(self, user_message: str, assistant_message: str) -> dict:
        """
        Formats user and assistant messages into a structured dictionary.
        
        Args:
            user_message (str): The message sent by the user
            assistant_message (str): The response from the assistant
            
        Returns:
            dict: A dictionary containing messages array and metadata
        """
        pass