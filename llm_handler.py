# llm_handler.py

from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import sys

class FastLanguageModel:
    """
    This class downloads and runs a GGUF-quantized model using llama-cpp-python.
    """
    def __init__(self, model_repo_id: str, model_filename: str):
        print("Downloading GGUF model (if not cached)...")
        try:
            self.model_path = hf_hub_download(
                repo_id=model_repo_id,
                filename=model_filename
            )
            print("✅ GGUF Model located.")
        except Exception as e:
            print(f"Error downloading model: {e}")
            sys.exit(1)
        
        print("Loading model with llama-cpp...")
        self.model = Llama(
            model_path=self.model_path,
            n_ctx=2048,
            n_gpu_layers=0,
            verbose=False
        )
        print("✅ Model loaded successfully on CPU.")

    def generate(self, messages: list, max_tokens: int = 150) -> str:
        """
        Generates a response using the model's chat completion method.
        """
        response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens
        )
        return response['choices'][0]['message']['content']

# from llama_cpp import Llama
# from huggingface_hub import hf_hub_download
# import sys

# class FastLanguageModel:
#     """
#     This class downloads and runs a GGUF-quantized model using llama-cpp-python.
#     """
#     def __init__(self, model_repo_id: str, model_filename: str):
#         print("Downloading GGUF model...")
#         try:
#             # Download the GGUF model file from Hugging Face
#             self.model_path = hf_hub_download(
#                 repo_id=model_repo_id,
#                 filename=model_filename
#             )
#             print("✅ GGUF Model downloaded.")
#         except Exception as e:
#             print(f"Error downloading model: {e}")
#             sys.exit(1)
        
#         print("Loading model with llama-cpp...")
#         # Load the model using llama-cpp-python
#         self.model = Llama(
#             model_path=self.model_path,
#             n_ctx=2048,           # Context window size
#             n_gpu_layers=0,       # Set to 0 to run fully on CPU
#             verbose=False         # Set to True to see more logs
#         )
#         print("✅ Model loaded successfully on CPU.")

#     def generate(self, messages: list, max_tokens: int = 150) -> str:
#         """
#         Generates a response using the model's chat completion method.
#         """
#         print(f"\n--- LLM Request ---")
#         print(f"Messages: {len(messages)} messages")
#         print(f"Max tokens: {max_tokens}")
        
#         response = self.model.create_chat_completion(
#             messages=messages,
#             max_tokens=max_tokens
#         )
        
#         print(f"\n--- LLM Response Details ---")
#         # print(f"Full response object: {response}")
#         # print(f"Response type: {type(response)}")
#         print(f"Response keys: {list(response.keys()) if isinstance(response, dict) else 'Not a dict'}")
        
#         content = response['choices'][0]['message']['content']
#         print(content)
#         # print(f"Extracted content: {content}")
#         # print(f"Content length: {len(content)} characters")
        
#         return content

# def run_main_app():
#     """The main logic of your application."""
#     # --- Model Configuration for TinyLlama ---
#     # We are using a repository that contains pre-quantized GGUF files for TinyLlama
#     gguf_repo = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
#     # This is a 4-bit quantized version, a good balance of speed and quality
#     gguf_filename = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

    
#     llm = FastLanguageModel(model_repo_id=gguf_repo, model_filename=gguf_filename)

#     # 1. Get the AI's persona from the user
#     persona_role = "HR Professional"
    
#     # 2. Get the specific job or task from the user
#     job_task = "Create and manage fillable forms for onboarding and compliance"
#     document_titles = [
#         "Learn Acrobat - Create and Convert_1",
#         "Learn Acrobat - Create and Convert_2",
#         "Learn Acrobat - Edit_1",
#         "Learn Acrobat - Edit_2",
#         "Learn Acrobat - Export_1",
#         "Learn Acrobat - Export_2",
#         "Learn Acrobat - Fill and Sign",
#         "Learn Acrobat - Generative AI_1",
#         "Learn Acrobat - Generative AI_2",
#         "Learn Acrobat - Request e-signatures_1",
#         "Learn Acrobat - Request e-signatures_2",
#         "Learn Acrobat - Share_1",
#         "Learn Acrobat - Share_2",
#         "Test Your Acrobat Exporting Skills",
#         "The Ultimate PDF Sharing Checklist"
#     ]

# #     system_prompt = (
# #         "You are an AI assistant that deconstructs user requests into a structured, prioritized analysis for semantic search. "
# #         "Your task is to analyze the provided user profile and available documents to generate a concise, bulleted list under the following headings: "
# #         "*Audience Profile, **Prioritized Needs & Keywords, **Key Constraints, and **Most Relevant Document Themes*. "
# #         "First, infer the audience's traits. Then, based on those traits, derive and rank the key needs and relevant document themes. "
# #         "Your response must be structured, concise, and contain no conversational filler."
# #     )
    
# #     # The user prompt is now clean, containing only the specific data for the task.
# #     user_content = (
# #         f"### User Request:\n"
# #         f"- Persona: {persona_role}\n"
# #         f"- Task: {job_task}\n\n"
# #         f"### Available Documents (for context):\n"
# #         f"- {', '.join(document_titles)}\n\n"
# #         f"### Deconstructed Analysis:"
# # )
    
#     # 3. Create a more effective and generalized system prompt
#     # This gives the AI clear instructions on its role and how to behave.
#     system_prompt = (
#     "You are a precise assistant helping HR professionals design better digital workflows. "
#     "Given a persona and task, write 7 to 10 short, direct suggestions (1 line each) describing creative ways to fulfill that task. "
#     "Each suggestion must be a single line (max 15 words), no bullet points, no numbering, no extra explanation. "
#     "Only output the 7–10 lines. Do not repeat persona or task. Do not give headings. "
#     "Do not write more than 10 lines. Keep it focused and practical."
# )


#     # The user's content is now a structured profile for the LLM to analyze.
#     user_content = (
       
#         # f"Deconstruct the following user request into its core semantic components based on the provided context. "
#         # f"Your output must be a concise, bulleted list under the specified headings.\n\n"
#         # f"### Context:\n"
#         f"- Persona: {persona_role}\n"
#         f"- Available Documents: {', '.join(document_titles)}\n\n"
#         f"### User Task:\n"
#         f"\"{job_task}\"\n\n"
#     #     f"### Deconstructed Analysis:\n"
#     #     f"- *Primary Goal*: [A 7-8 lined, concise sentence describing the main objective , predict yourself]\n"
#      )

    
#     # TinyLlama uses a specific chat template format
#     messages = [
#         {
#             "role": "system",
#             "content": system_prompt
#         },
#         {
#             "role": "user",
#             "content": user_content
#         }
#     ]
    
#     print("\nPROMPT: Plan a 4-day trip for 10 college friends.")
#     response = llm.generate(messages=messages, max_tokens=250)
#     print(f"\nLLM Response:\n{response}")

# if __name__ == '__main__':
#     run_main_app()