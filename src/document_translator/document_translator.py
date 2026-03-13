from typing import List, Dict
import json
import shutil
import os

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

class DocumentTranslator:
    """
    Handles text translation using a local Ollama LLM.
    
    This class supports translating raw text lists (batch mode) as well as
    processing JSON files containing OCR data, maintaining the original file structure.
    
    Attributes
    ----------
    debug : bool
        If True, prints detailed processing logs.
    llm : ChatOllama
        The configured LangChain Ollama instance.
    system_prompt : SystemMessage
        The system instruction to enforce strict translation output.
    """
    def __init__(self, 
                 model_name: str = "qwen3",
                 debug: bool = False):
        """
        Initializes the translator with a local Ollama model.

        Parameters
        ----------
        model_name : str, optional
            The Ollama model tag to use (e.g., "llama3.2", "qwen3"), by default "qwen3".
        debug : bool, optional
            Enable debug logging, by default False.
        """
        self.debug = debug
        self.llm = ChatOllama(model=model_name, 
                                temperature=0.0,
                                )
        
        # System prompt to ensure the LLM outputs ONLY the translation
        self.system_prompt = SystemMessage(
            content=(
                "You are an expert translator. Your sole task is to translate the "
                "user's text into the specified target language. "
                "Do NOT output anything else. Do not add apologies, "
                "commentary, or any surrounding text. "
                "Only output the raw, translated text."
            )
        )
        if self.debug: print("DocumentTranslator (Ollama) initialized.")

    def _translate_batch(self, texts: List[str], target_language: str) -> List[str]:
        """
        Translates a list of text strings in batch mode.

        Parameters
        ----------
        texts : List[str]
            A list of strings to be translated.
        target_language : str
            The target language (e.g., "German", "English").

        Returns
        -------
        List[str]
            A list of translated strings corresponding to the input.
            Returns empty strings for items that failed translation.
        """
        if not texts:
            return []

        if self.debug: print(f"INFO: Translating {len(texts)} text blocks to '{target_language}'...")
        
        # Prepare batch messages: [SystemPrompt, HumanMessage] for each input text
        messages_batch = [
            [
                self.system_prompt,
                HumanMessage(
                    content=f"Translate the following text to {target_language}: {text}"
                )
            ]
            for text in texts
        ]

        try:
            results = self.llm.batch(messages_batch)
            translated_texts = [res.content for res in results]
            if self.debug: print(f"INFO: Batch translation complete.")
            return translated_texts
        except Exception as e:
            if self.debug: print(f"ERROR during batch translation: {e}")
            # Return empty strings to maintain list length integrity
            return [""] * len(texts)

   
    def translate_json_file(self, json_path: str, target_language: str) -> Dict:
        """
        Reads a JSON file (e.g., OCR output), translates the text content, 
        and saves a new JSON file preserving the structure.

        This method specifically looks for keys like 'rec_texts' (from standard OCR outputs)
        to identify translatable content.

        Parameters
        ----------
        json_path : str
            Path to the source JSON file.
        target_language : str
            The target language for translation.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing:
            - status: "success" or "error"
            - message: Description of the result
            - translated_json_path: Path to the newly created file (on success)
        """
        if self.debug: print(f"INFO: Processing translation for file: {json_path}")

        if not os.path.exists(json_path):
            return {"status": "error", "message": f"File not found: {json_path}"}

        try:
            # Load Data
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # BackUp Input data
            output_dir = os.path.dirname(json_path)
            debug_dir = os.path.join(output_dir, ".backups")
            os.makedirs(debug_dir, exist_ok=True)
            backup_filename = f"input_doc_translator_{os.path.basename(json_path)}"
            backup_path = os.path.join(debug_dir, backup_filename)
            
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            if self.debug: print(f"INFO: Saved Input JSON to: {backup_path}")

            # Extract Text
            texts_to_translate = []
            # Standard OCR format with 'rec_texts'
            if isinstance(data, dict) and "rec_texts" in data:
                texts_to_translate = data["rec_texts"]
            # List of items (fallback for other formats)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, list) and len(item) >= 2:
                        texts_to_translate.append(item[1][0])
                    elif isinstance(item, dict) and "transcription" in item:
                        texts_to_translate.append(item["transcription"])

            if not texts_to_translate:
                return {"status": "error", "message": "No extractable text found in JSON."}

            # Translate via batch
            translated_texts = self._translate_batch(texts_to_translate, target_language)

            # If it matches the 'rec_texts' format, replace content in-place
            # This allows the ImageRenderer to use the file directly without modification.
            if isinstance(data, dict) and "rec_texts" in data:
                data["rec_texts"] = translated_texts
                data["original_texts"] = texts_to_translate # Backup
            else:
                # Fallback 
                data["translated_texts"] = translated_texts

            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            if self.debug: print(f"INFO: Saved translated JSON to {json_path}") 

            return {
                "status": "success",
                "message": "Translation successful.",
                "original_path": json_path,
                "translated_json_path": json_path
            }

        except Exception as e:
            if self.debug: print(f"ERROR in translate_json_file: {e}")
            return {"status": "error", "message": str(e)}

# Test
if __name__ == "__main__":
    # Path might need to be updated
    TEST_FILE = "example_files/layout_detector/get_ocr/ocr_result.json" 
    TARGET_LANG = "German" 

    translator = DocumentTranslator(model_name="qwen3", debug=True)
    result = translator.translate_json_file(TEST_FILE, TARGET_LANG)

    # Show result
    print("\n" + "="*30)
    print("ERGEBNIS:")
    print(json.dumps(result, indent=2))
    print("="*30)
    
    if result["status"] == "success":
        print(f"JUHU")