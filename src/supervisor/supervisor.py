import asyncio
from typing import Optional, Dict, List

from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain.messages import ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.tools import tool
from fastmcp import Client
import json


class AgentState(MessagesState):
    """
    Extends MessagesState to include custom fields for the agent's memory.
    
    Attributes:
        original_image_path (Optional[str]): Path to the original user-uploaded image.
        preprocessed_image_path: Optional[str]: Path to the preprocessed image.
        working_image_path (Optional[str]): Path to the last modified/processed image.
        target_language (Optional[str]): Target language for translation.
        preprocessing_settings (Dict[str, float]): Applied preprocessing values.
        
        text_blocks_path (Optional[str]): Path to the detected layout JSON file (Pass-by-Reference).
        document_class (Optional[str]): Detected document class.
        font_json_path (Optional[str]): Path to the detected font JSON file (Pass-by-Reference).
        translated_blocks_path (Optional[str]): Path to the JSON file containing translations (Pass-by-Reference).
        
        rendered_document_path (Optional[str]): Path to final rendered document.
        layout_coordinates: Optional[List[Dict]]: Detected layout coordinates (crop offsets).
    """
    original_image_path: Optional[str]
    preprocessed_image_path: Optional[str]
    working_image_path: Optional[str]
    target_language: Optional[str]
    preprocessing_settings: Dict[str, float]
    text_blocks_path: Optional[str]
    document_class: Optional[str]
    font_json_path: Optional[str]
    translated_blocks_path: Optional[str]
    rendered_document_path: Optional[str]
    layout_coordinates: Optional[List[Dict]]


class LangChainSupervisor:
    """
    Manages the agent's logic, state, and execution graph using LangGraph.
    """
    def __init__(self):
        """
        Initializes the LangChain supervisor, LLM, MCP client, system prompt,
        and compiles the execution graph.
        """
        self.model = ChatOllama(model="qwen3", temperature=0.0, max_tokens=3000)
        self.mcp_client = Client("http://localhost:8000/mcp")

        self.system_prompt = """You are the Smart Translator Agent.
        GOAL: Translate document images, classify them, and answer questions.

        CORE GUIDELINES:
        1. YOU decide the tool sequence based on the user's request.
        2. STATE CHECK: 
           - Before starting analysis, verify you have BOTH an Image AND a Target Language.
           - If user says "proceed" but Language is missing -> STOP and ask.
        3. FLOW & PROACTIVITY:
           - With Image + Language loaded: OFFER/PAUSE & ASK Preprocessing (Contrast/Brightness).
           - Show current values. Ask: "Adjust more or PROCEED?"
        4. MISSING INFO: Ask only for what is strictly missing.
        5. RAG / QUESTIONS: 
           - Use 'query_translated_document' for content questions.
           - OUTPUT RULE: When answering a question, ONLY provide the answer text. DO NOT repeat the file path or document class.
        6. PIPELINE COMPLETION (Only once):
           - WHEN (and ONLY when) the 'render_document' tool has just finished successfully:
           - You MUST run 'detect_document_class' (if not done).
           - You MUST explicitly report: 1. The Document Class (with confidence) and 2. The Final Image Path.
        """

        self.current_state: AgentState = self._get_initial_state()
        self._setup_tools_and_graph()
    
    def _get_initial_state(self) -> AgentState:
        """
        Generates the initial state dictionary for a new graph invocation.
        """
        return {
            "messages": [],
            "original_image_path": None,
            "preprocessed_image_path": None,
            "working_image_path": None,
            "target_language": None,
            "preprocessing_settings": {},
            "text_blocks_path": None,
            "document_class": None,
            "font_json_path": None,
            "translated_blocks_path": None,
            "rendered_document_path": None,
            "layout_coordinates": None
        }
    
    def _run_tool_sync(self, tool_name: str, tool_args: dict):
        """
        Synchronously runs a tool via the FastMCP client.
        Wraps async calls for synchronous execution.
        """
        async def with_client():
            async with self.mcp_client:
                try:
                    result = await self.mcp_client.call_tool(tool_name, tool_args)
                    if result.is_error:
                        error_text = result.content[0].text if result.content else "Unknown error"
                        return {"status": "error", "message": error_text}
                    return result.data or {"status": "success"}
                except Exception as e:
                    return {"status": "error", "message": str(e)}
        return asyncio.run(with_client())


    def _create_tools(self):
        """
        Creates tool definitions.
        Note: We use 'pass-by-reference' (file paths) for large data to save LLM context tokens.
        """
        @tool
        def ping() -> dict:
            """Test server connectivity."""
            pass

        @tool
        def set_target_language(language: str) -> dict:
            """
            Sets the target translation language (e.g. 'German', 'English').
            
            NEXT STEP STRATEGY:
            - If an image is ALREADY loaded: Do NOT ask for the image again. Instead, suggest 'apply_preprocessing' or ask if the user wants to 'proceed' with these settings.
            - If NO image is loaded: Ask the user to upload one.
            """

        @tool
        def load_image(path: str) -> dict:
            """
            Loads an image from a local file path.
            
            EFFECT: Resets the entire processing pipeline (clears old layouts, fonts, translations).
            STATE UPDATES: Sets 'original_image_path' and 'working_image_path' in the memory.
            
            CRITICAL: This is the FIRST step for any document task.
            IF MISSING: If the user wants to translate but no image is loaded, ask them for the file path.
            NEXT STEP STRATEGY:
            - If language is ALREADY set: Offer 'apply_preprocessing' immediately.
            - If language is MISSING: Ask the user for the target language.
            """
            pass

        @tool
        def capture_from_webcam(camera_index: int = 0) -> dict:
            """
            Capture image from webcam (SPACE=capture, ESC=cancel).
            PIPELINE BEHAVIOR: Resets pipeline.
            """
            pass

        @tool
        def apply_preprocessing(
            image_path: str, 
            contrast: float = 1.0, 
            brightness: float = 1.0, 
            sharpness: float = 0.0,
            denoise: float = 0.0
        ) -> dict:
            """
            Adjusts image visual properties.
            
            IMPORTANT DEFAULTS (Neutral values):
            - contrast: 1.0 (Use 1.0 for NO change. 0.0 makes it gray!)
            - brightness: 1.0 (Use 1.0 for NO change. 0.0 makes it black!)
            - sharpness: 0.0 (Use 0.0 for NO change)
            - denoise: 0.0 (Use 0.0 for NO change)
            
            TIMING: Call this ONLY after 'load_image' and BEFORE 'detect_layout'.
            INTERACTION: ASK the user if they want to use this before proceeding to analysis.
            USAGE:
            - ONLY include parameters the user explicitly changed.
            - Tell the user the new settings applied.
            - Then ask: "Do you want to adjust more values or PROCEED to analysis?"
            """
            pass

        @tool
        def detect_layout(image_path: str) -> dict:
            """
            Analyze document structure and extract text blocks with OCR.
            
            REQUIRES: An image must be loaded ('load_image').
            ENABLES: 'detect_fonts', 'translate_layout_file', 'detect_document_class'.
            OUTPUT: Returns a 'text_blocks_path' needed for translation.
            """
            pass

        @tool
        def detect_document_class(image_path: str) -> dict:
            """
            Classifies the document type (e.g. 'Invoice', 'Email', 'Resume').
            
            TIMING:
            - Run this automatically as part of the translation pipeline.
            - REQUIRED before generating the final success message to the user.
            """
            pass

        @tool
        def detect_fonts(text_blocks_path: str) -> dict:
            """
            Identifies fonts and sizes in the detected text blocks.
            
            REQUIRES: 'detect_layout' must be finished first (needs 'text_blocks_path').
            ENABLES: Realistic rendering in 'render_document'.
            """
            pass

        @tool
        def translate_layout_file(json_path: str) -> dict:
            """
            Translates the text content inside the provided layout JSON file.
            
            REQUIRES: 
            1. 'detect_layout' complete (needs 'json_path').
            2. 'set_target_language' complete (needs target language).
            
            OUTPUT: Returns 'translated_blocks_path'.
            """
            pass

        @tool
        def render_document(
            original_image_path: str, 
            translated_json_path: str
        ) -> dict:
            """
            Generates the final translated image file.
            
            REQUIRES: 
            1. 'detect_fonts' complete.
            2. 'translate_layout_file' complete (needs 'translated_json_path').
            
            TIMING: This is usually the FINAL step of the pipeline.
            """
            pass

        @tool
        def query_translated_document(query: str) -> dict:
            """
            Search the TRANSLATED document text to answer questions (RAG).
            
            WHEN TO USE: 
            - Use this for ANY question that might be related to the document content.
            - Examples: "What quotes to mark?", "What is the total?", "Summarize this".
            - Even if the question seems vague, try this tool first.
            
            REQUIRES: 
            - 'translate_layout_file' MUST be completed first (so we search the translated text).
            - If translation is not possible (e.g. no language set), it falls back to original text.
            
            OUTPUT INSTRUCTION:
            - Provide a direct, concise answer to the user's question.
            - Do NOT add meta-information like file paths or document types in the result.
            """
            pass

        
        return [
            ping, set_target_language, load_image, capture_from_webcam,
            apply_preprocessing, detect_layout, detect_document_class,
            detect_fonts, translate_layout_file, render_document, query_translated_document
        ]

    def call_model(self, state: AgentState):
            """
            Graph node: Invokes the LLM. Injects a state summary into the system prompt.
            """
            state_summary = ["\n\n--- CURRENT SESSION STATE (FOR YOUR INFORMATION) ---"]
            
            if state.get("original_image_path"):
                state_summary.append(f"STATUS: Original image uploaded. (Path: {state['original_image_path']})")
            else:
                state_summary.append("STATUS: No image is loaded yet.")

            if state.get("working_image_path"):
                state_summary.append(f"STATUS: Current working image ready. (Path: {state['working_image_path']})")

            if state.get("preprocessed_image_path"):
                state_summary.append(f"STATUS: Preprocessed image available for rendering background. (Path: {state['preprocessed_image_path']})")
            else:
                state_summary.append("STATUS: No specific preprocessed image (using original/working image).")

            if state.get("target_language"):
                state_summary.append(f"STATUS: Target language IS set. (Language: {state['target_language']})")
            else:
                state_summary.append("STATUS: Target language is NOT set.")
                
            if state.get("preprocessing_settings"):
                settings = state['preprocessing_settings']
                state_summary.append(f"STATUS: Preprocessing settings applied: {settings}")
            else:
                state_summary.append("STATUS: No preprocessing settings applied yet.")

            if state.get("text_blocks_path"):
                state_summary.append(f"STATUS: Layout detection IS complete. (File: {state['text_blocks_path']})")
            else:
                state_summary.append("STATUS: Layout detection is NOT complete.")

            if state.get("layout_coordinates"):
                 state_summary.append("STATUS: Layout coordinates are in memory.")

            if state.get("document_class"):
                state_summary.append(f"STATUS: Document class IS detected. (Class: {state['document_class']})")
            
            if state.get("font_json_path"):
                state_summary.append(f"STATUS: Font detection IS complete. (Ready for rendering)")
            else:
                state_summary.append("STATUS: Font detection is NOT complete.")
            
            if state.get("translated_blocks_path"):
                state_summary.append(f"STATUS: Text blocks ARE translated. (File: {state['translated_blocks_path']})")
            else:
                state_summary.append(f"STATUS: Text blocks are NOT translated.")

            if state.get("rendered_document_path"):
                state_summary.append(f"STATUS: Final document IS rendered. (Path: {state['rendered_document_path']})")

            updated_system_prompt = self.system_prompt + "\n".join(state_summary)
            
            messages = [msg for msg in state["messages"] if not isinstance(msg, SystemMessage)]
            updated_messages = [SystemMessage(content=updated_system_prompt)] + messages

            response = self.model_with_tools.invoke(updated_messages)
            return {"messages": [response]}
    
    def should_continue(self, state: AgentState) -> str:
        """Check if LLM returned a tool call."""
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "action"
        return END

    def call_tool(self, state: AgentState) -> dict:
        """
        Executes tools and updates the AgentState with results (specifically file paths).
        """
        last_message = state["messages"][-1]
        tool_call = last_message.tool_calls[0]
        
        called_tool = next((t for t in self.tools if t.name == tool_call["name"]), None)
        
        if not called_tool:
            return {
                "messages": [ToolMessage(
                    f"Error: Tool '{tool_call['name']}' not found.",
                    tool_call_id=tool_call["id"]
                )]
            }
        
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})
        state_update = {}
        
        
        if tool_name == "ping":
            response_dict = self._run_tool_sync("ping", {})
        
        elif tool_name == "set_target_language":
            response_dict = self._run_tool_sync("set_target_language", tool_args)
            if response_dict.get("status") == "success":
                state_update["target_language"] = response_dict.get("language")
        
        elif tool_name == "load_image":
            response_dict = self._run_tool_sync("load_image", tool_args)
            if response_dict.get("status") == "success":
                image_path = response_dict.get("path")
                state_update["original_image_path"] = image_path
                state_update["preprocessed_image_path"] = None
                state_update["working_image_path"] = image_path
                state_update["target_language"] = None
                state_update["preprocessing_settings"] = {}
                state_update["rendered_document_path"] = None  
                state_update["translated_blocks_path"] = None       
                state_update["text_blocks_path"] = None             
                state_update["font_json_path"] = None
                state_update["document_class"] = None
                state_update["layout_coordinates"] = None
        
        elif tool_name == "capture_from_webcam":
            response_dict = self._run_tool_sync("capture_from_webcam", tool_args)
            if response_dict.get("status") == "success":
                image_path = response_dict.get("path")
                state_update["original_image_path"] = image_path
                state_update["preprocessed_image_path"] = None
                state_update["working_image_path"] = image_path
                state_update["target_language"] = None
                state_update["preprocessing_settings"] = {}
                state_update["rendered_document_path"] = None  
                state_update["translated_blocks_path"] = None       
                state_update["text_blocks_path"] = None             
                state_update["font_json_path"] = None
                state_update["document_class"] = None
                state_update["layout_coordinates"] = None

        elif tool_name == "apply_preprocessing":
            current_settings = state.get("preprocessing_settings")
            if not current_settings:
                current_settings = {"contrast": 1.0, "brightness": 1.0, "sharpness": 0.0, "denoise": 0.0}

            new_args = tool_args.copy()
            current_settings.update(new_args)
            current_settings["image_path"] = state.get("original_image_path")
            
            response_dict = self._run_tool_sync("apply_preprocessing", current_settings)
            
            if response_dict.get("status") == "success":
                state_update["working_image_path"] = response_dict.get("processed_image_path")
                state_update["preprocessed_image_path"] = response_dict.get("processed_image_path")
                state_update["preprocessing_settings"] = response_dict.get("applied_settings", {})

        elif tool_name == "detect_layout":
            img = state.get("working_image_path") or state.get("original_image_path")
            
            if not img:
                response_dict = {"status": "error", "message": "FAILED: No image path found. Please tell the user to upload an image first."}
            else:
                response_dict = self._run_tool_sync("detect_layout", {"image_path": img})
                
                if response_dict.get("status") == "success":
                    state_update["text_blocks_path"] = response_dict.get("text_blocks_path") 
                    state_update["working_image_path"] = response_dict.get("extracted_document_path")
                    state_update["layout_coordinates"] = response_dict.get("coordinates")
                    response_dict["message"] = "SUCCESS: Layout detected. Call 'translate_layout_file' next."

        elif tool_name == "translate_layout_file":
            target_lang = state.get("target_language")
            json_path_arg = tool_args.get("json_path") or state.get("text_blocks_path")

            mcp_args = {
                "json_path": json_path_arg,
                "target_language": target_lang
            }
    
            response_dict = self._run_tool_sync("translate_layout_file", mcp_args)

            if response_dict.get("status") == "success":
                state_update["translated_blocks_path"] = response_dict.get("translated_json_path")
                

        elif tool_name == "detect_document_class":
            response_dict = self._run_tool_sync("detect_document_class", tool_args)
            
            if response_dict.get("status") == "success":
                doc_class = response_dict.get("document_class")
                confidence = response_dict.get("confidence", 0.0) 
                
                state_update["document_class"] = doc_class
                
                response_dict["message"] = (
                    f"SUCCESS: Document classified as '{doc_class}' "
                    f"with confidence {confidence:.1%}."
                )

        elif tool_name == "detect_fonts":
            args = tool_args.copy() if tool_args else {}
            
            if not args.get("text_blocks_path"):
                 args["text_blocks_path"] = state.get("text_blocks_path")
            if not args.get("image_path"):
                args["image_path"] = state.get("working_image_path") or state.get("original_image_path")

            if not args.get("text_blocks_path") or not args.get("image_path"):
                response_dict = {
                    "status": "error",
                    "message": "Missing arguments for detect_fonts. Need ocr_json_path and image_path."
                }
            else:
                response_dict = self._run_tool_sync("detect_fonts", args)
            
            if response_dict.get("status") == "success":
                state_update["font_json_path"] = response_dict.get("font_json_path")
                
                response_dict["message"] = "SUCCESS: Font detection complete. Font data saved to JSON."


        elif tool_name == "render_document":
            if state.get("preprocessed_image_path") is None:
                original_img = tool_args.get("original_image_path") or state.get("original_image_path")
            else:
                original_img = state.get("preprocessed_image_path")
            cropped_img = state.get("working_image_path")
            has_translation = state.get("translated_blocks_path")
            has_fonts = state.get("font_json_path")
            layout_data = state.get("layout_coordinates") 

            if not all([original_img, cropped_img, has_translation, has_fonts, layout_data]):
                missing = []
                if not original_img: missing.append("Original Image")
                if not cropped_img: missing.append("Cropped Image")
                if not has_translation: missing.append("Translation (run translate_layout_file)")
                if not has_fonts: missing.append("Font Info (run detect_fonts)")
                if not layout_data: missing.append("Layout Data")

                print(f"Cannot render. Missing data in state: {', '.join(missing)}.")
                response_dict = {
                    "status": "error", 
                    "message": f"Cannot render. Missing data in state: {', '.join(missing)}. Please run layout detection and translation first or the missing tools first."
                }
            else:
                trans_text_json = has_fonts                  
                args = {
                    "cropped_image_path": cropped_img,
                    "original_image_path": original_img,
                    "translated_json_path": trans_text_json,  
                    "layout_json_path": layout_data
                    }
                
                response_dict = self._run_tool_sync("render_document", args)

            if response_dict.get("status") == "success":
                state_update["rendered_document_path"] = response_dict.get("rendered_document_path")


        elif tool_name == "query_translated_document":
            json_path = state.get("translated_blocks_path")
            
            if not json_path:
                json_path = state.get("text_blocks_path")

            if not json_path:
                 response_dict = {
                     "status": "error", 
                     "message": "I cannot answer questions yet. Please let me analyze and translate the document first."
                 }
            else:
                mcp_args = {
                    "json_path": json_path,
                    "query": tool_args.get("query")
                }
                response_dict = self._run_tool_sync("query_translated_document", mcp_args)
            
            if response_dict.get("status") == "success":
                ans = response_dict.get("answer", "No answer found.")
                response_dict["message"] = f"RAG ANSWER: {ans}"
                
        else:
            response_dict = {"status": "error", "message": f"Tool logic for '{tool_name}' not implemented in call_tool."}
        
        
        if response_dict.get("status") == "success":
            response_content = response_dict.get('message', 'Tool successful')
        elif response_dict.get("status") == "cancelled":
            response_content = response_dict.get('message', 'Operation was cancelled')
        else:
            response_content = f"Error: {response_dict.get('message', 'unknown')}"

        state_update["messages"] = [ToolMessage(content=response_content, tool_call_id=tool_call["id"])]
        return state_update


    def _setup_tools_and_graph(self):
        """
        Initializes the tools list, binds them to the model,
        and compiles the LangGraph workflow.
        """
        self.tools = self._create_tools()
        self.model_with_tools = self.model.bind_tools(self.tools)
        
        workflow = StateGraph(AgentState)
        workflow.add_node("llm", self.call_model)
        workflow.add_node("action", self.call_tool)
        workflow.add_edge(START, "llm")
        workflow.add_conditional_edges("llm", self.should_continue)
        workflow.add_edge("action", "llm")
        
        self.runnable = workflow.compile()


    def process_user_message(self, message: str) -> AgentState:
        """
        Main entry point to process a user's message through the LangGraph.
        """
        cleaned_message = message.strip('\'" ')

        initial_graph_state = self._get_initial_state()
        
        for key in ["target_language", "original_image_path", "working_image_path", 
                   "preprocessing_settings", "text_blocks_path", "document_class", 
                   "font_json_path", "translated_blocks_path", "rendered_document_path", 
                   "layout_coordinates"]:
            initial_graph_state[key] = self.current_state.get(key)
        
        initial_graph_state["messages"] = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=cleaned_message)
        ]

        final_state = self.runnable.invoke(initial_graph_state)

        self.current_state = final_state.copy()

        return final_state