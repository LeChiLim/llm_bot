from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
from pydantic import BaseModel
import os
import requests
import json
import re

THINK_START_TAG = "<think>"
THINK_END_TAG = "</think>"

# --- Helper function for citation text insertion ---
def _insert_citations(text: str, citations: list[str]) -> str:
    """
    Replace citation markers [n] in text with markdown links to the corresponding citation URLs.

    Args:
        text: The text containing citation markers like [1], [2], etc.
        citations: A list of citation URLs, where index 0 corresponds to [1] in the text

    Returns:
        Text with citation markers replaced with markdown links
    """
    if not citations or not text:
        return text

    pattern = r"\[(\d+)\]"

    def replace_citation(match_obj):
        try:
            num = int(match_obj.group(1))
            if 1 <= num <= len(citations):
                url = citations[num - 1]
                return f"[[{num}]]({url})"
            else:
                return match_obj.group(0)
        except (ValueError, IndexError):
            return match_obj.group(0)

    try:
        return re.sub(pattern, replace_citation, text)
    except Exception as e:
        print(f"Error during citation insertion: {e}")
        return text


# --- Helper function for formatting the final citation list ---
def _format_citation_list(citations: list[str]) -> str:
    """
    Formats a list of citation URLs into a markdown string.

    Args:
        citations: A list of citation URLs.

    Returns:
        A formatted markdown string (e.g., "\n\n---\nCitations:\n1. url1\n2. url2")
        or an empty string if no citations are provided.
    """
    if not citations:
        return ""

    try:
        citation_list = [f"{i+1}. {url}" for i, url in enumerate(citations)]
        return "\n\n---\nCitations:\n" + "\n".join(citation_list)
    except Exception as e:
        print(f"Error formatting citation list: {e}")
        return ""


class Pipeline:
    class Valves(BaseModel):
        OPENROUTER_API_KEY: str = ""
        OPENROUTER_MODEL: str = ""
        INCLUDE_REASONING: bool = True
        REQUEST_TIMEOUT: int = 60  # seconds
        pass

    def __init__(self):
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "openai_pipeline"
        self.name = "OpenAI Pipeline Test"
        self.valves = self.Valves(
            **{
                "OPENROUTER_API_KEY": os.getenv(
                    "OPENROUTER_API_KEY", "your-openai-api-key-here"
                ),
                "OPENROUTER_MODEL": os.getenv(
                    "OPENROUTER_MODEL", "e.g: tngtech/deepseek-r1t-chimera:free"
                ),
                "INCLUDE_REASONING": os.getenv(
                    "INCLUDE_REASONING", "true"
                ).lower() == "true",
                "REQUEST_TIMEOUT": int(
                    os.getenv(
                        "REQUEST_TIMEOUT", "60"
                    )
                )
            }
        )
        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    def pipelines(self) -> List[dict]:
        """
        Model Discovery Method: Called by Open WebUI at startup to get a list of models provided by this pipeline.
        This must take NO arguments (besides self).
        """
        return [
            {
                "id": self.name.lower().replace(" ", "-"), # Creates a unique ID, e.g., 'openai-pipeline'
                "name": self.name
            },
        ]

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        #T his is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        print(messages)
        print(user_message)
        body["stream"] = True
        print("\n\n\n\nbody:\n", body)
        
        # --- NEW CODE: Inject prompt for reasoning tags ---
        if self.valves.INCLUDE_REASONING:
            reasoning_prompt = {
                "role": "system",
                "content": f"You are a helpful assistant. Before providing your answer, you MUST enclose your entire thought process/reasoning within the tags {THINK_START_TAG} and {THINK_END_TAG}. Your final answer must appear *after* the closing tag. Do not include any other text outside these tags except the final answer. Example: {THINK_START_TAG}My thoughts{THINK_END_TAG} Final Answer."
            }
            # Append the instruction to the message list
            body["messages"].append(reasoning_prompt)
        # --- END NEW CODE ---

        OPENROUTER_API_KEY = self.valves.OPENROUTER_API_KEY
        OPENROUTER_MODEL =  self.valves.OPENROUTER_MODEL #"tngtech/deepseek-r1t-chimera:free"

        headers = {}
        headers["Authorization"] = f"Bearer {OPENROUTER_API_KEY}"
        headers["Content-Type"] = "application/json"

        payload = {**body, 
            "model": OPENROUTER_MODEL
        }
        
        payload["include_reasoning"] = True

        if "user" in payload:
            del payload["user"]
        if "chat_id" in payload:
            del payload["chat_id"]
        if "title" in payload:
            del payload["title"]

        print(payload)

        try:
            r = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                json=payload,
                headers=headers,
                stream=True,
            )

            r.raise_for_status()
            url="https://openrouter.ai/api/v1/chat/completions"
            if body["stream"]:
                return self.stream_response(
                    url,
                    headers,
                    payload,
                    _insert_citations,
                    _format_citation_list,
                    self.valves.REQUEST_TIMEOUT,
                )
            else:
                return r.json()
        except Exception as e:
            return f"Error: {e}"

    # def stream_response(
    #     self, url, headers, payload, citation_inserter, citation_formatter, timeout
    # ) -> Generator[str, None, None]:
    #     """Handles streaming API requests using a generator."""
    #     response = None
    #     try:
    #         response = requests.post(
    #             url, headers=headers, json=payload, stream=True, timeout=timeout
    #         )

    #         print("============")
    #         print("\nresponse: \n", response)
    #         print("============")

    #         response.raise_for_status()

    #         buffer = ""
    #         in_think = False
    #         latest_citations: List[str] = []

    #         for line in response.iter_lines():
    #             print("++++++++++++")
    #             print("\nline: \n", line)
    #             print("++++++++++++")

    #             if not line or not line.startswith(b"data: "):
    #                 continue
    #             data = line[len(b"data: ") :].decode("utf-8")
    #             if data == "[DONE]":
    #                 break
    #             try:
    #                 chunk = json.loads(data)
    #             except json.JSONDecodeError:
    #                 continue

    #             if "choices" in chunk:
    #                 choice = chunk["choices"][0]
    #                 citations = chunk.get("citations")
    #                 if citations is not None:
    #                     latest_citations = citations
    #                 delta = choice.get("delta", {})
    #                 content = delta.get("content", "")
    #                 reasoning = delta.get("reasoning", "")

    #                 # reasoning
    #                 if reasoning != None and reasoning != "":
    #                     if not in_think:
    #                         if buffer:
    #                             yield citation_inserter(buffer, latest_citations)
    #                             buffer = ""
    #                         yield "<think>\n"
    #                         in_think = True
    #                     buffer += reasoning

    #                 # content
    #                 if content:
    #                     if in_think:
    #                         if buffer:
    #                             yield citation_inserter(buffer, latest_citations)
    #                             buffer = ""
    #                         yield "\n</think>\n\n"
    #                         in_think = False
    #                     buffer += content

    #         # flush buffer
    #         if buffer:
    #             yield citation_inserter(buffer, latest_citations)
    #         yield citation_formatter(latest_citations)

    #     except requests.exceptions.Timeout:
    #         yield f"Pipe Error: Request timed out ({timeout}s)"
    #     except requests.exceptions.HTTPError as e:
    #         yield f"Pipe Error: API returned HTTP {e.response.status_code}"
    #     except Exception as e:
    #         yield f"Pipe Error: Unexpected error during streaming: {e}"
    #     finally:
    #         if response:
    #             response.close()
    def stream_response(
            self, url, headers, payload, citation_inserter, citation_formatter, timeout
        ) -> Generator[str, None, None]:
            """Handles streaming API requests using a generator."""
            response = None
            
            THINK_START_TAG = "<think>"
            THINK_END_TAG = "</think>"
            
            try:
                response = requests.post(
                    url, headers=headers, json=payload, stream=True, timeout=timeout
                )
                response.raise_for_status()

                buffer = ""
                in_think = False
                latest_citations: List[str] = []
                
                # --- NEW STATE VARIABLE ---
                reasoning_started_without_tag = False 

                for line in response.iter_lines():
                    if not line or not line.startswith(b"data: "):
                        continue
                    data = line[len(b"data: ") :].decode("utf-8")
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    if "choices" in chunk:
                        choice = chunk["choices"][0]
                        citations = chunk.get("citations")
                        if citations is not None:
                            latest_citations = citations
                            
                        delta = choice.get("delta", {})
                        content = delta.get("content", "")
                        reasoning = delta.get("reasoning", "")

                        # --- 1. Handle Native Reasoning (Existing logic - works if API uses this field) ---
                        if reasoning and self.valves.INCLUDE_REASONING:
                            if not in_think:
                                if buffer:
                                    yield citation_inserter(buffer, latest_citations)
                                    buffer = ""
                                yield "\n" + THINK_START_TAG + "\n"
                                in_think = True
                            buffer += reasoning
                        
                        # --- 2. Handle Content (The final answer or possibly mixed output) ---
                        if content:
                            
                            # Optimization: Remove tags if native reasoning was already used
                            if reasoning:
                                content = content.replace(THINK_START_TAG, "").replace(THINK_END_TAG, "")

                            # --- HYBRID PARSING & AUTO-INSERT LOGIC ---
                            
                            # Auto-insert <think> if reasoning is enabled and we are at the start of the stream
                            if self.valves.INCLUDE_REASONING and not in_think and not buffer and not reasoning_started_without_tag and not content.strip().startswith("{"):
                                # If content starts flowing AND we are configured for reasoning,
                                # AND it's not starting with the final answer (e.g., JSON), assume it's reasoning
                                reasoning_started_without_tag = True
                                yield "\n" + THINK_START_TAG + "\n"
                                in_think = True


                            # Check for the end of the thinking block in the content stream
                            if in_think and THINK_END_TAG in content:
                                # Existing logic to close the tag
                                think_end_index = content.find(THINK_END_TAG)
                                
                                buffer += content[:think_end_index]
                                
                                if buffer:
                                    yield citation_inserter(buffer, latest_citations)
                                    buffer = ""
                                yield "\n" + THINK_END_TAG + "\n\n"
                                in_think = False
                                reasoning_started_without_tag = False # Reset flag

                                buffer += content[think_end_index + len(THINK_END_TAG):]
                                
                            # Check for the start of the thinking block in the content stream 
                            elif not in_think and self.valves.INCLUDE_REASONING and THINK_START_TAG in content:
                                # Existing logic to open the tag
                                think_start_index = content.find(THINK_START_TAG)
                                
                                buffer += content[:think_start_index]
                                if buffer:
                                    yield citation_inserter(buffer, latest_citations)
                                    buffer = ""
                                
                                yield "\n" + THINK_START_TAG + "\n"
                                in_think = True
                                reasoning_started_without_tag = False # Found the tag, so don't need auto-insert
                                
                                buffer += content[think_start_index + len(THINK_START_TAG):]
                                
                            # Standard stream handling (append to current buffer)
                            else:
                                buffer += content

                # Final check to close <think> block (including auto-inserted ones)
                if in_think:
                    if buffer:
                        yield citation_inserter(buffer, latest_citations)
                        buffer = ""
                    yield "\n" + THINK_END_TAG + "\n\n"
                
                # flush final buffer and citations
                if buffer:
                    yield citation_inserter(buffer, latest_citations)
                yield citation_formatter(latest_citations)

            # ... (rest of exception handling remains the same)
            except requests.exceptions.Timeout:
                yield f"Pipe Error: Request timed out ({timeout}s)"
            except requests.exceptions.HTTPError as e:
                yield f"Pipe Error: API returned HTTP {e.response.status_code}"
            except Exception as e:
                yield f"Pipe Error: Unexpected error during streaming: {e}"
            finally:
                if response:
                    response.close()    