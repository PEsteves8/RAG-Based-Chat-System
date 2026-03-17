from typing import Dict, List
from openai import OpenAI

def generate_response(openai_key: str, user_message: str, context: str,
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""

    # TODO: Define system prompt
    system_prompt = """You are an expert AI assistant specializing in NASA space missions, \
including Apollo 11, Apollo 13, and the Space Shuttle Challenger.

When answering questions:
- Base your answers strictly on the retrieved context provided. Do not introduce facts, \
dates, or details that are not present in the context.
- If the context does not contain enough information to answer fully, say so clearly \
rather than speculating or guessing.
- Cite the source of your information using the [Source N] labels from the context \
(e.g., "According to [Source 1]..."). This helps the user verify your answer.
- Be informative and precise while remaining grounded in the provided documents."""

    # TODO: Set context in messages
    messages = [{"role": "system", "content": system_prompt}]

    if context:
        messages.append({
            "role": "system",
            "content": f"Relevant context from NASA mission documents:\n\n{context}"
        })

    # TODO: Add chat history
    messages.extend(conversation_history)

    # Append the current user message
    messages.append({"role": "user", "content": user_message})

    # TODO: Creaet OpenAI Client
    client = OpenAI(
        base_url="https://openai.vocareum.com/v1",
        api_key=openai_key
    )

    # TODO: Send request to OpenAI
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )

    # TODO: Return response
    return response.choices[0].message.content