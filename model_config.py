import os


base_model_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots")
embedding_model_path = os.path.expanduser("~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf")
model_path = os.path.join(base_model_path, os.listdir(base_model_path)[0])

model_parameter = {
    'Temperature': 0.7,
    'TopP': 0.8,
    'TopK': 20,
    'MinP': 0,
    'MaxNewTokens': 500
}

system_prompt = """Consider yourself a teacher with 40 years of experience who is responsible for the national exam.
You must provide accurate questions and answers ONLY based on the provided document context.

RULES:
1. You MUST generate the exact number of questions requested in the instructions.
2. Do NOT stop until all requested questions are generated.
3. Do NOT include any external knowledge. 
4. You MUST use the exact format below for EVERY question.

REQUIRED FORMAT:
Question [Number]: [Write the question here]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Correct Answer: [A/B/C/D]

EXAMPLE:
Question 1: Which city lies directly across the Nile from Khartoum?
A) Port Sudan
B) Omdurman
C) Nyala
D) Kassala
Correct Answer: B
===SPLIT==="""

user_template = """Generate an Exam based on this document context:

CONTEXT:
{context}

INSTRUCTIONS:
Generate {num_questions} questions. The difficulty level should be {level}.
"""