import os
from openai import OpenAI

token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

def main():
    print("Test GPT-4.1 in your terminal. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == 'exit':
            break
        messages = [
            {"role": "system", "content": "You are PIXEL, a polite and helpful student assistant at the CpE Department. You are familiar with department procedures, requirements, and academic policies. You help students in a friendly and professional manner. If a question is out of scope, politely say so. Do not answer homework or assignment questions."},
            {"role": "user", "content": user_input}
        ]
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=1.0,
                top_p=1.0
            )
            print("PIXEL:", response.choices[0].message["content"].strip())
        except Exception as e:
            print("[Error]", e)

if __name__ == "__main__":
    main()

# response = client.chat.completions.create(
#     messages=[
#         {
#             "role": "system",
#             "content": "You are a helpful assistant.",
#         },
#         {
#             "role": "user",
#             "content": "What is the capital of France?",
#         }
#     ],
#     temperature=1.0,
#     top_p=1.0,
#     model=model
# )

# print(response.choices[0].message.content)

