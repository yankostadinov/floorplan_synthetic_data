from openai import OpenAI

from utils import list_filenames, plan_to_base64, save_text_to_file

client = OpenAI()

QUESTIONS = {
    "bathroom_path_from_entrance": "What's the shortest path from the entrance to a bathroom? Present the answer as an ordered comma-separated string of which rooms you need to go through.",
    "bedroom_count": "How many bedrooms are there?",
    "bathroom_count": "How many bathrooms are there?",
}


def build_question_input(question_type, plan_name):
    return {
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": QUESTIONS[question_type],
            },
            {"type": "input_image", "image_url": plan_to_base64(plan_name)},
        ],
    }


def save_question_answer_to_file(question_type, plan_name, question_answer):
    filename = f"output/questions/{question_type}/{plan_name}.txt"

    save_text_to_file(filename, question_answer)


def ask_question_for_plan(question_type, plan_name):
    print(f'Asking question "{question_type}" for {plan_name}...')
    response = client.responses.create(
        model="gpt-5.2",
        temperature=0.2,
        top_p=0.1,
        input=[
            {
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": "You will receive a PNG file containing a floor plan for a house or apartment. The format of the floor plan varies and may contain handwritten details.You must analyze the floor plan and answer text questions in a succinct manner.",
                    }
                ],
            },
            build_question_input(question_type, plan_name),
        ],
        reasoning={"summary": "auto"},
        store=True,
        include=["reasoning.encrypted_content", "web_search_call.action.sources"],
    )
    save_question_answer_to_file(question_type, plan_name, response.output_text)


def ask_questions_for_plan(plan_name):
    for question_type in QUESTIONS:
        ask_question_for_plan(question_type, plan_name)


for plan in list_filenames("plans"):
    ask_questions_for_plan(plan)
