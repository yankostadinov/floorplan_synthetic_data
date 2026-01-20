from openai import OpenAI

from utils import list_filenames, plan_to_base64, save_text_to_file

client = OpenAI()


def save_json_to_file(plan_name, question_answer):
    filename = f"output/json/{plan_name}.json"

    save_text_to_file(filename, question_answer)


def generate_json_for_plan(plan_name):
    print(f"Generating JSON for {plan_name}...")
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
                        "text": "You will receive a PNG file containing a floor plan for a house or apartment. The format of the floor plan varies and may contain handwritten details. You must analyze the floor plan and output only a JSON-formatted object of the format { rooms: [], doors_with_endpoints: [], adjacency_list: [] }, where rooms contains strings for each room label, doors_with_endpoints contains door tuples specifying which two rooms are connected by the door and adjacency_list contains an object for each room label with the room label and adjacent rooms.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": plan_to_base64(plan_name)},
                ],
            },
        ],
        reasoning={"summary": "auto"},
        store=True,
        include=["reasoning.encrypted_content", "web_search_call.action.sources"],
    )
    save_json_to_file(plan_name, response.output_text)


for plan in list_filenames("plans"):
    generate_json_for_plan(plan)
