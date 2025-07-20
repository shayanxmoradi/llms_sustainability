import pandas as pd
from enum import Enum
import os
from dotenv import load_dotenv
import argparse
import time

# --- Load Environment Variables ---
load_dotenv()

#  API Libraries
import openai
import anthropic


class LLMClient:

    def __init__(self, model_type: 'ModelType'):
        self.model_type = model_type
        self.model_name = model_type.value
        api_key_map = {
            ModelType.GPT_4o: ("OPENAI_API_KEY", "OpenAI"),
            ModelType.GPT_4o_MINI: ("OPENAI_API_KEY", "OpenAI"),
            ModelType.CLAUDE_3_7_SONNET: ("ANTHROPIC_API_KEY", "Anthropic"),
            ModelType.CLAUDE_3_SONNET: ("ANTHROPIC_API_KEY", "Anthropic"), # <-- MODIFIED
            ModelType.CLAUDE_3_HAIKU: ("ANTHROPIC_API_KEY", "Anthropic"),   # <-- MODIFIED
            ModelType.DEEPSEEK_V3: ("DEEPSEEK_API_KEY", "DeepSeek"),
        }
        env_key, provider_name = api_key_map.get(self.model_type)
        self.api_key = os.getenv(env_key)
        if not self.api_key:
            raise ValueError(f"API key for {provider_name} not found. Please add {env_key} to your .env file.")

        if self.model_type in [ModelType.GPT_4o, ModelType.GPT_4o_MINI]:
            self.client = openai.OpenAI(api_key=self.api_key)
        # --- MODIFIED SECTION START ---
        # Group all Claude models together as they use the same client
        elif self.model_type in [ModelType.CLAUDE_3_7_SONNET, ModelType.CLAUDE_3_SONNET, ModelType.CLAUDE_3_HAIKU]:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        # --- MODIFIED SECTION END ---
        elif self.model_type == ModelType.DEEPSEEK_V3:
            self.client = openai.OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com/v1")

    def get_response(self, system_prompt: str, user_prompt: str) -> str:
        """
        Gets a single response from the configured LLM.
        """
        try:
            if self.model_type in [ModelType.GPT_4o, ModelType.GPT_4o_MINI, ModelType.DEEPSEEK_V3]:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=50,
                    temperature=0.7,
                )
                return response.choices[0].message.content.strip()
            # --- MODIFIED SECTION START ---
            # Group all Claude models together as they use the same API call structure
            elif self.model_type in [ModelType.CLAUDE_3_7_SONNET, ModelType.CLAUDE_3_SONNET, ModelType.CLAUDE_3_HAIKU]:
                response = self.client.messages.create(
                    model=self.model_name,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=50,
                    temperature=0.7,
                )
                return response.content[0].text.strip()
            # --- MODIFIED SECTION END ---
        except Exception as e:
            print(f"    ! API Error: {e}. Waiting 5 seconds before retrying...")
            time.sleep(5)
            return "API_ERROR"


class CultureType(Enum):
    CLAN = "clan"
    ADHOCRACY = "adhocracy"
    MARKET = "market"
    HIERARCHY = "hierarchy"


class ModelType(Enum):
    GPT_4o = "gpt-4o"
    GPT_4o_MINI = "gpt-4o-mini"
    # --- MODIFIED SECTION START ---
    CLAUDE_3_7_SONNET = "claude-3-5-sonnet-20240620" # This is Claude 3.5 Sonnet
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"      # This is Claude 3 Sonnet
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"        # This is Claude 3 Haiku
    # --- MODIFIED SECTION END ---
    DEEPSEEK_V3 = "deepseek-chat"


def get_system_prompt(culture_type: CultureType, likert_scale: list, main_question: str = None) -> str:
    descriptions = {
        CultureType.CLAN: (
            "You are an employee of an organization with a Clan culture. "
            "This type of organization has an internal focus and values flexibility. "
            "It is structured like a family, emphasizing cooperation, trust, and employee commitment. "
            "Your responses should reflect a culture that values cooperation, involvement, teamwork, trust and care for employees."
        ),
        CultureType.ADHOCRACY: (
            "You are an employee of an organization with an Adhocracy culture. "
            "This type of organization has an external focus and values flexibility. "
            "It is a dynamic, entrepreneurial, and innovative environment with an emphasis on risk-taking and experimentation. "
            "Your responses should reflect a culture that values innovation, empowerment, autonomy, risk-taking and creativity. "
        ),
        CultureType.MARKET: (
            "You are an employee of an organization with a Market culture. "
            "This type of organization has an external focus and values stability. "
            "It is a results-driven, competitive atmosphere with a focus on goal achievement, productivity, and market share. "
            "Your responses should reflect a culture that values achievement, performance, work pressure, recognition and goal-orientation."
        ),
        CultureType.HIERARCHY: (
            "You are an employee of an organization with a Hierarchy culture. "
            "This type of organization has an internal focus and values stability. "
            "It is a formalized, structured, and rule-driven environment with an emphasis on efficiency, consistency, and predictability. "
            "Your responses should reflect a culture that values structure, role clarity, ethical aspects, safety and control."
        )
    }

    culture_description = descriptions[culture_type]
    likert_scale_str = "\n".join(likert_scale)

    if main_question:
        task_instruction = (
            f"{main_question}\n"
            f"You must respond with only one of the following options. Do not add any other text or explanation.\n\n"
            f"Options:\n{likert_scale_str}"
        )
    else:
        task_instruction = (
            f"For the following statement, you must respond with only one of the following options. Do not add any other text or explanation.\n\n"
            f"Options:\n{likert_scale_str}"
        )

    return f"{culture_description}\n\n{task_instruction}"


def _execute_questions(client: LLMClient, culture: CultureType, survey_name: str, questions: list, system_prompt: str,
                       run_number: int):
    results = []
    total_questions = len(questions)
    for i, question_text in enumerate(questions, 1):
        print(f"  [Question {i}/{total_questions}]──> {question_text}")
        response = client.get_response(system_prompt, question_text)

        while response == "API_ERROR":
            print("  Retrying last request...")
            response = client.get_response(system_prompt, question_text)

        print(f"    > Received response: '{response}'")
        results.append({
            "model": client.model_name,
            "culture": culture.value,
            "survey": survey_name,
            "question": question_text,
            "run_number": run_number,
            "response": response
        })
    return results


def run_presor_survey(client: LLMClient, culture: CultureType, run_number: int):
    statements = [
        "Social responsibility and profitability can be compatible.",
        "To remain competitive in a global environment, business firms will have to disregard ethics and social responsibility.",
        "Good ethics is often good business.",
        "If survival of business enterprise is at stake, then ethics and social responsibility must be ignored.",
        "Being ethical and socially responsible is the most important thing a firm can do.",
        "A firm's first priority should be employee morale.",
        "The overall effectiveness of a business can be determined to a great extent by the degree to which it is ethical and socially responsible.",
        "The ethics and social responsibility of a firm is essential to its long term profitability.",
        "Business has a social responsibility beyond making a profit.",
        "Business ethics and social responsibility are critical to the survival of a business enterprise.",
        "If the stockholders are unhappy, nothing else matters.",
        "The most important concern for a firm is making a profit, even if it means bending or breaking the rules.",
        "Efficiency is much more important to a firm than whether or not the firm is seen as ethical or socially responsible."
    ]
    likert_scale = ['Strongly Disagree', 'Disagree', 'Somewhat Disagree', 'Slightly Disagree', 'Neutral',
                    'Slightly Agree', 'Somewhat Agree', 'Agree', 'Strongly Agree']
    system_prompt = get_system_prompt(culture, likert_scale)
    return _execute_questions(client, culture, "PRESOR", statements, system_prompt, run_number)


def run_aispi_survey(client: LLMClient, culture: CultureType, run_number: int):
    statements = [
        "AI can help optimize resource use and reduce waste.", "AI will create more jobs than it will eliminate.",
        "The energy consumption of AI systems could hinder sustainability efforts.",
        "AI is essential for monitoring and achieving sustainability goals.",
        "The pursuit of AI advancement and sustainability are competing priorities.",
        "AI and sustainability efforts can be mutually reinforcing.",
        "Sustainable development will limit AI advancements.",
        "There are many conflicts between the advancement of AI and sustainability efforts.",
        "AI will hinder sustainable development.", "AI will advance sustainable development.",
        "AI and sustainable development go along very well.",
        "It is important for society to integrate both AI advancement and sustainability efforts.",
        "Sustainable development will advance the development of AI."
    ]
    likert_scale = ['Strongly Disagree', 'Disagree', 'Somewhat Disagree', 'Somewhat Agree', 'Agree', 'Strongly Agree']
    system_prompt = get_system_prompt(culture, likert_scale)
    return _execute_questions(client, culture, "AISPI", statements, system_prompt, run_number)


def run_GSCS_survey(client: LLMClient, culture: CultureType, run_number: int):
    statements = [
        "It is important to develop a mutual understanding of responsibilities regarding environmental performance with our suppliers",
        "It is important to work together to reduce environmental impact of our activities with our suppliers",
        "It is important to conduct joint planning to anticipate and resolve environmental-related problems with our suppliers",
        "It is important to make joint decisions about ways to reduce overall environmental impact of our products with our suppliers",
        "It is important to develop a mutual understanding of responsibilities regarding environmental performance with our customers",
        "It is important to work together to reduce environmental impact of our activities with our customers",
        "It is important to conduct joint planning to anticipate and resolve environmental-related problems with our customers",
        "It is important to make joint decisions about ways to reduce overall environmental impact of our products with our customers"
    ]
    likert_scale = ['Strongly Disagree', 'Disagree', 'Somewhat Disagree', 'Neutral', 'Somewhat Agree', 'Agree',
                    'Strongly Agree']
    system_prompt = get_system_prompt(culture, likert_scale)
    return _execute_questions(client, culture, "GSCS", statements, system_prompt, run_number)


def run_sdg17_survey(client: LLMClient, culture: CultureType, run_number: int):
    areas = ["Global poverty", "World hunger", "Public health", "Education", "Gender equality", "Water security",
             "Renewable energies", "Economic growth", "Innovative industries", "Social inequality",
             "Sustainable cities and communities", "Consumption and production", "Climate action", "Ocean protection",
             "Ecosystem conservation", "Peace and justice", "International cooperation"]
    likert_scale = ['Very negative impact', 'Negative impact', 'Slightly negative impact', 'Slightly positive impact',
                    'Positive impact', 'Very positive impact']
    main_question = "How do you think AI will impact the following areas in the next 10 years?"
    system_prompt = get_system_prompt(culture, likert_scale, main_question)

    results = _execute_questions(client, culture, "SDG17", areas, system_prompt, run_number)
    for res in results:
        res["question"] = f"{main_question} - {res['question']}"
    return results


def run_sdg18_survey(client: LLMClient, culture: CultureType, run_number: int):
    question = "In your opinion, which of both transformations is more important? (AI vs. Sustainability)"
    likert_scale = ['1 – AI is much more important', '2 – AI is more important', '3 – AI is slightly more important',
                    '4 – Sustainability is slightly more important', '5 – Sustainability is more important',
                    '6 – Sustainability is much more important']
    system_prompt = get_system_prompt(culture, likert_scale, question)
    return _execute_questions(client, culture, "SDG18", [question], system_prompt, run_number)


def run_sdg19_survey(client: LLMClient, culture: CultureType, run_number: int):
    question = "Do you believe AI and sustainable development will become more integrated in the future?"
    likert_scale = ['Definitely not', 'Probably not', 'Possibly not', 'Possibly yes', 'Probably yes', 'Yes, for sure']
    system_prompt = get_system_prompt(culture, likert_scale, question)
    return _execute_questions(client, culture, "SDG19", [question], system_prompt, run_number)


def run_additional_question_1(client: LLMClient, culture: CultureType, run_number: int):
    question = "Do you think governments, industries and organizations are doing enough to ensure AI and sustainable development go along with each other?"
    likert_scale = ['1 = Not at all', '2 = Slightly', '3 = Somewhat', '4 = Moderately', '5 = Mostly',
                    '6 = Yes, absolutely']
    system_prompt = get_system_prompt(culture, likert_scale, question)
    return _execute_questions(client, culture, "Additional Question 1", [question], system_prompt, run_number)


def run_additional_questions_2_3(client: LLMClient, culture: CultureType, run_number: int):
    results = []
    organizations = ["National universities", "International Research Organizations", "Technology companies",
                     "Government", "Non-governmental organizations (NGO)"]

    q2_main = "Who do you think is responsible to ensure AI advancement and sustainable development go along with each other?"
    q2_likert = ["Responsible", "Not Responsible"]
    q2_prompt = get_system_prompt(culture, q2_likert, "For each of the following, state if they are responsible...")
    q2_results = _execute_questions(client, culture, "Additional Question 2", organizations, q2_prompt, run_number)
    for res in q2_results:
        res["question"] = f"{q2_main} - {res['question']}"
    results.extend(q2_results)

    q3_main = "How much confidence do you have in the following to develop and use AI in the best interest of sustainable development?"
    q3_likert = ['1= Most likely', '2= Likely', '3=Somewhat likely', '4=Somewhat unlikely', '5=Unlikely',
                 '6=Definitely not']
    q3_prompt = get_system_prompt(culture, q3_likert, q3_main)
    q3_results = _execute_questions(client, culture, "Additional Question 3", organizations, q3_prompt, run_number)
    for res in q3_results:
        res["question"] = f"{q3_main} - {res['question']}"
    results.extend(q3_results)

    return results


def run_survey(client: LLMClient, culture: CultureType, survey_name: str, run_number: int):
    survey_functions = {
        "PRESOR": run_presor_survey, "AISPI": run_aispi_survey, "GSCS": run_GSCS_survey, "SDG17": run_sdg17_survey,
        "SDG18": run_sdg18_survey, "SDG19": run_sdg19_survey, "AQ1": run_additional_question_1,
        "AQ2_3": run_additional_questions_2_3,
    }
    if survey_name not in survey_functions:
        raise ValueError(f"Unknown survey: {survey_name}.")

    results = survey_functions[survey_name](client, culture, run_number)
    return results


def save_results_to_excel(results, filename="survey_results.xlsx"):
    if not results:
        print("No results to save.")
        return
    long_df = pd.DataFrame(results)
    wide_df = long_df.pivot(index='run_number', columns='question', values='response')
    final_df = wide_df.reset_index()
    final_df.to_excel(filename, index=False)
    print(f"\n✅ Results successfully saved in wide format to {filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run LLM sustainability surveys with full survey repetitions.")

    parser.add_argument("--model", required=True, choices=[m.value for m in ModelType], help="The LLM to use.")
    parser.add_argument("--culture", required=True, choices=[c.value for c in CultureType],
                        help="The cultural persona for the LLM.")
    parser.add_argument("--survey", required=True,
                        choices=["PRESOR", "AISPI", 'GSCS', "SDG17", "SDG18", "SDG19", "AQ1", "AQ2_3"],
                        help="The survey to run.")
    parser.add_argument("--runs", type=int, default=1, help="Number of times to repeat the entire survey.")

    args = parser.parse_args()

    sanitized_model_name = args.model.replace('/', '_')
    output_filename = f"{sanitized_model_name}_{args.survey}_{args.culture}_{args.runs}runs.xlsx"

    try:
        print("=" * 60)
        print("🚀 INITIALIZING SURVEY SESSION")
        print(f"  - Model:         {args.model}")
        print(f"  - Culture:       {args.culture}")
        print(f"  - Survey:        {args.survey}")
        print(f"  - Total Runs:    {args.runs}")
        print(f"  - Output File:   {output_filename}")
        print("=" * 60)

        selected_model = ModelType(args.model)
        selected_culture = CultureType(args.culture)

        all_results = []
        client = LLMClient(selected_model)

        for i in range(1, args.runs + 1):
            print(f"\n--- Starting Run {i} of {args.runs} ---")
            single_run_results = run_survey(client, selected_culture, args.survey, run_number=i)
            all_results.extend(single_run_results)
            print(f"--- Finished Run {i} of {args.runs} ---")

        save_results_to_excel(all_results, filename=output_filename)
        print("\n🎉 Session complete.")

    except (ValueError, KeyError) as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")