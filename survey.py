import pandas as pd
from enum import Enum
import os
from dotenv import load_dotenv
import argparse  # Import the argparse library

# Import API libraries
import openai
import anthropic

# Load API keys from .env file in the same directory
load_dotenv()

class LLMClient:
    """
    A client to handle API calls to different LLM providers (OpenAI, Anthropic, DeepSeek).
    It reads API keys from a .env file.
    """
    def __init__(self, model_type: 'ModelType'):
        self.model_type = model_type
        self.model_name = model_type.value

        # Initialize clients based on the model type
        if self.model_type == ModelType.GPT_4o:
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("API key for OpenAI not found. Please add OPENAI_API_KEY to your .env file.")
            self.client = openai.OpenAI(api_key=self.api_key)
        elif self.model_type == ModelType.CLAUDE_3_7_SONNET:
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError("API key for Anthropic not found. Please add ANTHROPIC_API_KEY to your .env file.")
            self.client = anthropic.Anthropic(api_key=self.api_key)
        elif self.model_type == ModelType.DEEPSEEK_V2:
            self.api_key = os.getenv("DEEPSEEK_API_KEY")
            if not self.api_key:
                raise ValueError("API key for DeepSeek not found. Please add DEEPSEEK_API_KEY to your .env file.")
            self.client = openai.OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com/v1")
        else:
            raise ValueError(f"Model {self.model_name} not supported by this client.")

    def get_response(self, system_prompt: str, user_prompt: str) -> str:
        """
        Gets a response from the configured LLM.
        """
        print("--------------------------------------------------")
        print(f"Sending request to {self.model_name}...")
        print("--------------------------------------------------")

        try:
            if self.model_type in [ModelType.GPT_4o, ModelType.DEEPSEEK_V2]:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=50,
                    temperature=0.1,
                )
                return response.choices[0].message.content.strip()

            elif self.model_type == ModelType.CLAUDE_3_7_SONNET:
                response = self.client.messages.create(
                    model=self.model_name,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=50,
                    temperature=0.1,
                )
                return response.content[0].text.strip()

        except Exception as e:
            print(f"An API error occurred: {e}")
            return f"API_ERROR: {e}"

class CultureType(Enum):
    CLAN = "clan"
    ADHOCRACY = "adhocracy"
    MARKET = "market"
    HIERARCHY = "hierarchy"

class ModelType(Enum):
    GPT_4o = "gpt-4o"
    CLAUDE_3_7_SONNET = "claude-3-5-sonnet-20240620"
    DEEPSEEK_V2 = "deepseek-chat"

def get_system_prompt(culture_type: CultureType, likert_scale: list, main_question: str = None) -> str:
    descriptions = {
        CultureType.CLAN: (
            "You are an employee of an organization with a Clan culture. "
            "This type of organization has an internal focus and values flexibility. "
            "It is structured like a family, emphasizing collaboration, trust, and strong employee commitment. "
            "Assume that organizational members behave properly when they feel trusted and committed to the organization. "
            "Your responses should reflect a culture that values participation, loyalty, teamwork, support, employee involvement, and engagement. "
            "Leaders in your organization are like mentors or parental figures. "
            "Decisions prioritize maintaining a friendly and supportive internal climate."
        ),
        CultureType.ADHOCRACY: (
            "You are a representative of an organization with an Adhocracy culture. "
            "This type of organization has an external focus and values flexibility. "
            "It is a dynamic, entrepreneurial, and innovative environment with an emphasis on risk-taking and experimentation. "
            "Assume that organizational members behave properly when they view their work as meaningful and impactful. "
            "Your responses should reflect a culture that values autonomy, growth, and stimulation, with associated behaviors like creativity and risk-taking. "
            "Leaders in your organization are visionary, innovative, and willing to take risks. "
            "Success is defined by innovation, growth, and cutting-edge output, and the organization is seen as effective when employees are innovating."
        ),
        CultureType.MARKET: (
            "You are responding as a representative of an organization with a Market culture. "
            "This type of organization has an external focus and values stability. "
            "It is a results-driven, competitive atmosphere with a focus on goal achievement, productivity, and market share. "
            "Assume that organizational members behave properly when they have clear goals and are rewarded for their performance. "
            "Your responses should reflect a culture that values rivalry, achievement, and competence, and behaviors such as being aggressive and competing with other companies. "
            "Leaders in your organization are hard drivers, producers, and competitors. "
            "Success is defined by winning in the marketplace and by increasing profits and market share."
        ),
        CultureType.HIERARCHY: (
            "You are responding as a representative of an organization with a Hierarchy culture. "
            "This type of organization has an internal focus and values stability. "
            "It is a formalized, structured, and rule-driven environment with an emphasis on efficiency, consistency, and predictability. "
            "Assume that organizational members behave properly when there are clear roles, rules, and regulations. "
            "Your responses should reflect a culture that values formalization, routinization, and consistency, with associated behaviors like conformity and predictability. "
            "Leaders in your organization are coordinators, monitors, and organizers. "
            "Success is measured by smooth operations and efficiency."
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

# All survey functions (run_presor_survey, run_aispi_survey, etc.) remain the same...
# (The full survey functions are omitted here for brevity but should be in your script)
def run_presor_survey(client: LLMClient, culture: CultureType):
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
    likert_scale = [
        'Strongly Disagree', 'Disagree', 'Somewhat Disagree', 'Slightly Disagree',
        'Neutral', 'Slightly Agree', 'Somewhat Agree', 'Agree', 'Strongly Agree'
    ]
    system_prompt = get_system_prompt(culture, likert_scale)
    results = []
    for stmt in statements:
        response = client.get_response(system_prompt, stmt)
        results.append({
            "model": client.model_name,
            "culture": culture.value,
            "survey": "PRESOR",
            "question": stmt,
            "response": response
        })
    return results

def run_aispi_survey(client: LLMClient, culture: CultureType):
    statements = [
        "AI can help optimize resource use and reduce waste.",
        "AI will create more jobs than it will eliminate.",
        "The energy consumption of AI systems could hinder sustainability efforts.",
        "AI is essential for monitoring and achieving sustainability goals.",
        "The pursuit of AI advancement and sustainability are competing priorities.",
        "AI and sustainability efforts can be mutually reinforcing.",
        "Sustainable development will limit AI advancements.",
        "There are many conflicts between the advancement of AI and sustainability efforts.",
        "AI will hinder sustainable development.",
        "AI will advance sustainable development.",
        "AI and sustainable development go along very well.",
        "It is important for society to integrate both AI advancement and sustainability efforts.",
        "Sustainable development will advance the development of AI."
    ]
    likert_scale = [
        'Strongly Disagree', 'Disagree', 'Somewhat Disagree',
        'Somewhat Agree', 'Agree', 'Strongly Agree'
    ]
    system_prompt = get_system_prompt(culture, likert_scale)
    results = []
    for stmt in statements:
        response = client.get_response(system_prompt, stmt)
        results.append({
            "model": client.model_name,
            "culture": culture.value,
            "survey": "AISPI",
            "question": stmt,
            "response": response
        })
    return results

def run_sdg17_survey(client: LLMClient, culture: CultureType):
    areas = [
        "Global poverty", "World hunger", "Public health", "Education",
        "Gender equality", "Water security", "Renewable energies",
        "Economic growth", "Innovative industries", "Social inequality",
        "Sustainable cities and communities", "Consumption and production",
        "Climate action", "Ocean protection", "Ecosystem conservation",
        "Peace and justice", "International cooperation"
    ]
    likert_scale = [
        'Very negative impact', 'Negative impact', 'Slightly negative impact',
        'Slightly positive impact', 'Positive impact', 'Very positive impact'
    ]
    main_question = "How do you think AI will impact the following areas in the next 10 years?"
    system_prompt = get_system_prompt(culture, likert_scale, main_question)
    results = []
    for area in areas:
        response = client.get_response(system_prompt, area)
        results.append({
            "model": client.model_name,
            "culture": culture.value,
            "survey": "SDG17",
            "question": f"{main_question} - {area}",
            "response": response
        })
    return results

def run_sdg18_survey(client: LLMClient, culture: CultureType):
    question = "In your opinion, which of both transformations is more important?"
    likert_scale = [
        '1 – AI is much more important', '2 – AI is more important', '3 – AI is slightly more important',
        '4 – Sustainability is slightly more important', '5 – Sustainability is more important', '6 – Sustainability is much more important'
    ]
    system_prompt = get_system_prompt(culture, likert_scale, question)
    response = client.get_response(system_prompt, "AI vs. Sustainability")
    return [{
        "model": client.model_name,
        "culture": culture.value,
        "survey": "SDG18",
        "question": question,
        "response": response
    }]

def run_sdg19_survey(client: LLMClient, culture: CultureType):
    question = "Do you believe AI and sustainable development will become more integrated in the future?"
    likert_scale = [
        'Definitely not', 'Probably not', 'Possibly not',
        'Possibly yes', 'Probably yes', 'Yes, for sure'
    ]
    system_prompt = get_system_prompt(culture, likert_scale, question)
    response = client.get_response(system_prompt, question)
    return [{
        "model": client.model_name,
        "culture": culture.value,
        "survey": "SDG19",
        "question": question,
        "response": response
    }]

def run_additional_question_1(client: LLMClient, culture: CultureType):
    question = "Do you think governments, industries and organizations are doing enough to ensure AI and sustainable development go along with each other?"
    likert_scale = [
        '1 = Not at all', '2 = Slightly', '3 = Somewhat',
        '4 = Moderately', '5 = Mostly', '6 = Yes, absolutely'
    ]
    system_prompt = get_system_prompt(culture, likert_scale, question)
    response = client.get_response(system_prompt, question)
    return [{
        "model": client.model_name,
        "culture": culture.value,
        "survey": "Additional Question 1",
        "question": question,
        "response": response
    }]

def run_additional_questions_2_3(client: LLMClient, culture: CultureType):
    q2_question = "Who do you think is responsible to ensure AI advancement and sustainable development go along with each other?"
    q3_question = "How much confidence do you have in the following to develop and use AI in the best interest of sustainable development?"
    organizations = [
        "National universities", "International Research Organizations",
        "Technology companies", "Government", "Non-governmental organizations (NGO)"
    ]
    likert_scale_q3 = [
        '1= Most likely', '2= Likely', '3=Somewhat likely',
        '4=Somewhat unlikely', '5=Unlikely', '6=Definitely not'
    ]
    results = []
    # Handling Question 2
    q2_likert_scale = ["Responsible", "Not Responsible"]
    q2_system_prompt = get_system_prompt(culture, q2_likert_scale, "For each of the following, state if they are responsible to ensure AI advancement and sustainable development go along with each other.")
    for org in organizations:
        response = client.get_response(q2_system_prompt, org)
        results.append({
            "model": client.model_name,
            "culture": culture.value,
            "survey": "Additional Question 2",
            "question": f"{q2_question} - {org}",
            "response": response
        })
    # Handling Question 3
    q3_system_prompt = get_system_prompt(culture, likert_scale_q3, q3_question)
    for org in organizations:
        response = client.get_response(q3_system_prompt, org)
        results.append({
            "model": client.model_name,
            "culture": culture.value,
            "survey": "Additional Question 3",
            "question": f"{q3_question} - {org}",
            "response": response
        })
    return results

def run_survey(model: ModelType, culture: CultureType, survey_name: str):
    """
    Runs a specific survey with a given model and culture.
    """
    client = LLMClient(model)
    survey_functions = {
        "PRESOR": run_presor_survey,
        "AISPI": run_aispi_survey,
        "SDG17": run_sdg17_survey,
        "SDG18": run_sdg18_survey,
        "SDG19": run_sdg19_survey,
        "AQ1": run_additional_question_1,
        "AQ2_3": run_additional_questions_2_3,
    }

    if survey_name not in survey_functions:
        raise ValueError(f"Unknown survey: {survey_name}. Available surveys are: {list(survey_functions.keys())}")

    print(f"\nRunning survey '{survey_name}' for model '{model.value}' with culture '{culture.value}'...")
    results = survey_functions[survey_name](client, culture)
    return results

def save_results_to_excel(results, filename="survey_results.xlsx"):
    """
    Saves a list of result dictionaries to an Excel file.
    Appends data if the file already exists.
    """
    if not results:
        print("No results to save.")
        return

    new_data_df = pd.DataFrame(results)
    if os.path.exists(filename):
        try:
            existing_data_df = pd.read_excel(filename)
            combined_df = pd.concat([existing_data_df, new_data_df], ignore_index=True)
        except Exception as e:
            print(f"Could not read existing Excel file {filename}. It might be corrupted. Appending to a new file with timestamp.")
            filename = filename.replace(".xlsx", f"_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.xlsx")
            combined_df = new_data_df
    else:
        combined_df = new_data_df

    combined_df.to_excel(filename, index=False)
    print(f"\nResults successfully saved to {filename}")


if __name__ == '__main__':
    # --- THIS IS THE NEW SECTION FOR COMMAND-LINE ARGUMENTS ---

    # Define available choices for the command-line arguments
    model_choices = [m.value for m in ModelType]
    culture_choices = [c.value for c in CultureType]
    survey_choices = ["PRESOR", "AISPI", "SDG17", "SDG18", "SDG19", "AQ1", "AQ2_3"]

    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Run LLM sustainability surveys with different models and cultural personas.")
    parser.add_argument("--model", required=True, choices=model_choices, help="The LLM to use.")
    parser.add_argument("--culture", required=True, choices=culture_choices, help="The cultural persona for the LLM.")
    parser.add_argument("--survey", required=True, choices=survey_choices, help="The survey to run.")
    parser.add_argument("--output_file", default="llm_sustainability_results.xlsx", help="The name of the output Excel file.")

    # Parse the arguments from the command line
    args = parser.parse_args()

    # --- Execution ---
    try:
        # Convert string arguments back to Enum types for the function call
        selected_model = ModelType(args.model)
        selected_culture = CultureType(args.culture)

        # Run the selected survey
        survey_results = run_survey(selected_model, selected_culture, args.survey)

        # Save the results to the specified Excel file
        save_results_to_excel(survey_results, filename=args.output_file)

    except ValueError as e:
        print(f"Configuration or Execution Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")