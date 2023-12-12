import os
import csv
import asyncio
import aiohttp
import aiofiles
import logging
from markdown import markdown
import xhtml2pdf.pisa as pisa
from prompt_generator import generate_prompts, sample_prompts
from tenacity import retry, wait_random_exponential, stop_after_attempt
from dotenv import load_dotenv
import openai
import pandas as pd
import json
from tqdm import tqdm
import re

# Load environment variables from a .env file
load_dotenv()



# Access OpenAI organization ID and API key
openai.organization_id = os.getenv("OPENAI_ORGANIZATION_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

########Params and configuration########
USE_SAMPLE_PROMPTS = True  # Set to False to use input file
GENERATE_PDFS = True  # Set to False to disable PDF generation
SAMPLES_PER_CAT = 1  # Number of samples per category
SEO_TOKENS = "SEO_Template_full.csv"
TOKEN_LENGTH = 3000  # Max size of the generated prompts in tokens

PROMPT_CSV_FILENAME = "sampled_prompts.csv"  # If not None, read prompts from this CSV file instead of generating prompts from the tokens.


PRIMER = (
    "You are an advanced SEO article outline generator specialized in creating structured outlines "
    "Your task is to produce a detailed SEO article outline based on the given topic. "
    "The article must be informative, well-researched, and adhere to SEO best practices. "
    "Generate an article structure with an H1 title, followed by 10 relevant H2 subtitles, and each H2 subtitle should have one relevant H3 subheader. "
    "Include relevant headings and subheadings to structure the content effectively. "
    "The article outline should be in an informational guide format, optimized for search rankings."
    "Write as if you are a knowledgeable researcher and journalist."
    "Format the output in markdown, including titles, headings, and body content."
    "Output the article skeleton in JSON format: {"
        "\"Category\": \"<Category>\","
        "\"H1_Title\": \"<H1_Title>\","
        "\"H2_Subtitles\": ["
            "{ \"H2_Title\": \"<H2_Title1>\", \"H3_Subheader\": \"<H3_Subheader1>\" },"
            "{ \"H2_Title\": \"<H2_Title2>\", \"H3_Subheader\": \"<H3_Subheader2>\" },"
            "{ \"H2_Title\": \"<H2_Title3>\", \"H3_Subheader\": \"<H3_Subheader3>\" },"
            "{ \"H2_Title\": \"<H2_Title4>\", \"H3_Subheader\": \"<H3_Subheader4>\" },"
            "{ \"H2_Title\": \"<H2_Title5>\", \"H3_Subheader\": \"<H3_Subheader5>\" },"
            "{ \"H2_Title\": \"<H2_Title6>\", \"H3_Subheader\": \"<H3_Subheader6>\" },"
            "{ \"H2_Title\": \"<H2_Title7>\", \"H3_Subheader\": \"<H3_Subheader7>\" },"
            "{ \"H2_Title\": \"<H2_Title8>\", \"H3_Subheader\": \"<H3_Subheader8>\" },"
            "{ \"H2_Title\": \"<H2_Title9>\", \"H3_Subheader\": \"<H3_Subheader9>\" },"
            "{ \"H2_Title\": \"<H2_Title10>\", \"H3_Subheader\": \"<H3_Subheader10>\" }"
        "]"
    "}"
    
"The following is the prompt and any included links:"
""
""
)
########################################
def sanitize_filename(filename):
    """
    Function to sanitize filenames to remove illegal characters
    """
    return re.sub(r'[\\/*?:"<>|]', "", filename)


def create_pdf(title, subtitle, body, output_filename):
     # Sanitize the title and subtitle
    title = sanitize_filename(title)
    subtitle = sanitize_filename(subtitle)
    styles = """
        <style>
            h1 {
                font-size: 24pt;
            }
            h2 {
                font-size: 18pt;
            }
            h3 {
                font-size: 16pt;
            }
            h4 {
                font-size: 15pt;
            }
            p, li {
                font-size: 12pt;
                text-align: justify;
            }
        </style>
    """

    body_html = markdown(body)
    html = f"""
        {styles}
        <h1>{title}</h1>
        <h2>{subtitle}</h2>
        {body_html}
    """

    with open(output_filename, "wb") as pdf_file:
        pisa_status = pisa.CreatePDF(html, dest=pdf_file)

    if pisa_status.err:
        print(f"Error creating PDF file: {output_filename}")

def fetch_openai_completion_async(model, prompt, temperature, max_tokens, top_p, frequency_penalty, presence_penalty):
    return openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )

def fetch_openai_completion(**kwargs):
    return openai.Completion.create(**kwargs)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def fetch_prompt(session, category, input_prompt):
    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(
            None,
            fetch_openai_completion_async,
            "text-davinci-003",
            PRIMER + input_prompt,
            0.35,
            TOKEN_LENGTH,
            1.0,
            0.3,
            0.1
        )
        logger.debug(f"Received response from OpenAI API: {response}")
        logger.info(f"Full API response: {response}")

        if not response or 'choices' not in response or not response['choices']:
            logger.error("Received empty or invalid response from OpenAI API")
            return category, input_prompt, None

        output = response['choices'][0]['text'].strip()
        logger.debug(f"API Response Text: {output}")  # Logging the raw response text
        if not output:
            logger.error("Received empty text from OpenAI API response")
            return category, input_prompt, None

        # Find the start of the JSON structure
        json_start = output.find('{')
        if json_start != -1:
            json_output = output[json_start:]
            try:
                article_skeleton = json.loads(json_output)
                return category, input_prompt, article_skeleton
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                return category, input_prompt, None
        else:
            logger.error("No JSON structure found in response")
            return category, input_prompt, None
    except Exception as e:
        logger.error(f"Exception in fetch_prompt: {e}")
        raise



    
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def generate_section_content(session, h2_title, h3_subheader):
    loop = asyncio.get_event_loop()
    # Construct a specific prompt for generating content for the H2 and H3 section
    section_prompt = f"Write a detailed article section about '{h2_title}' with a focus on '{h3_subheader}'."

    try:
        response = await loop.run_in_executor(
            None,
            fetch_openai_completion_async,
            "text-davinci-003",
            section_prompt,
            0.35,
            TOKEN_LENGTH,
            1.0,
            0.3,
            0.1
        )
        logger.debug(f"Received response from OpenAI API: {response}")
        section_content = response['choices'][0]['text'].strip()
        return section_content
    except Exception as e:
        logger.error(f"Exception in generate_section_content: {e}")
        raise
    



async def process_prompt(session, writer, sent_prompts, category, prompt):
    if prompt in sent_prompts:
        logger.info(f"Skipping already sent prompt: {prompt}")
        return

    logger.info(f"Sending to OpenAI API: {prompt[:100]}...")
    sent_prompts.add(prompt)

    category, prompt, article_skeleton = await fetch_prompt(session, category, prompt)

    if not article_skeleton:
        logger.error(f"No skeleton received for prompt: {prompt}")
        return

    try:
        combined_content = f"# {article_skeleton.get('H1_Title', '')}\n"
        for section in article_skeleton['H2_Subtitles']:
            h2_title = section.get('H2_Title', '')
            h3_subheader = section.get('H3_Subheader', '')
            section_content = await generate_section_content(session, h2_title, h3_subheader)
            combined_content += f"\n## {h2_title}\n### {h3_subheader}\n{section_content}"

        # Write the combined content to CSV file
        await writer.writerow([category, prompt, combined_content])
        logger.info(f"Processed content for {prompt}")

        if GENERATE_PDFS:
            # Create PDF for the entire article
            sanitized_title = sanitize_filename(article_skeleton.get('H1_Title', 'Article'))
            pdf_filename = f"Articles/{sanitized_title}.pdf"
            create_pdf(article_skeleton.get('H1_Title', ''), "", combined_content, pdf_filename)
            logger.info(f"Created PDF: {pdf_filename}")

    except Exception as e:
        logger.error(f"Error processing prompt '{prompt}': {e}")



def clean_csv_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        content = file.read()
        content = content.replace('\x00', '')  # Remove null characters

    with open(file_path, 'w', encoding='utf-8', errors='replace') as file:
        file.write(content)

def read_csv_file(file_path):
    return pd.read_csv(file_path, encoding='utf-8', dtype=str, on_bad_lines='skip')

async def main():
    logger.info("Starting up")

    # log sent prompts
    sent_prompts = set()

    # Setup CSV file
    output_file = 'output_results.csv'
    logger.info(f"Output file: {output_file}")

    # Change the input file to 'output_prompts.csv'
    input_file = 'C:\\aiseo\\SEOArticlegenAI\\output_prompts.csv'

    # Clean the input CSV file before processing
    clean_csv_file(input_file)

    # Load existing prompts from output CSV
    existing_prompts = set()
    if os.path.exists(output_file):
        existing_df = read_csv_file(output_file)
        existing_prompts.update(existing_df['prompt'].values)  # Add prompt to the set

    async with aiohttp.ClientSession() as session:
        # Open output CSV in append mode
        async with aiofiles.open(output_file, 'a', newline='', encoding='utf-8', errors='replace') as csvfile:
            writer = csv.writer(csvfile)

            # Write header only if the file is new (empty)
            if not existing_prompts:
                await writer.writerow(['category', 'prompt', 'title', 'subtitle', 'body'])

            input_df = read_csv_file(input_file)
            input_rows = input_df[input_df['prompt'].apply(lambda x: x not in existing_prompts)].to_numpy()

            # Combine all columns into a single string to form the prompt
            combined_input_rows = [(row[0], ' '.join(map(str, row[1:]))) for row in input_rows if str(row[1]) not in existing_prompts]

            # Process prompts that do not exist in the output CSV
            all_tasks = [process_prompt(session, writer, sent_prompts, row[0], row[1]) for row in input_rows]

            # Process tasks in a rolling manner, up to 60 concurrent requests
            concurrency_limit = 30
            semaphore = asyncio.Semaphore(concurrency_limit)

            async def process_with_semaphore(task):
                async with semaphore:
                    await task

            tasks_with_semaphore = [process_with_semaphore(task) for task in all_tasks]

            progress_bar = tqdm(total=len(all_tasks))
            for i, coro in enumerate(asyncio.as_completed(tasks_with_semaphore), 1):
                await coro
                progress_bar.update(1)
                logger.info(f"Completed {i}/{len(all_tasks)}")

            progress_bar.close()

    logger.info(f"Finished processing {len(all_tasks)} prompts in {input_file}. Results saved to {output_file}")

if __name__ == '__main__':
    asyncio.run(main())