import os
import re
import json
import base64

import requests


def change_model(settings):
    """
    Changes the selected model by making requests to the provided API endpoint.

    :param settings: A dictionary containing 'backend_api' and 'model_name' keys.
    """
    # Extract the URL and the new model name from settings
    url = settings.get('backend_api')
    new_model_name = settings.get('model_name')

    # Safety check to ensure the required settings are provided
    if not url or not new_model_name:
        print("Error: 'backend_api' and 'model_name' must be provided in settings.")
        return

    try:
        # Fetch current options
        opt_response = requests.get(url=f'{url}/sdapi/v1/options')
        opt_response.raise_for_status()  # Raises an exception for 4XX/5XX errors

        # Update the model checkpoint name in the options
        opt_json = opt_response.json()
        print(f"Current model name: {opt_json['sd_model_checkpoint']}")
        opt_json['sd_model_checkpoint'] = new_model_name

        # Post the updated options back to the server
        update_response = requests.post(url=f'{url}/sdapi/v1/options', json=opt_json)
        update_response.raise_for_status()  # Raises an exception for 4XX/5XX errors

        # check if the model was set correctly
        opt_response = requests.get(url=f'{url}/sdapi/v1/options')
        opt_response.raise_for_status()  # Raises an exception for 4XX/5XX errors

        # Update the model checkpoint name in the options
        opt_json = opt_response.json()
        print(f"Model name AFTER CHANGE: {opt_json['sd_model_checkpoint']}")

        print(f"Model name changed successfully to: {new_model_name}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to change the model name. Error: {e}")


def sanitize_for_path(value):
    """
    Sanitizes a string to make it suitable for use as a directory or file name.
    Removes or replaces characters that may not be allowed in file names.
    """
    return "".join(c if c.isalnum() or c in " _-" else "_" for c in value)


def existing_files_overwrite_protection():
    full_model_name = settings.get('model_name')
    model_name = full_model_name.split('.safetensors')[0]
    main_directory = f"generations_on_{model_name}"

    if os.path.exists(main_directory):
        print(f"Directory {main_directory} already exists.")
        response = input("Are you aware that main directory exists for this model.\n"
                         "By proceeding further you would certainly fuck up the whole thing.\n"
                         "(y/n) where y == im dumb and n == exit the program and make a backup:\n"
                         )

        if response.lower() != "y":
            print("Exiting...")
            exit(0)


def generate_image(prompt, settings, seed):
    full_model_name = settings.get('model_name')
    model_name = full_model_name.split('.safetensors')[0]
    main_directory = f"generations_on_{model_name}"
    prompt_directory = sanitize_for_path(prompt[:100])  # Using first 100 chars of prompt for directory name

    output_directory = os.path.join(main_directory, prompt_directory)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    api_url = settings.get("backend_api")
    response = requests.post(
        f"{api_url}/sdapi/v1/txt2img",
        headers={"Content-type": "application/json"},
        json={
            "seed": seed,
            "prompt": prompt,
            "width": settings.get("width", 1024),
            "height": settings.get("height", 1024),
            "cfg_scale": settings.get("cfg_scale", 7),
            "steps": settings.get("steps", 25),
            "sampler_name": settings.get("sampler_name", "DPM++ 2M Karras"),
            "negative_prompt": settings.get("negative_prompt", ""),
        },
    )

    if response.status_code == 200:
        json_data = response.json()
        images_data = json_data["images"]

        for i, img_data in enumerate(images_data):
            img_bytes = base64.b64decode(img_data)
            # Updated code to calculate max_index with consideration for the new filename format
            # Use regex to find all instances of the pattern "image_[index]_seed_[seed].png" in the filenames
            file_indices = [int(re.search(r"image_(\d+)_seed", filename).group(1)) for filename in
                            os.listdir(output_directory) if re.search(r"image_\d+_seed", filename)]
            max_index = max(file_indices, default=0)  # Use the max index found, default to 0 if none found

            img_path = os.path.join(output_directory, f"image_{max_index + i + 1}_seed_{seed}.png")
            with open(img_path, "wb") as f:
                f.write(img_bytes)
            print(f"Saved: {img_path}")
    else:
        print(f"Failed to get a response from {api_url}, status code: {response.status_code}")


def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


# Prompt user for the settings file name
settings_file_name = input("Please enter the name of the settings JSON file (e.g., settings_1.json): ")

# Constructing the full file path assuming the file is in the current directory
settings_file_path = os.path.join(os.getcwd(), settings_file_name)

# Load settings
if os.path.exists(settings_file_path):
    settings = load_json(settings_file_path)
else:
    print(f"Settings file not found: {settings_file_path}")
    exit(1)

# Check if the output directory already exists and prompt the user to confirm overwriting
existing_files_overwrite_protection()

# Load seeds from random_seeds.txt
with open("random_seeds.txt", "r") as file:
    seeds = [int(line.strip()) for line in file.readlines()]

# Load the prompts from prompts.json
with open('prompts.json', 'r') as file:
    prompts_data = json.load(file)

# Change the model to the one specified in the settings
change_model(settings)

prompts = prompts_data['prompts']
seed_index = 0  # Start from the first seed

for prompt in prompts:
    for _ in range(settings.get("number_of_generations_per_prompt", 1)):
        # Make sure we don't run out of seeds
        if seed_index < len(seeds):
            seed = seeds[seed_index]
            generate_image(prompt, settings, seed)
            seed_index += 1
        else:
            print("Ran out of seeds.")
    seed_index = 0  # Reset seed index for the next prompt
