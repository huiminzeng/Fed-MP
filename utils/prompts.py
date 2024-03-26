def get_prompts(attributes):
    prompts = []
    for attribute in attributes:
        prompts.append("A photo of " + attribute + '.')
    return prompts