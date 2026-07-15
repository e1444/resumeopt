if __name__ == "__main__":
    extract(...)  # extract skills, uses llm
    # one thing to consider is that the extracted skills may not be in a standardized format, so some post-processing may be needed to clean up the output and ensure consistency
    
    match(...)  # match extracted skills to available skills
    # implementation details unclear
    # could use vector embeddings and cosine similarity to match extracted skills to a predefined list of skills
    # could also use a more sophisticated approach like fine-tuning a model on a labeled dataset of skills to improve matching accuracy
    
    generate_skills(...)  # generate skills in yaml format
    # implementation details unclear
    # could just use a list of skills and output them in yaml format
    # the skills section in the resume is relatively simple and doesn't require complex logic, so a straightforward approach should suffice
    # score skills, so that we have some sort of ordering of the skills based on relevance or importance, which can be used to prioritize the skills in the resume
    
    validate_skills(...)  # make sure that the generated skills are valid and relevant, there are no duplicates or miscategorized skills
    # uses llm
    
    format(...)  # insert skills into .tex template
    
    render(...)  # render .tex to .pdf
    
    validate_pdf(...)  # validate that the generated pdf is correct and matches the expected output, e.g 1 page
    # tex provides the necessary tools