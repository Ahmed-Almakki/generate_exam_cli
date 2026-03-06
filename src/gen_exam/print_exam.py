def print_exam(response: str) -> None:
    """
    Print the generated exam response in a readable format.
    Args:
        response (str): The generated exam response as a string.
    Returns:
        None
    """
    lines = response.split('\n')
    questions_only = []
    answers_only = []

    for line in lines:
        if line.strip().startswith("Correct Answer:"):
            # Extract the answer and number it based on how many we've found
            answer_num = len(answers_only) + 1
            answers_only.append(f"Question {answer_num} {line.strip()}")
        else:
            questions_only.append(line)

    # Join the text back together
    formatted_questions = "\n".join(questions_only)
    formatted_answers = "\n".join(answers_only)

    # Write to a text file with a "page break" between them
    with open("Generated_Exam.txt", "w", encoding="utf-8") as file:
        file.write("=== EXAM QUESTIONS ===\n\n")
        file.write(formatted_questions.strip())
        file.write("\n\n" + "="*40 + "\n\n") # Visual separator
        file.write("=== ANSWER KEY ===\n\n")
        file.write(formatted_answers.strip())

    print("\n✅ Exam successfully saved to 'Generated_Exam.txt'!")