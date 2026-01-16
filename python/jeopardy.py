# Jeopardy Game
# A trivia game where players pick categories and dollar values.
# Correct answers add to score, wrong answers subtract.
# Game ends when score hits $0 or all questions are answered.

import random, sys

# List of all category names displayed on the board
CATEGORIES = ['SCIENCE', 'HISTORY', 'POP CULTURE', 'GEOGRAPHY', 'WORD PLAY']

# Each question has: category, value (100-400), question text, answer, and accepted keywords
QUESTIONS = [
    # SCIENCE - 100 to 400
    {'category': 'SCIENCE', 'value': 100,
     'question': "This planet is known as the Red Planet.",
     'answer': "What is Mars?",
     'accept': ['mars']},
    {'category': 'SCIENCE', 'value': 200,
     'question': "This is the chemical symbol for gold.",
     'answer': "What is Au?",
     'accept': ['au']},
    {'category': 'SCIENCE', 'value': 300,
     'question': "This force keeps us grounded on Earth.",
     'answer': "What is gravity?",
     'accept': ['gravity']},
    {'category': 'SCIENCE', 'value': 400,
     'question': "This is the hardest natural substance on Earth.",
     'answer': "What is diamond?",
     'accept': ['diamond']},

    # HISTORY - 100 to 400
    {'category': 'HISTORY', 'value': 100,
     'question': "This ship sank on its maiden voyage in 1912.",
     'answer': "What is the Titanic?",
     'accept': ['titanic']},
    {'category': 'HISTORY', 'value': 200,
     'question': "This ancient wonder was located in Giza, Egypt.",
     'answer': "What are the Pyramids?",
     'accept': ['pyramid', 'pyramids']},
    {'category': 'HISTORY', 'value': 300,
     'question': "This wall fell in 1989, reuniting East and West Germany.",
     'answer': "What is the Berlin Wall?",
     'accept': ['berlin']},
    {'category': 'HISTORY', 'value': 400,
     'question': "This explorer is credited with discovering America in 1492.",
     'answer': "Who is Christopher Columbus?",
     'accept': ['columbus', 'christopher']},

    # POP CULTURE - 100 to 400
    {'category': 'POP CULTURE', 'value': 100,
     'question': "This wizard attended Hogwarts School of Witchcraft and Wizardry.",
     'answer': "Who is Harry Potter?",
     'accept': ['harry', 'potter']},
    {'category': 'POP CULTURE', 'value': 200,
     'question': "This streaming service is known for originals like Stranger Things.",
     'answer': "What is Netflix?",
     'accept': ['netflix']},
    {'category': 'POP CULTURE', 'value': 300,
     'question': "This band was known as the Fab Four.",
     'answer': "Who are The Beatles?",
     'accept': ['beatles']},
    {'category': 'POP CULTURE', 'value': 400,
     'question': "This superhero is also known as the Dark Knight.",
     'answer': "Who is Batman?",
     'accept': ['batman', 'bruce wayne']},

    # GEOGRAPHY - 100 to 400
    {'category': 'GEOGRAPHY', 'value': 100,
     'question': "This is the largest ocean on Earth.",
     'answer': "What is the Pacific Ocean?",
     'accept': ['pacific']},
    {'category': 'GEOGRAPHY', 'value': 200,
     'question': "This river flows through Egypt.",
     'answer': "What is the Nile?",
     'accept': ['nile']},
    {'category': 'GEOGRAPHY', 'value': 300,
     'question': "This country is home to the Great Barrier Reef.",
     'answer': "What is Australia?",
     'accept': ['australia']},
    {'category': 'GEOGRAPHY', 'value': 400,
     'question': "This mountain is the tallest in the world.",
     'answer': "What is Mount Everest?",
     'accept': ['everest']},

    # WORD PLAY - 100 to 400
    {'category': 'WORD PLAY', 'value': 100,
     'question': "This word is spelled wrong in every dictionary.",
     'answer': "What is 'wrong'?",
     'accept': ['wrong']},
    {'category': 'WORD PLAY', 'value': 200,
     'question': "This begins with 'e', ends with 'e', but only has one letter in it.",
     'answer': "What is an envelope?",
     'accept': ['envelope']},
    {'category': 'WORD PLAY', 'value': 300,
     'question': "This word becomes shorter when you add two letters to it.",
     'answer': "What is 'short'?",
     'accept': ['short']},
    {'category': 'WORD PLAY', 'value': 400,
     'question': "This five-letter word becomes shorter when you add two letters.",
     'answer': "What is 'short'? (short + er = shorter)",
     'accept': ['short']},
]

# Messages displayed when player answers correctly
CORRECT_TEXT = ['Correct!', 'That is right.', "You're right.",
                'You got it.', 'Righto!']
# Messages displayed when player answers incorrectly
INCORRECT_TEXT = ['Incorrect!', "Nope, that isn't it.", 'Nope.',
                  'Not quite.', 'You missed it.']


# --- HELPER FUNCTIONS ---

def display_board(available_questions):
    """Display the Jeopardy board with available questions."""
    print('\n' + '=' * 60)
    values = [100, 200, 300, 400]

    # Print category headers
    for cat in CATEGORIES:
        print(f'{cat:^12}', end='')
    print()
    print('-' * 60)

    # Print values for each row
    for value in values:
        for cat in CATEGORIES:
            # Check if this question is still available
            available = any(q['category'] == cat and q['value'] == value
                          for q in available_questions)
            if available:
                print(f'{"$" + str(value):^12}', end='')
            else:
                print(f'{"---":^12}', end='')
        print()
    print('=' * 60)


def get_question(available_questions, category, value):
    """Get a question by category and value."""
    for q in available_questions:
        if q['category'] == category and q['value'] == value:
            return q
    return None


# --- MAIN PROGRAM ---

# Display welcome message and game instructions
print('''
Welcome to Jeopardy!
(Enter QUIT to quit at any time.)

Pick a category and dollar amount.
- Correct answers ADD that amount to your score.
- Wrong answers SUBTRACT that amount from your score.
- If your score reaches $0 or below, the game is over!
''')

input('Press Enter to begin...')

# Initialize game state
score = 1000  # Starting score
available_questions = QUESTIONS.copy()  # Copy so we can remove answered questions

# --- MAIN GAME LOOP ---
# Continue until all questions answered or score hits $0
while available_questions:
    print('\n' * 40)
    print(f'Current Score: ${score}')

    # Check for game over condition (score hit $0 or below)
    if score <= 0:
        print('\nYour score hit $0! GAME OVER!')
        print(f'Questions answered: {len(QUESTIONS) - len(available_questions)} / {len(QUESTIONS)}')
        sys.exit()

    # Show the current game board
    display_board(available_questions)

    # --- CATEGORY SELECTION ---
    print('\nAvailable categories:', ', '.join(CATEGORIES))
    category_input = input('Choose a category: ').strip().upper()

    if category_input == 'QUIT':
        print('Thanks for playing!')
        print(f'Final Score: ${score}')
        sys.exit()

    # Validate category input - keep asking until valid
    if category_input not in CATEGORIES:
        print(f"Invalid category. Please choose one of: {', '.join(CATEGORIES)}")
        category_input = input('Choose a category: ').strip().upper()
        while category_input not in CATEGORIES and category_input != 'QUIT':
            print(f"Invalid category. Please choose one of: {', '.join(CATEGORIES)}")
            category_input = input('Choose a category: ').strip().upper()
        if category_input == 'QUIT':
            print('Thanks for playing!')
            print(f'Final Score: ${score}')
            sys.exit()

    matched_category = category_input

    # --- VALUE SELECTION ---
    value_input = input('Choose a value (100, 200, 300, 400): ').strip()

    # Validate value input - keep asking until valid
    valid_value = False
    while not valid_value:
        if value_input.upper() == 'QUIT':
            print('Thanks for playing!')
            print(f'Final Score: ${score}')
            sys.exit()
        try:
            # Try to convert input to integer (remove $ if present)
            value = int(value_input.replace('$', ''))
            if value in [100, 200, 300, 400]:
                valid_value = True
            else:
                print("Value must be 100, 200, 300, or 400.")
                value_input = input('Choose a value (100, 200, 300, 400): ').strip()
        except ValueError:
            print("Invalid value. Please enter 100, 200, 300, or 400.")
            value_input = input('Choose a value (100, 200, 300, 400): ').strip()

    # --- FIND AND DISPLAY QUESTION ---
    question = get_question(available_questions, matched_category, value)

    # If question was already answered, go back to start of loop
    if not question:
        print("That question has already been answered. Choose another.")
        continue

    # Display the question to the player
    print(f'\n{matched_category} for ${value}:')
    print(f'QUESTION: {question["question"]}')
    response = input('  ANSWER: ').lower()

    if response == 'quit':
        print('Thanks for playing!')
        print(f'Final Score: ${score}')
        sys.exit()

    # --- CHECK ANSWER ---
    # Look for any accepted keyword in the player's response
    correct = False
    for acceptanceWord in question['accept']:
        if acceptanceWord in response:
            correct = True
            break

    # --- UPDATE GAME STATE ---
    # Remove this question from available questions (can't be picked again)
    available_questions.remove(question)

    # Update score based on whether answer was correct or wrong
    if correct:
        score += value  # Add value to score for correct answer
        text = random.choice(CORRECT_TEXT)
        print(f'\n{text} {question["answer"]}')
        print(f'+${value}! New score: ${score}')
    else:
        score -= value  # Subtract value from score for wrong answer
        text = random.choice(INCORRECT_TEXT)
        print(f'\n{text} The answer is: {question["answer"]}')
        print(f'-${value}! New score: ${score}')

    input('\nPress Enter to continue...')

# --- GAME COMPLETE ---
# Player answered all questions without hitting $0
print("\nCongratulations! You've answered all the questions!")
print(f'Final Score: ${score}')