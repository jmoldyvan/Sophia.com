"""
Escape The Room - A Text-Based Adventure Game
==============================================
A Crimson Room inspired escape game where the player must find items
and solve puzzles to escape a mysterious locked room.

Requirements fulfilled:
1. display_title() function for welcome/instructions
2. main() function controlling game flow
3. Regex validation for player name (re module)
4. if/elif/else statements for choices
5. Multiple additional functions (display_wall, interact_with_object, handle_keypad)
6. Player decisions stored in a list
7. Loop to replay the game
8. While/for loops and dictionary for outcomes
9. Game data saved to JSON file
"""

import re
import json
from datetime import datetime


# =============================================================================
# GAME DATA - Dictionaries for walls and object interactions
# =============================================================================

WALLS = {
    'south': {
        'description': "You face the SOUTH wall. You see the bed you woke up on.",
        'objects': ['bed', 'pillow'],
        'details': "A simple metal-frame bed with a thin mattress. A fluffy white pillow rests on top."
    },
    'west': {
        'description': "You face the WEST wall. You see a wooden table against the wall.",
        'objects': ['table', 'locked box'],
        'details': "A sturdy oak table with worn edges. On it sits a small metal box with a brass keyhole."
    },
    'north': {
        'description': "You face the NORTH wall. It's a blank concrete wall.",
        'objects': [],
        'details': "Nothing but cold, gray concrete. No windows, no vents, no hope... or is there?"
    },
    'east': {
        'description': "You face the EAST wall. This is where the exit must be.",
        'objects': ['door', 'keypad'],
        'details': "A heavy steel door with no visible handle. Next to it, an electronic keypad glows faintly red."
    }
}

# Outcomes dictionary mapping objects to their interaction results
OUTCOMES = {
    'bed': {
        'default': "You examine the bed frame. Cold metal, simple construction. The thin mattress offers no secrets... but what about that pillow?",
        'searched': "Just an ordinary bed. You've already checked it thoroughly.",
        'item': None,
        'requires': None
    },
    'pillow': {
        'default': "You lift the fluffy pillow and find a small BRASS KEY hidden underneath! Someone left this here...",
        'searched': "You've already found what was hidden here. Just an empty pillowcase now.",
        'item': 'brass key',
        'requires': None
    },
    'table': {
        'default': "A solid wooden table. Some scratches on the surface, but nothing useful. The locked box sitting on it seems more interesting.",
        'searched': "Still just a table. Focus on that box instead.",
        'item': None,
        'requires': None
    },
    'locked box': {
        'default': "You try to open the box but it's locked tight. There's a small keyhole on the front. You'll need a key.",
        'with_item': "You insert the brass key and turn it... *CLICK* The box springs open! Inside you find a crumpled NOTE with '4721' written on it. A CODE!",
        'searched': "The box is now open and empty. You already have the code.",
        'item': 'code 4721',
        'requires': 'brass key'
    },
    'door': {
        'default': "A reinforced steel door, cold to the touch. No handle, no hinges visible. The only way to open it must be through that keypad.",
        'searched': "The door remains sealed. You need the correct code for the keypad.",
        'item': None,
        'requires': None
    },
    'keypad': {
        'default': "A numeric keypad with digits 0-9. The display shows four blank spaces: [____]. It requires a 4-digit code.",
        'with_item': "KEYPAD_PUZZLE",  # Special flag to trigger puzzle
        'searched': "KEYPAD_PUZZLE",  # Always allow attempts if you have the code
        'item': None,
        'requires': 'code 4721'
    }
}


# =============================================================================
# FUNCTION DEFINITIONS
# =============================================================================

def display_title():
    """
    Display the welcome message and game instructions.
    This is called at the start of the game.
    """
    print("\n" + "=" * 62)
    print("  +-------------------------------------------------------+")
    print("  |         E S C A P E   T H E   R O O M                 |")
    print("  |            A Text Adventure Game                      |")
    print("  +-------------------------------------------------------+")
    print("=" * 62)
    print()
    print("  You wake up in a mysterious locked room with no memory")
    print("  of how you got here. Your only goal: ESCAPE!")
    print()
    print("  +==================== INSTRUCTIONS =====================+")
    print("  |                                                       |")
    print("  |  * Type a direction (north/south/east/west) to look   |")
    print("  |  * Type an object name to interact with it            |")
    print("  |  * Type 'inventory' to check your items               |")
    print("  |  * Type 'look' to see the current wall again          |")
    print("  |  * Type 'help' for commands                           |")
    print("  |  * Type 'quit' to give up                             |")
    print("  |                                                       |")
    print("  +=======================================================+")
    print()


def get_player_name():
    """
    Get and validate the player's name using a regular expression.
    Name must be 3-15 characters, start with a letter, and contain
    only letters, numbers, or underscores.
    
    Returns:
        str: The validated player name
    """
    # Regex pattern: starts with letter, 3-15 chars total, alphanumeric + underscore
    pattern = r'^[A-Za-z][A-Za-z0-9_]{2,14}$'
    
    print("Before we begin, we need your codename for the mission log.")
    print("(3-15 characters, must start with a letter, letters/numbers/underscores only)")
    
    while True:
        name = input("\nEnter your codename: ").strip()
        
        if re.match(pattern, name):
            return name
        else:
            print("[X] Invalid codename! Please follow the format:")
            print("    - Must be 3-15 characters long")
            print("    - Must start with a letter")
            print("    - Can only contain letters, numbers, and underscores")


def display_wall(direction, inventory):
    """
    Display the description of a wall and its interactable objects.
    
    Args:
        direction: The wall direction (north, south, east, west)
        inventory: The player's current inventory list
    
    Returns:
        list: The objects available on this wall
    """
    wall = WALLS[direction]
    
    print()
    print("-" * 55)
    print(f"  {wall['description']}")
    print("-" * 55)
    print()
    print(f"  {wall['details']}")
    print()
    
    # List interactable objects if any exist
    if wall['objects']:
        print("  You can interact with: ", end="")
        print(" | ".join([f"[{obj}]" for obj in wall['objects']]))
    else:
        print("  There's nothing to interact with here.")
    
    # Show inventory reminder if player has items
    if inventory:
        print(f"\n  Inventory: {', '.join(inventory)}")
    
    return wall['objects']


def interact_with_object(obj, room_state, inventory, decisions):
    """
    Handle player interaction with an object.
    
    Args:
        obj: The object name to interact with
        room_state: Dictionary tracking searched objects
        inventory: Player's inventory list
        decisions: List tracking player decisions
    
    Returns:
        bool: True if the player has escaped, False otherwise
    """
    obj = obj.lower().strip()
    
    # Check if object exists in our outcomes
    if obj not in OUTCOMES:
        print(f"\n  [?] You don't see '{obj}' here. Try looking at a wall first.")
        return False
    
    outcome = OUTCOMES[obj]
    decisions.append(f"Interacted with: {obj}")
    
    # Check if this object requires an item
    if outcome['requires']:
        required_item = outcome['requires']
        
        if required_item in inventory:
            # Player has the required item
            if outcome['with_item'] == "KEYPAD_PUZZLE":
                return handle_keypad(inventory, decisions)
            elif obj not in room_state['searched']:
                print(f"\n  {outcome['with_item']}")
                room_state['searched'].append(obj)
                if outcome['item']:
                    inventory.append(outcome['item'])
                    print(f"\n  *** {outcome['item'].upper()} added to inventory! ***")
            else:
                print(f"\n  {outcome['searched']}")
        else:
            # Player doesn't have required item
            print(f"\n  {outcome['default']}")
    else:
        # Object doesn't require anything special
        if obj not in room_state['searched']:
            print(f"\n  {outcome['default']}")
            room_state['searched'].append(obj)
            if outcome['item']:
                inventory.append(outcome['item'])
                print(f"\n  *** {outcome['item'].upper()} added to inventory! ***")
        else:
            print(f"\n  {outcome['searched']}")
    
    return False


def handle_keypad(inventory, decisions):
    """
    Handle the keypad puzzle sequence.
    
    Args:
        inventory: Player's inventory list
        decisions: List tracking player decisions
    
    Returns:
        bool: True if player enters correct code and escapes
    """
    print("\n  =========================================")
    print("  |        KEYPAD ACTIVATED               |")
    print("  =========================================")
    print("\n  The keypad beeps and the display lights up: [____]")
    
    # Hint if player has the code
    if 'code 4721' in inventory:
        print("  (You remember the code from the note: 4721)")
    
    max_attempts = 3
    attempts = 0
    
    while attempts < max_attempts:
        remaining = max_attempts - attempts
        code_input = input(f"\n  Enter 4-digit code ({remaining} attempts left): ").strip()
        decisions.append(f"Keypad attempt: {code_input}")
        
        # Validate input format
        if not re.match(r'^\d{4}$', code_input):
            print("  [!] Please enter exactly 4 digits.")
            continue
        
        attempts += 1
        
        if code_input == '4721':
            print("\n  *BEEP BEEP BEEP*")
            print()
            print("  +-------------------------------------------------------+")
            print("  |                                                       |")
            print("  |         THE DOOR SLIDES OPEN!                         |")
            print("  |                                                       |")
            print("  |   Bright light floods the room. Fresh air rushes in.  |")
            print("  |   You step through the doorway into freedom!          |")
            print("  |                                                       |")
            print("  |          * * *  YOU HAVE ESCAPED!  * * *              |")
            print("  |                                                       |")
            print("  +-------------------------------------------------------+")
            return True
        else:
            print("  [X] *BUZZ* Wrong code!")
    
    print("\n  The keypad locks temporarily. Keep exploring for clues...")
    return False


def display_help():
    """Display available commands to the player."""
    print("\n  +============== COMMANDS ================+")
    print("  |  north/south/east/west - Look at wall  |")
    print("  |  [object name] - Interact with object  |")
    print("  |  inventory - Check your items          |")
    print("  |  look - See current wall again         |")
    print("  |  hint - Get a small hint               |")
    print("  |  quit - Give up and exit               |")
    print("  +=========================================+")


def get_hint(room_state, inventory):
    """
    Provide a contextual hint based on game progress.
    
    Args:
        room_state: Dictionary tracking searched objects
        inventory: Player's inventory list
    
    Returns:
        str: A hint message
    """
    # Check progress and give appropriate hint
    if 'brass key' not in inventory:
        return "Hint: The bed looks comfortable... especially that pillow."
    elif 'code 4721' not in inventory:
        return "Hint: You have a key. What could it unlock? Check the west wall."
    else:
        return "Hint: You have everything you need! The keypad on the east wall awaits..."


def save_game_data(player_name, decisions, outcome, inventory):
    """
    Save the game data to a JSON file.
    
    Args:
        player_name: The player's codename
        decisions: List of all player decisions
        outcome: Final game outcome (ESCAPED or GAVE_UP)
        inventory: Final inventory state
    """
    game_data = {
        'player_name': player_name,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'outcome': outcome,
        'total_actions': len(decisions),
        'final_inventory': inventory,
        'decisions': decisions
    }
    
    filename = f"escape_room_save_{player_name}.json"
    
    with open(filename, 'w') as file:
        json.dump(game_data, file, indent=4)
    
    print(f"\n  Game data saved to '{filename}'")


def main():
    """
    Main function controlling the game flow.
    Handles the game loop, player input, and replay functionality.
    """
    # Display title and get player name
    display_title()
    player_name = get_player_name()
    
    print(f"\n  Welcome, Agent {player_name}.")
    print("  Your mission briefing: Find a way out. Trust no one.")
    
    # Outer loop for replay functionality
    play_again = True
    
    while play_again:
        # Initialize game state for new game
        inventory = []          # Stores collected items
        decisions = []          # Stores all player decisions (requirement #6)
        room_state = {
            'searched': []      # Tracks which objects have been searched
        }
        current_direction = None
        escaped = False
        
        # Opening narrative
        print("\n" + "-" * 55)
        print("  Your eyes flutter open. Harsh fluorescent light.")
        print("  You're lying on a hard bed in a small, windowless room.")
        print("  Your head throbs. How did you get here?")
        print("  ")
        print("  You stand up slowly. Four walls surround you.")
        print("  There must be a way out...")
        print("-" * 55)
        print("\n  Which direction do you want to look?")
        print("  (north / south / east / west)")
        
        # Main game loop
        while not escaped:
            # Get player input
            command = input("\n  > ").strip().lower()
            
            # Record decision
            if command:
                decisions.append(f"Command: {command}")
            
            # Process commands using if/elif/else (requirement #4)
            if command == 'quit':
                print("\n  You slump against the wall, defeated...")
                print("  The room has won... this time.")
                break
            
            elif command == 'inventory':
                if inventory:
                    print(f"\n  Inventory: {', '.join(inventory)}")
                else:
                    print("\n  Your pockets are empty.")
            
            elif command == 'help':
                display_help()
            
            elif command == 'hint':
                print(f"\n  {get_hint(room_state, inventory)}")
            
            elif command == 'look':
                if current_direction:
                    display_wall(current_direction, inventory)
                else:
                    print("\n  Choose a direction first: north, south, east, or west")
            
            elif command in ['north', 'south', 'east', 'west']:
                current_direction = command
                display_wall(command, inventory)
            
            elif command == '':
                print("  Type a command or 'help' for options.")
            
            else:
                # Assume it's an object interaction
                if current_direction:
                    # Get valid objects for current wall
                    valid_objects = WALLS[current_direction]['objects']
                    
                    # Check if command matches any object
                    if command in [obj.lower() for obj in valid_objects]:
                        escaped = interact_with_object(command, room_state, inventory, decisions)
                    else:
                        # Check if it's a valid object but on different wall
                        all_objects = []
                        for wall in WALLS.values():
                            all_objects.extend([o.lower() for o in wall['objects']])
                        
                        if command in all_objects:
                            print(f"\n  [?] You don't see '{command}' on this wall.")
                            print("      Try facing a different direction.")
                        else:
                            print(f"\n  [?] '{command}' isn't something you can interact with.")
                else:
                    print("\n  Choose a direction first: north, south, east, or west")
        
        # Game ended - save data (requirement #9)
        outcome = "ESCAPED" if escaped else "GAVE_UP"
        save_game_data(player_name, decisions, outcome, inventory)
        
        # Display final stats
        print("\n  ============== GAME STATS ==============")
        print(f"   Agent: {player_name}")
        print(f"   Outcome: {outcome}")
        print(f"   Total Actions: {len(decisions)}")
        print(f"   Items Found: {len(inventory)}")
        print("  =========================================")
        
        # Replay prompt (requirement #7)
        print("\n  Would you like to try again?")
        replay_input = input("  (yes/no): ").strip().lower()
        play_again = replay_input in ['yes', 'y']
        
        if play_again:
            print("\n  Resetting the room...")
    
    print(f"\n  Thanks for playing, Agent {player_name}!")
    print("  Until next time... if there is a next time.\n")



if __name__ == "__main__":
    main()