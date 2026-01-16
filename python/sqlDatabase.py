import sqlite3


def connect_to_database(db_name="students.db"):
    """
    Connect to the SQLite database. Creates the file if it doesn't exist.
    
    Args:
        db_name: Name of the database file (default: students.db)
    
    Returns:
        sqlite3.Connection: Database connection object
    """
    try:
        connection = sqlite3.connect(db_name)
        print(f"Connected to database: {db_name}")
        return connection
    except sqlite3.Error as e:
        print(f"[ERROR] Failed to connect to database: {e}")
        return None


def create_table(connection):
    """
    Create the students table if it doesn't already exist.
    
    Args:
        connection: SQLite database connection
    """
    try:
        cursor = connection.cursor()
        
        # Create table with id, name, grade, and email columns
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                grade TEXT NOT NULL,
                email TEXT NOT NULL
            )
        ''')
        
        connection.commit()
        print("Table 'students' is ready.")
        
    except sqlite3.Error as e:
        print(f"[ERROR] Failed to create table: {e}")


def validate_email(email):
    """
    Validate that an email address contains the '@' symbol.
    
    Args:
        email: Email string to validate
    
    Returns:
        bool: True if valid, False otherwise
    """
    if '@' in email and len(email) >= 3:
        return True
    return False


def validate_integer(value):
    """
    Validate that a value can be converted to an integer.
    
    Args:
        value: Value to validate
    
    Returns:
        int or None: Integer value if valid, None otherwise
    """
    try:
        return int(value)
    except ValueError:
        return None


def get_valid_id(prompt="Enter student ID: "):
    """
    Prompt user for a valid integer ID.
    
    Args:
        prompt: Message to display to user
    
    Returns:
        int: Valid integer ID
    """
    while True:
        user_input = input(prompt).strip()
        
        student_id = validate_integer(user_input)
        
        if student_id is not None:
            return student_id
        else:
            print("[!] Invalid input. Please enter a valid integer ID.")


def get_valid_email(prompt="Enter email: "):
    """
    Prompt user for a valid email address.
    
    Args:
        prompt: Message to display to user
    
    Returns:
        str: Valid email address
    """
    while True:
        email = input(prompt).strip()
        
        if validate_email(email):
            return email
        else:
            print("[!] Invalid email. Please include an '@' symbol.")


def confirm_action(message="Are you sure? (yes/no): "):
    """
    Prompt user to confirm an action.
    
    Args:
        message: Confirmation message
    
    Returns:
        bool: True if confirmed, False otherwise
    """
    response = input(message).strip().lower()
    return response in ['yes', 'y']


def add_student(connection):
    """
    Add a new student record to the database.
    
    Args:
        connection: SQLite database connection
    """
    print("\n--- Add New Student ---")
    
    try:
        # Get student information with validation
        student_id = get_valid_id("Enter student ID: ")
        name = input("Enter student name: ").strip()
        
        # Validate name is not empty
        if not name:
            print("[!] Name cannot be empty. Operation cancelled.")
            return
        
        grade = input("Enter student grade: ").strip()
        
        # Validate grade is not empty
        if not grade:
            print("[!] Grade cannot be empty. Operation cancelled.")
            return
        
        email = get_valid_email("Enter student email: ")
        
        # Insert the record into the database
        cursor = connection.cursor()
        cursor.execute('''
            INSERT INTO students (id, name, grade, email)
            VALUES (?, ?, ?, ?)
        ''', (student_id, name, grade, email))
        
        connection.commit()
        print(f"[OK] Student '{name}' added successfully!")
        
    except sqlite3.IntegrityError:
        print(f"[ERROR] A student with ID {student_id} already exists.")
    except sqlite3.Error as e:
        print(f"[ERROR] Database error: {e}")


def view_all_students(connection):
    """
    Display all student records from the database.
    
    Args:
        connection: SQLite database connection
    """
    print("\n--- All Student Records ---")
    
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT id, name, grade, email FROM students ORDER BY id")
        
        students = cursor.fetchall()
        
        if not students:
            print("No student records found.")
            return
        
        # Display header
        print("-" * 70)
        print(f"{'ID':<8} {'Name':<20} {'Grade':<10} {'Email':<30}")
        print("-" * 70)
        
        # Display each student record
        for student in students:
            student_id, name, grade, email = student
            print(f"{student_id:<8} {name:<20} {grade:<10} {email:<30}")
        
        print("-" * 70)
        print(f"Total students: {len(students)}")
        
    except sqlite3.Error as e:
        print(f"[ERROR] Failed to retrieve records: {e}")


def update_student(connection):
    """
    Update an existing student's information.
    
    Args:
        connection: SQLite database connection
    """
    print("\n--- Update Student Record ---")
    
    try:
        # Get the ID of the student to update
        student_id = get_valid_id("Enter the ID of the student to update: ")
        
        # Check if student exists
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM students WHERE id = ?", (student_id,))
        student = cursor.fetchone()
        
        if not student:
            print(f"[!] No student found with ID {student_id}.")
            return
        
        # Display current information
        print(f"\nCurrent record:")
        print(f"  ID: {student[0]}")
        print(f"  Name: {student[1]}")
        print(f"  Grade: {student[2]}")
        print(f"  Email: {student[3]}")
        
        # Get new information (press Enter to keep current value)
        print("\nEnter new values (press Enter to keep current value):")
        
        new_name = input(f"  Name [{student[1]}]: ").strip()
        new_name = new_name if new_name else student[1]
        
        new_grade = input(f"  Grade [{student[2]}]: ").strip()
        new_grade = new_grade if new_grade else student[2]
        
        new_email = input(f"  Email [{student[3]}]: ").strip()
        if new_email:
            # Validate new email if provided
            if not validate_email(new_email):
                print("[!] Invalid email format. Keeping original email.")
                new_email = student[3]
        else:
            new_email = student[3]
        
        # Update the record
        cursor.execute('''
            UPDATE students
            SET name = ?, grade = ?, email = ?
            WHERE id = ?
        ''', (new_name, new_grade, new_email, student_id))
        
        connection.commit()
        print(f"[OK] Student record updated successfully!")
        
    except sqlite3.Error as e:
        print(f"[ERROR] Failed to update record: {e}")


def delete_student(connection):
    """
    Delete a student record from the database.
    
    Args:
        connection: SQLite database connection
    """
    print("\n--- Delete Student Record ---")
    
    try:
        # Get the ID of the student to delete
        student_id = get_valid_id("Enter the ID of the student to delete: ")
        
        # Check if student exists
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM students WHERE id = ?", (student_id,))
        student = cursor.fetchone()
        
        if not student:
            print(f"[!] No student found with ID {student_id}.")
            return
        
        # Display the record to be deleted
        print(f"\nStudent to delete:")
        print(f"  ID: {student[0]}")
        print(f"  Name: {student[1]}")
        print(f"  Grade: {student[2]}")
        print(f"  Email: {student[3]}")
        
        # Confirm deletion
        if confirm_action("\nAre you sure you want to delete this record? (yes/no): "):
            cursor.execute("DELETE FROM students WHERE id = ?", (student_id,))
            connection.commit()
            print(f"[OK] Student record deleted successfully!")
        else:
            print("Deletion cancelled.")
        
    except sqlite3.Error as e:
        print(f"[ERROR] Failed to delete record: {e}")


def display_menu():
    """Display the main menu options."""
    print("\n" + "=" * 40)
    print("   STUDENT RECORDS MANAGEMENT SYSTEM")
    print("=" * 40)
    print("  1. Add a new student")
    print("  2. View all students")
    print("  3. Update a student record")
    print("  4. Delete a student record")
    print("  5. Exit")
    print("=" * 40)


def main():
    """
    Main function that runs the application.
    Handles the menu loop and user interaction.
    """
    print("\nWelcome to the Student Records Management System!")
    print("-" * 50)
    
    # Connect to the database
    connection = connect_to_database("students.db")
    
    if connection is None:
        print("Failed to start application. Exiting.")
        return
    
    # Create the students table if it doesn't exist
    create_table(connection)
    
    # Main menu loop
    while True:
        display_menu()
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            add_student(connection)
        
        elif choice == '2':
            view_all_students(connection)
        
        elif choice == '3':
            update_student(connection)
        
        elif choice == '4':
            delete_student(connection)
        
        elif choice == '5':
            # Confirm exit
            if confirm_action("Are you sure you want to exit? (yes/no): "):
                print("\nClosing database connection...")
                connection.close()
                print("Goodbye!")
                break
            else:
                print("Returning to menu...")
        
        else:
            print("[!] Invalid choice. Please enter a number between 1 and 5.")


if __name__ == "__main__":
    main()