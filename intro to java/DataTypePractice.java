/**
 * Java Data Types Practice Problems
 *
 * Instructions: Complete each exercise by replacing the comments with your code.
 * Run the program to check your answers!
 */
public class DataTypePractice {
    public static void main(String[] args) {

        System.out.println("=== Java Data Types Practice ===\n");

        // ============================================
        // EXERCISE 1: Integer Types
        // ============================================
        // Declare an int variable called "age" and set it to your age
        // YOUR CODE HERE:
        int age = 31;

        // Declare a long variable called "worldPopulation" and set it to 8000000000L
        // (Note: add 'L' at the end for long literals)
        // YOUR CODE HERE:
        long worldPopulation = 800000000L;

        // Uncomment the next lines when you've completed the exercise:
        System.out.println("Exercise 1 - Integers:");
        System.out.println("Age: " + age);
        System.out.println("World Population: " + worldPopulation);
        System.out.println();


        // ============================================
        // EXERCISE 2: Decimal Types
        // ============================================
        // Declare a double variable called "pi" and set it to 3.14159
        // YOUR CODE HERE:

        double pi = 3.1234;
        // Declare a float variable called "price" and set it to 19.99f
        // (Note: add 'f' at the end for float literals)
        // YOUR CODE HERE:
        float price = 19.99f;

        // Uncomment when done:
        System.out.println("Exercise 2 - Decimals:");
        System.out.println("Pi: " + pi);
        System.out.println("Price: $" + price);
        System.out.println();


        // ============================================
        // EXERCISE 3: Boolean Type
        // ============================================
        // Declare a boolean variable called "isJavaFun" and set it to true
        // YOUR CODE HERE:
        boolean isJavaFun = true;

        // Declare a boolean variable called "isRaining" and set it to false
        // YOUR CODE HERE:
        boolean isRaining = false;

        // Uncomment when done:
        System.out.println("Exercise 3 - Booleans:");
        System.out.println("Is Java fun? " + isJavaFun);
        System.out.println("Is it raining? " + isRaining);
        System.out.println();


        // ============================================
        // EXERCISE 4: Character Type
        // ============================================
        // Declare a char variable called "firstInitial" and set it to your first initial
        // (Remember: use single quotes for char, like 'A')
        // YOUR CODE HERE:


        // Declare a char variable called "grade" and set it to 'A'
        // YOUR CODE HERE:


        // Uncomment when done:
        // System.out.println("Exercise 4 - Characters:");
        // System.out.println("First Initial: " + firstInitial);
        // System.out.println("Grade: " + grade);
        // System.out.println();


        // ============================================
        // EXERCISE 5: String Type
        // ============================================
        // Declare a String variable called "firstName" and set it to your first name
        // (Remember: use double quotes for String, like "Hello")
        // YOUR CODE HERE:


        // Declare a String variable called "favoriteFood" and set it to your favorite food
        // YOUR CODE HERE:


        // Uncomment when done:
        // System.out.println("Exercise 5 - Strings:");
        // System.out.println("First Name: " + firstName);
        // System.out.println("Favorite Food: " + favoriteFood);
        // System.out.println();


        // ============================================
        // EXERCISE 6: Type Conversion (Casting)
        // ============================================
        // Convert this double to an int using casting: (int)
        double myDouble = 9.78;
        // Declare an int called "myInt" and cast myDouble to it
        // YOUR CODE HERE:
        int myInt = (int) myDouble;

        // Uncomment when done:
        // System.out.println("Exercise 6 - Type Casting:");
        // System.out.println("Original double: " + myDouble);
        // System.out.println("After casting to int: " + myInt);
        // System.out.println();


        // ============================================
        // EXERCISE 7: Simple Math with Variables
        // ============================================
        int num1 = 10;
        int num2 = 3;

        // Declare an int called "sum" that adds num1 and num2
        // YOUR CODE HERE:
        int sum = num1 + num2;

        // Declare an int called "difference" that subtracts num2 from num1
        // YOUR CODE HERE:


        // Declare an int called "product" that multiplies num1 and num2
        // YOUR CODE HERE:


        // Declare an int called "quotient" that divides num1 by num2
        // YOUR CODE HERE:


        // Uncomment when done:
        // System.out.println("Exercise 7 - Math Operations:");
        // System.out.println("num1 = " + num1 + ", num2 = " + num2);
        // System.out.println("Sum: " + sum);
        // System.out.println("Difference: " + difference);
        // System.out.println("Product: " + product);
        // System.out.println("Quotient: " + quotient);
        // System.out.println();


        // ============================================
        // BONUS: String Concatenation
        // ============================================
        String greeting = "Hello";
        String name = "Java Learner";

        // Create a String called "message" that combines greeting and name
        // with a space between them (should be "Hello Java Learner")
        // YOUR CODE HERE:
        String message = greeting + name;

        // Uncomment when done:
        System.out.println("Bonus - String Concatenation:");
        System.out.println(message);


        System.out.println("=== Great job completing the exercises! ===");
    }
}
