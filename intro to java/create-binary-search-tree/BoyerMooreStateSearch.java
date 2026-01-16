import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

/**
 * BoyerMooreStateSearch Application
 * 
 * This application uses the Boyer-Moore algorithm with the bad character rule
 * to search for patterns within the names of the 50 US states.
 */
public class BoyerMooreStateSearch {

    // Array containing the names of all 50 US states
    private static final String[] STATES = {
        "Alabama", "Alaska", "Arizona", "Arkansas", "California",
        "Colorado", "Connecticut", "Delaware", "Florida", "Georgia",
        "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa",
        "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland",
        "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri",
        "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey",
        "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
        "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina",
        "South Dakota", "Tennessee", "Texas", "Utah", "Vermont",
        "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"
    };

    // The combined text of all state names separated by spaces
    private String text;

    /**
     * Constructor - initializes the text by joining all state names
     */
    public BoyerMooreStateSearch() {
        // Join all state names with spaces to create a single searchable text
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < STATES.length; i++) {
            sb.append(STATES[i]);
            if (i < STATES.length - 1) {
                sb.append(" ");
            }
        }
        this.text = sb.toString();
    }

    /**
     * Preprocesses the pattern to create the bad character table.
     * The table stores the rightmost occurrence of each character in the pattern.
     * 
     * @param pattern The pattern to preprocess
     * @return A map containing the rightmost index of each character in the pattern
     */
    private Map<Character, Integer> buildBadCharacterTable(String pattern) {
        Map<Character, Integer> badCharTable = new HashMap<>();
        int patternLength = pattern.length();

        // For each character in the pattern, store its rightmost position
        for (int i = 0; i < patternLength; i++) {
            badCharTable.put(pattern.charAt(i), i);
        }

        return badCharTable;
    }

    /**
     * Searches for a pattern in the text using the Boyer-Moore algorithm
     * with the bad character rule.
     * 
     * @param pattern The pattern to search for (case-insensitive)
     * @return A list of starting indices where the pattern was found
     */
    public List<Integer> boyerMooreSearch(String pattern) {
        List<Integer> matches = new ArrayList<>();

        // Convert both text and pattern to lowercase for case-insensitive search
        String lowerText = text.toLowerCase();
        String lowerPattern = pattern.toLowerCase();

        int textLength = lowerText.length();
        int patternLength = lowerPattern.length();

        // Edge case: pattern is empty or longer than text
        if (patternLength == 0 || patternLength > textLength) {
            return matches;
        }

        // Build the bad character table for the pattern
        Map<Character, Integer> badCharTable = buildBadCharacterTable(lowerPattern);

        // Start comparing from the beginning of the text
        int shift = 0;

        while (shift <= textLength - patternLength) {
            // Start comparing from the rightmost character of the pattern
            int j = patternLength - 1;

            // Keep moving left while characters match
            while (j >= 0 && lowerPattern.charAt(j) == lowerText.charAt(shift + j)) {
                j--;
            }

            if (j < 0) {
                // Pattern found at current shift position
                matches.add(shift);

                // Shift the pattern to align the next character in the text
                // with its last occurrence in the pattern
                if (shift + patternLength < textLength) {
                    char nextChar = lowerText.charAt(shift + patternLength);
                    Integer lastOccurrence = badCharTable.get(nextChar);
                    if (lastOccurrence != null) {
                        shift += patternLength - lastOccurrence;
                    } else {
                        shift += patternLength + 1;
                    }
                } else {
                    shift += 1;
                }
            } else {
                // Mismatch occurred - use bad character rule to determine shift
                char mismatchChar = lowerText.charAt(shift + j);
                Integer lastOccurrence = badCharTable.get(mismatchChar);

                if (lastOccurrence != null) {
                    // Character exists in pattern - align it with the mismatch position
                    // Ensure we always shift by at least 1
                    shift += Math.max(1, j - lastOccurrence);
                } else {
                    // Character doesn't exist in pattern - shift past the mismatch
                    shift += j + 1;
                }
            }
        }

        return matches;
    }

    /**
     * Displays the text content (all 50 state names)
     */
    public void displayText() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEXT CONTENT: Names of the 50 US States");
        System.out.println("=".repeat(60));
        System.out.println(text);
        System.out.println("=".repeat(60));
        System.out.println("Total length: " + text.length() + " characters");
        System.out.println();
    }

    /**
     * Handles the search operation by prompting for a pattern and displaying results
     * 
     * @param scanner The scanner object for reading user input
     */
    public void performSearch(Scanner scanner) {
        System.out.print("\nEnter a pattern to search for: ");
        String pattern = scanner.nextLine().trim();

        if (pattern.isEmpty()) {
            System.out.println("Error: Pattern cannot be empty.");
            return;
        }

        System.out.println("\nSearching for pattern: \"" + pattern + "\"");
        System.out.println("Using Boyer-Moore Algorithm with Bad Character Rule");
        System.out.println("-".repeat(50));

        // Perform the search
        List<Integer> matches = boyerMooreSearch(pattern);

        // Display results
        if (matches.isEmpty()) {
            System.out.println("No matches found for pattern \"" + pattern + "\"");
        } else {
            System.out.println("Pattern \"" + pattern + "\" found at " + 
                             matches.size() + " location(s):");
            System.out.println("Indices: " + matches);
            
            // Show context for each match
            System.out.println("\nMatches in context:");
            for (int index : matches) {
                // Calculate context boundaries
                int start = Math.max(0, index - 10);
                int end = Math.min(text.length(), index + pattern.length() + 10);
                
                String prefix = (start > 0) ? "..." : "";
                String suffix = (end < text.length()) ? "..." : "";
                String context = text.substring(start, end);
                
                System.out.println("  Index " + index + ": " + prefix + context + suffix);
            }
        }
        System.out.println();
    }

    /**
     * Displays the main menu
     */
    public void displayMenu() {
        System.out.println("\n" + "=".repeat(40));
        System.out.println("   BOYER-MOORE STATE SEARCH APPLICATION");
        System.out.println("=".repeat(40));
        System.out.println("1) Display the text");
        System.out.println("2) Search");
        System.out.println("3) Exit program");
        System.out.println("-".repeat(40));
        System.out.print("Please select an option (1-3): ");
    }

    /**
     * Main method - runs the application
     */
    public static void main(String[] args) {
        BoyerMooreStateSearch app = new BoyerMooreStateSearch();
        Scanner scanner = new Scanner(System.in);
        boolean running = true;

        System.out.println("\nWelcome to the Boyer-Moore State Search Application!");
        System.out.println("This application searches for patterns in US state names");
        System.out.println("using the Bad Character Rule of the Boyer-Moore algorithm.");

        while (running) {
            app.displayMenu();
            String input = scanner.nextLine().trim();

            switch (input) {
                case "1":
                    app.displayText();
                    break;
                case "2":
                    app.performSearch(scanner);
                    break;
                case "3":
                    System.out.println("\nThank you for using the application. Goodbye!");
                    running = false;
                    break;
                default:
                    System.out.println("\nInvalid option. Please enter 1, 2, or 3.");
            }
        }

        scanner.close();
    }
}