import java.util.ArrayList;
import java.util.Scanner;
import java.util.Collections;

public class BlackJack {
    // Scanner for reading user input
    static Scanner scanner = new Scanner(System.in);

    // Unicode symbols for card suits
    static final String HEARTS   = "\u2665";
    static final String DIAMONDS = "\u2666";
    static final String SPADES   = "\u2660";
    static final String CLUBS    = "\u2663";
    static final String BACKSIDE = "backside";  // Used to represent a hidden card

    public static void main(String[] args) {

        System.out.println("BlackJack by Jordan Moldovan");

        // Player starts with $5000
        int money = 5000;

        // Main game loop: continues until player runs out of money
        while (money > 0) {

            // Display current balance and get bet from player
            System.out.println("Money: " + money );
            int bet = getBet(money);

            // Create a fresh shuffled deck for each round
            ArrayList<String[]> deck = new ArrayList<> (getDeck());

            // Deal two cards to dealer
            ArrayList<String[]> dealerHand = new ArrayList<>();
            dealerHand.add(deck.remove(deck.size() - 1));
            dealerHand.add(deck.remove(deck.size() - 1));

            // Deal two cards to player
            ArrayList<String[]> playerHand = new ArrayList<>();
            playerHand.add(deck.remove(deck.size() - 1));
            playerHand.add(deck.remove(deck.size() - 1));

            System.out.println("Bet: " + bet);

            //  PLAYER'S TURN 
            while (true) {
                // Show current hands (dealer's first card hidden)
                displayHands(playerHand, dealerHand, false);
                System.out.println();

                // If player already busted, end their turn
                if (getHandValue(playerHand) > 21) {
                    break;
                }

                // Get player's move (pass remaining money to determine if double down is available)
                String move = getMove(playerHand, money - bet);

                // Handle double down  player increases their bet
                if (move.equals("D")){
                    // Additional bet can be up to original bet or remaining money, whichever is less
                    int additionalBet = getBet(Math.min(bet, (money - bet)));
                    bet += additionalBet;
                    System.out.println("Bet increased to " + bet);
                    System.out.println("Bet: " + bet);
                }

                // Handle stand: player keeps their current hand
                if (move.equals("S")) {
                    break;
                }

                // Handle hit or double down: player draws a card
                if (move.equals("H") || move.equals("D")){
                    String[] newCard = deck.remove(deck.size() -1);
                    String rank = newCard[0];
                    String suit = newCard[1];

                    System.out.println("You drew a " + rank + " of " + suit);
                    playerHand.add(newCard);

                    // If player busts, continue to next iteration (which will break)
                    if (getHandValue(playerHand) > 21) {
                        continue;
                    }

                    // Double down only allows one card, so end turn
                    if (move.equals("D")){
                        break;
                    }
                }
            }

            // DEALER'S TURN 
            // Only play dealer's turn if player didn't bust
            if (getHandValue(playerHand) <= 21) {
                // Dealer must hit until they have 17 or more
                while (getHandValue(dealerHand) < 17){
                    System.out.println("Dealer hits");
                    dealerHand.add(deck.remove(deck.size() -1 ));
                    displayHands(playerHand, dealerHand, false);

                    // If dealer busts, end their turn
                    if (getHandValue(dealerHand) > 21) {
                        System.out.print("Press Enter to continue");
                        scanner.nextLine();
                        System.out.println("\n\n");
                        break;
                    }
                }
                // Show message when dealer stands
                if (getHandValue(dealerHand) >= 17 && getHandValue(dealerHand) <= 21) {
                    System.out.println("Dealer stands at " + getHandValue(dealerHand));
                }
            }

            //  DETERMINE WINNER 
            // Show final hands with dealer's hidden card revealed
            displayHands(playerHand, dealerHand, true);

            int playerHandValue = getHandValue(playerHand);
            int dealerHandValue = getHandValue(dealerHand);

            // Check all possible outcomes and adjust money accordingly
            if (dealerHandValue > 21){
                System.out.println("Dealer busts, you win: " + bet);
                money += bet;
            }
            else if (playerHandValue > 21){
                System.out.println("You bust, you lose: " + bet);
                money -= bet;
            }
            else if (dealerHandValue > playerHandValue){
                System.out.println("Dealer wins, you lose: " + bet);
                money -= bet;
            }
            else if (playerHandValue > dealerHandValue){
                System.out.println("You beat the dealer and won: " + bet);
                money += bet;
            }
            else if (playerHandValue == dealerHandValue){
                System.out.println("TIE, no winners");
            }

            // Wait for player before starting next round
            System.out.print("Press Enter to continue");
            scanner.nextLine();
            System.out.println("\n\n");
        }
    }

    /**
     * Prompts the player to enter a bet amount
     */
    public static int getBet(int maxBet) {
        while (true) {
            System.out.println("Enter your bet: (1-" + maxBet + " or QUIT");
            String bet = scanner.nextLine().toUpperCase();

            // Allow player to quit the game
            if (bet.equals("QUIT")) {
                System.out.println("Thanks for playing!");
                System.exit(0);
            }

            // Validate that input is a number
            if (!bet.matches("\\d+")) {
                System.out.println("Invalid bet. Please enter a number.");
                continue;
            }

            // Validate bet is within allowed range
            int betAmount = Integer.parseInt(bet);
            if (1 <= betAmount && betAmount <= maxBet) {
                return betAmount;
            }
        }
    }

    /**
     * Creates and shuffles a standard 52 card deck
     */
    public static ArrayList<String[]> getDeck() {
        ArrayList<String[]> deck = new ArrayList<>();
        String[] suits = {HEARTS, DIAMONDS, SPADES, CLUBS};

        // Add all cards for each suit
        for (String suit : suits) {
            // Add number cards 2-10
            for (int rank = 2; rank <= 10; rank++) {
                deck.add(new String[]{String.valueOf(rank), suit});
            }
            // Add face cards and ace
            for (String rank : new String[]{"J", "Q", "K", "A"}) {
                deck.add(new String[]{rank, suit});
            }
        }

        // Randomize the deck order
        Collections.shuffle(deck);
        return deck;
    }

    /**
     * Displays both hands on screen */
    public static void displayHands(ArrayList<String[]> playerHand, ArrayList<String[]> dealerHand, boolean showDealerHand) {
        System.out.println();

        if (showDealerHand) {
            // Show dealer's full hand with value
            System.out.println("DEALER: " + getHandValue(dealerHand));
            displayCards(dealerHand);
        } else {
            // Hide dealer's first card and show ??? for value
            System.out.println("DEALER: ???");
            ArrayList<String[]> hiddenHand = new ArrayList<>();
            hiddenHand.add(new String[]{BACKSIDE});  // First card shows as hidden
            for (int i = 1; i < dealerHand.size(); i++) {
                hiddenHand.add(dealerHand.get(i));   // Rest of cards visible
            }
            displayCards(hiddenHand);
        }

        // Always show player's full hand
        System.out.println("PLAYER: " + getHandValue(playerHand));
        displayCards(playerHand);
    }

    /**
     * Calculates the value of a hand
     * Aces are worth 11 if it doesn't cause a bust, otherwise 1
     */
    static int getHandValue(ArrayList<String[]> cards) {
        int value = 0;
        int numberOfAces = 0;

        // First pass: count non ace cards and track aces
        for (String[] card : cards) {
            String rank = card[0];
            if (rank.equals("A")) {
                numberOfAces++;
            } else if (rank.equals("K") || rank.equals("Q") || rank.equals("J")) {
                value += 10;  // Face cards worth 10
            } else {
                value += Integer.parseInt(rank);  // Number cards worth face value
            }
        }

        // Second pass: add aces (start with 1 each, upgrade to 11 if possible)
        value += numberOfAces;  // Count each ace as 1 first
        for (int i = 0; i < numberOfAces; i++) {
            if (value + 10 <= 21) {
                value += 10;  // Upgrade ace from 1 to 11 if it won't bust
            }
        }

        return value;
    }

    /**
     * Displays cards as ASCII art
     */
    static void displayCards(ArrayList<String[]> cards) {
        // Build each row of the card display
        String[] rows = {"", "", "", "", ""};

        for (String[] card : cards) {
            rows[0] += " ___  ";  // Top border

            if (card[0].equals(BACKSIDE)) {
                // Hidden card pattern
                rows[1] += "|## | ";
                rows[2] += "|###| ";
                rows[3] += "|_##| ";
            } else {
                // Visible card with rank and suit
                String rank = card[0];
                String suit = card[1];
                rows[1] += "|" + String.format("%-2s", rank) + "| ";  // Rank at top
                rows[2] += "| " + suit + " | ";                        // Suit in middle
                rows[3] += "|_" + padLeft(rank, 2, '_') + "| ";        // Rank at bottom
            }
        }

        // Print all rows
        for (String row : rows) {
            System.out.println(row);
        }
    }

    /**
     * Pads a string on the left with a specified character
     */
    static String padLeft(String str, int length, char padChar) {
        if (str.length() >= length) {
            return str;
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < length - str.length(); i++) {
            sb.append(padChar);
        }
        sb.append(str);
        return sb.toString();
    }

    /**
     * Prompts player for their move and validates the input
     */
    static String getMove(ArrayList<String[]> playerHand, int money) {
        while (true) {
            // Build list of available moves
            ArrayList<String> moves = new ArrayList<>();
            moves.add("(H)it");
            moves.add("(S)tand");

            // Double down only available on first move (2 cards) and if player has extra money
            if (playerHand.size() == 2 && money > 0) {
                moves.add("(D)ouble down");
            }

            // Display options and get input
            String movePrompt = String.join(", ", moves) + "> ";
            System.out.print(movePrompt);
            String move = scanner.nextLine().toUpperCase();

            // Validate and return the move
            if (move.equals("H") || move.equals("S")) {
                return move;
            }
            if (move.equals("D") && moves.contains("(D)ouble down")) {
                return move;
            }
            // Invalid input: loop will continue and prompt again
        }
    }

}
