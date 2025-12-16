import java.util.ArrayList;
import java.util.Collection;
import java.util.Scanner;
import java.util.Collections;

public class BlackJack {
    static Scanner scanner = new Scanner(System.in);

    static final String HEARTS   = "\u2665";
    static final String DIAMONDS = "\u2666";
    static final String SPADES   = "\u2660";
    static final String CLUBS    = "\u2663";
    static final String BACKSIDE = "backside";

    public static void main(String[] args) {

        System.out.println("BlackJack by Jordan Moldovan");

        int money = 5000;
        while (money > 0) {

            if (money <= 0){
                System.out.println("Oh No, You're broke");
                System.out.println("Better Luck Next Time");
                System.exit(0);
            }

            System.out.println("Money: " + money );

            int bet = getBet(money);
            ArrayList<String[]> deck = new ArrayList<> (getDeck());
            ArrayList<String[]> dealerHand = new ArrayList<>();
            dealerHand.add(deck.remove(deck.size() - 1));
            dealerHand.add(deck.remove(deck.size() - 1));
            ArrayList<String[]> playerHand = new ArrayList<>();
            playerHand.add(deck.remove(deck.size() - 1));
            playerHand.add(deck.remove(deck.size() - 1));

            System.out.println("Bet: " + bet);
            while (true) {
                displayHands(playerHand, dealerHand, false);
                System.out.println();

                if (getHandValue(playerHand) > 21) {
                    break;
                }

                String move = getMove(playerHand, money - bet);

                if (move.equals("D")){
                    int additionalBet = getBet(Math.min(bet, (money - bet)));
                    bet += additionalBet;
                    System.out.println("Bet increased to " + bet);
                    System.out.println("Bet: " + bet);
                }

                if (move.equals("H")  || move.equals("D")){
                    String[] newCard = deck.remove(deck.size() -1);
                    String rank = newCard[0];
                    String suit = newCard[1];

                    System.out.println("You drew a " + rank + " of " + suit);
                    playerHand.add(newCard);

                    if (getHandValue(playerHand) > 21) {
                        continue;
                    }

                    if (move.equals("S") || move.equals("D")){
                        break;
                    }
                }
            }
        
            if (getHandValue(playerHand) <= 21) {
                while (getHandValue(dealerHand) < 17){
                    System.out.println("Dealer hits");
                    dealerHand.add(deck.remove(deck.size() -1 ));
                    displayHands(playerHand, dealerHand, false);

                    if (getHandValue(dealerHand) > 21) {
                        System.out.print("Press Enter to continue");
                        scanner.nextLine();
                        System.out.println("\n\n");
                        break;
                    }
                }
            }

            displayHands(playerHand, dealerHand, true);

            int playerHandValue = getHandValue(playerHand);
            int dealerHandValue = getHandValue(dealerHand);

            if (dealerHandValue > 21){
                System.out.println("Dealer busts, you win: " + bet);
                money += bet;
            }
            else if ((playerHandValue > 21) || (dealerHandValue > playerHandValue)){
                System.out.println("You bust, you lose: " + bet);
                money -= bet;
            }
            else if (playerHandValue > dealerHandValue){
                System.out.println("You beat the dealer and won: " + bet);
                money += bet;
            }
            else if (playerHandValue == dealerHandValue){
                System.out.println("TIE, no winners");
            }
            System.out.print("Press Enter to continue");
            scanner.nextLine();
            System.out.println("\n\n");
        }
    }

    public static int getBet(int maxBet) {
        while (true) {
            System.out.println("Enter your bet: (1-" + maxBet + " or QUIT");
            String bet = scanner.nextLine().toUpperCase();
            if (bet.equals("QUIT")) {
                System.out.println("Thanks for playing!");
                System.exit(0);
            }
            if (!bet.matches("\\d+")) {
                System.out.println("Invalid bet. Please enter a number.");
                continue;
            }
            int betAmount = Integer.parseInt(bet);
            if (1 <= betAmount && betAmount <= maxBet) {
                return betAmount;
            }
        }
    }
    public static ArrayList<String[]> getDeck() {
        ArrayList<String[]> deck = new ArrayList<>();
        String[] suits = {HEARTS, DIAMONDS, SPADES, CLUBS};
        
        for (String suit : suits) {
            for (int rank = 2; rank <= 10; rank++) {
                deck.add(new String[]{String.valueOf(rank), suit});
            }
            for (String rank : new String[]{"J", "Q", "K", "A"}) {
                deck.add(new String[]{rank, suit});
            }
        }
        
        Collections.shuffle(deck);
        return deck;
    }
    public static void displayHands(ArrayList<String[]> playerHand, ArrayList<String[]> dealerHand, boolean showDealerHand) {
        System.out.println();

        if (showDealerHand) {
            System.out.println("DEALER: " + getHandValue(dealerHand));
            displayCards(dealerHand);
        } else {
            System.out.println("DEALER: ???");
            ArrayList<String[]> hiddenHand = new ArrayList<>();
            hiddenHand.add(new String[]{BACKSIDE});  
            for (int i = 1; i < dealerHand.size(); i++) {
                hiddenHand.add(dealerHand.get(i));   
            }
            displayCards(hiddenHand);
        }
        
        System.out.println("PLAYER: " + getHandValue(playerHand));
        displayCards(playerHand);
    }

    static int getHandValue(ArrayList<String[]> cards) {
        int value = 0;
        int numberOfAces = 0;
        
        for (String[] card : cards) {
            String rank = card[0];
            if (rank.equals("A")) {
                numberOfAces++;
            } else if (rank.equals("K") || rank.equals("Q") || rank.equals("J")) {
                value += 10;
            } else {
                value += Integer.parseInt(rank);
            }
        }
        
        value += numberOfAces;  
        for (int i = 0; i < numberOfAces; i++) {
            if (value + 10 <= 21) {
                value += 10;
            }
        }
        
        return value;
    }

    static void displayCards(ArrayList<String[]> cards) {
    String[] rows = {"", "", "", "", ""};
    
    for (String[] card : cards) {
        rows[0] += " ___  ";
        
        if (card[0].equals(BACKSIDE)) {
            rows[1] += "|## | ";
            rows[2] += "|###| ";
            rows[3] += "|_##| ";
        } else {
            String rank = card[0];
            String suit = card[1];
            rows[1] += "|" + String.format("%-2s", rank) + "| ";
            rows[2] += "| " + suit + " | ";
            rows[3] += "|_" + padLeft(rank, 2, '_') + "| ";
        }
    }
    
    for (String row : rows) {
        System.out.println(row);
    }
}

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

static String getMove(ArrayList<String[]> playerHand, int money) {
    while (true) {
        ArrayList<String> moves = new ArrayList<>();
        moves.add("(H)it");
        moves.add("(S)tand");
        

        if (playerHand.size() == 2 && money > 0) {
            moves.add("(D)ouble down");
        }
        
        String movePrompt = String.join(", ", moves) + "> ";
        System.out.print(movePrompt);
        String move = scanner.nextLine().toUpperCase();
        
        if (move.equals("H") || move.equals("S")) {
            return move;
        }
        if (move.equals("D") && moves.contains("(D)ouble down")) {
            return move;
        }
    }
}

}
