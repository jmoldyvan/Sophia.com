import java.util.Scanner;


public class BinarySearchTree {
    
    // Node class representing each node in the BST
    static class Node {
        int data;
        Node left;
        Node right;
        
        public Node(int data) {
            this.data = data;
            this.left = null;
            this.right = null;
        }
    }
    
    // Root of the BST
    private Node root;
    
    // Constructor
    public BinarySearchTree() {
        root = null;
    }
    
    /**
     * Creates a balanced BST with values 1-7
     * The tree structure will be:
     *           4
     *         /   \
     *        2     6
     *       / \   / \
     *      1   3 5   7
     */
    public void createBalancedTree() {
        // Clear any existing tree
        root = null;
        
        // Build the balanced tree by inserting in the correct order
        // Insert 4 as root first, then 2, 6, 1, 3, 5, 7 to maintain balance
        int[] values = {4, 2, 6, 1, 3, 5, 7};
        
        for (int value : values) {
            root = insertNode(root, value);
        }
        
        System.out.println("\nBinary search tree created successfully with values 1-7.");
        System.out.println("Tree structure:");
        System.out.println("       4");
        System.out.println("      / \\");
        System.out.println("     2   6");
        System.out.println("    / \\ / \\");
        System.out.println("   1  3 5  7");
    }
    
    /**
     * Inserts a new node with the given value into the BST
     */
    private Node insertNode(Node node, int value) {
        // If the tree/subtree is empty, create a new node
        if (node == null) {
            return new Node(value);
        }
        
        // Otherwise, recur down the tree
        if (value < node.data) {
            node.left = insertNode(node.left, value);
        } else if (value > node.data) {
            node.right = insertNode(node.right, value);
        }
        // If value equals node.data, it's a duplicate - don't insert
        
        return node;
    }
    
    /**
     * Public method to add a node
     */
    public void addNode(int value) {
        if (root == null) {
            System.out.println("\nError: Please create a binary search tree first (Option 1).");
            return;
        }
        
        // Check if value already exists
        if (search(root, value)) {
            System.out.println("\nValue " + value + " already exists in the tree. Duplicates not allowed.");
            return;
        }
        
        root = insertNode(root, value);
        System.out.println("\nNode with value " + value + " added successfully.");
    }
    
    /**
     * Search for a value in the BST
     */
    private boolean search(Node node, int value) {
        if (node == null) {
            return false;
        }
        
        if (value == node.data) {
            return true;
        } else if (value < node.data) {
            return search(node.left, value);
        } else {
            return search(node.right, value);
        }
    }
    
    /**
     * Deletes a node with the given value from the BST
     */
    public void deleteNode(int value) {
        if (root == null) {
            System.out.println("\nError: Please create a binary search tree first (Option 1).");
            return;
        }
        
        // Check if value exists
        if (!search(root, value)) {
            System.out.println("\nValue " + value + " not found in the tree.");
            return;
        }
        
        root = deleteNodeRecursive(root, value);
        System.out.println("\nNode with value " + value + " deleted successfully.");
    }
    
    /**
     * Recursive helper method to delete a node
     */
    private Node deleteNodeRecursive(Node node, int value) {
        if (node == null) {
            return null;
        }
        
        if (value < node.data) {
            // Value is in left subtree
            node.left = deleteNodeRecursive(node.left, value);
        } else if (value > node.data) {
            // Value is in right subtree
            node.right = deleteNodeRecursive(node.right, value);
        } else {
            // This is the node to delete
            
            // Case 1: Node has no children (leaf node)
            if (node.left == null && node.right == null) {
                return null;
            }
            
            // Case 2: Node has only one child
            if (node.left == null) {
                return node.right;
            } else if (node.right == null) {
                return node.left;
            }
            
            // Case 3: Node has two children
            // Find the inorder successor (smallest value in right subtree)
            node.data = findMinValue(node.right);
            // Delete the inorder successor
            node.right = deleteNodeRecursive(node.right, node.data);
        }
        
        return node;
    }
    
    /**
     * Finds the minimum value in a subtree
     */
    private int findMinValue(Node node) {
        int minValue = node.data;
        while (node.left != null) {
            minValue = node.left.data;
            node = node.left;
        }
        return minValue;
    }
    
    /**
     * InOrder traversal: Left -> Root -> Right
     * Prints nodes in ascending order
     */
    public void printInOrder() {
        if (root == null) {
            System.out.println("\nError: Please create a binary search tree first (Option 1).");
            return;
        }
        
        System.out.print("\nInOrder Traversal: ");
        inOrderTraversal(root);
        System.out.println();
    }
    
    private void inOrderTraversal(Node node) {
        if (node != null) {
            inOrderTraversal(node.left);
            System.out.print(node.data + " ");
            inOrderTraversal(node.right);
        }
    }
    
    /**
     * PreOrder traversal: Root -> Left -> Right
     */
    public void printPreOrder() {
        if (root == null) {
            System.out.println("\nError: Please create a binary search tree first (Option 1).");
            return;
        }
        
        System.out.print("\nPreOrder Traversal: ");
        preOrderTraversal(root);
        System.out.println();
    }
    
    private void preOrderTraversal(Node node) {
        if (node != null) {
            System.out.print(node.data + " ");
            preOrderTraversal(node.left);
            preOrderTraversal(node.right);
        }
    }
    
    /**
     * PostOrder traversal: Left -> Right -> Root
     */
    public void printPostOrder() {
        if (root == null) {
            System.out.println("\nError: Please create a binary search tree first (Option 1).");
            return;
        }
        
        System.out.print("\nPostOrder Traversal: ");
        postOrderTraversal(root);
        System.out.println();
    }
    
    private void postOrderTraversal(Node node) {
        if (node != null) {
            postOrderTraversal(node.left);
            postOrderTraversal(node.right);
            System.out.print(node.data + " ");
        }
    }
    
    /**
     * Displays the menu options
     */
    public static void displayMenu() {
        System.out.println("\n========================================");
        System.out.println("    Binary Search Tree Application");
        System.out.println("========================================");
        System.out.println("1. Create a binary search tree");
        System.out.println("2. Add a node");
        System.out.println("3. Delete a node");
        System.out.println("4. Print nodes by InOrder");
        System.out.println("5. Print nodes by PreOrder");
        System.out.println("6. Print nodes by PostOrder");
        System.out.println("7. Exit program");
        System.out.println("========================================");
        System.out.print("Please select an option (1-7): ");
    }
    
    /**
     * Main method - entry point of the application
     */
    public static void main(String[] args) {
        BinarySearchTree bst = new BinarySearchTree();
        Scanner scanner = new Scanner(System.in);
        int choice;
        int value;
        
        System.out.println("\nWelcome to the Binary Search Tree Application!");
        
        do {
            displayMenu();
            
            // Handle non-integer input
            while (!scanner.hasNextInt()) {
                System.out.println("\nInvalid input. Please enter a number between 1 and 7.");
                scanner.next(); // Clear the invalid input
                displayMenu();
            }
            
            choice = scanner.nextInt();
            
            switch (choice) {
                case 1:
                    bst.createBalancedTree();
                    break;
                    
                case 2:
                    System.out.print("\nEnter the value for the new node: ");
                    while (!scanner.hasNextInt()) {
                        System.out.println("Invalid input. Please enter an integer value.");
                        scanner.next();
                        System.out.print("Enter the value for the new node: ");
                    }
                    value = scanner.nextInt();
                    bst.addNode(value);
                    break;
                    
                case 3:
                    System.out.print("\nEnter the value of the node to delete: ");
                    while (!scanner.hasNextInt()) {
                        System.out.println("Invalid input. Please enter an integer value.");
                        scanner.next();
                        System.out.print("Enter the value of the node to delete: ");
                    }
                    value = scanner.nextInt();
                    bst.deleteNode(value);
                    break;
                    
                case 4:
                    bst.printInOrder();
                    break;
                    
                case 5:
                    bst.printPreOrder();
                    break;
                    
                case 6:
                    bst.printPostOrder();
                    break;
                    
                case 7:
                    System.out.println("\nThank you for using the Binary Search Tree Application.");
                    System.out.println("Goodbye!");
                    break;
                    
                default:
                    System.out.println("\nInvalid option. Please select a number between 1 and 7.");
            }
            
        } while (choice != 7);
        
        scanner.close();
    }
}