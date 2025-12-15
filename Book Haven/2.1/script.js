// ============================================
// Book Haven - JavaScript Functionality
// ============================================

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {

    // ============================================
    // Mobile Menu Toggle
    // ============================================
    const mobileMenuBtn = document.querySelector('.mobile-menu-btn');
    const navMobile = document.querySelector('.nav-mobile');

    if (mobileMenuBtn) {
        mobileMenuBtn.addEventListener('click', function() {
            navMobile.classList.toggle('active');
        });
    }

    // ============================================
    // Subscribe Button (All Pages)
    // ============================================
    const subscribeBtn = document.getElementById('subscribeBtn');

    if (subscribeBtn) {
        subscribeBtn.addEventListener('click', function() {
            alert('Thank you for subscribing.');
        });
    }

    // ============================================
    // Shopping Cart - sessionStorage Functions
    // ============================================

    // Get cart from sessionStorage
    function getCart() {
        const cart = sessionStorage.getItem('bookHavenCart');
        return cart ? JSON.parse(cart) : [];
    }

    // Save cart to sessionStorage
    function saveCart(cart) {
        sessionStorage.setItem('bookHavenCart', JSON.stringify(cart));
    }

    // Add item to cart
    function addToCart(productName, productPrice) {
        const cart = getCart();
        const existingItem = cart.find(item => item.name === productName);

        if (existingItem) {
            existingItem.quantity += 1;
        } else {
            cart.push({
                name: productName,
                price: parseFloat(productPrice),
                quantity: 1
            });
        }

        saveCart(cart);
    }

    // Clear cart from sessionStorage
    function clearCart() {
        sessionStorage.removeItem('bookHavenCart');
    }

    // Calculate cart total
    function getCartTotal() {
        const cart = getCart();
        return cart.reduce((total, item) => total + (item.price * item.quantity), 0);
    }

    // Display cart items in modal
    function displayCart() {
        const cart = getCart();
        const cartItemsContainer = document.getElementById('cartItems');

        if (!cartItemsContainer) return;

        if (cart.length === 0) {
            cartItemsContainer.innerHTML = '<p class="empty-cart">Your cart is empty.</p>';
        } else {
            let cartHTML = '';
            cart.forEach(function(item) {
                cartHTML += '<div class="cart-item">';
                cartHTML += '<span class="item-name">' + item.name + '</span>';
                cartHTML += '<span class="item-quantity">Qty: ' + item.quantity + '</span>';
                cartHTML += '<span class="item-price">$' + (item.price * item.quantity).toFixed(2) + '</span>';
                cartHTML += '</div>';
            });
            cartHTML += '<div class="cart-total">';
            cartHTML += '<strong>Total: $' + getCartTotal().toFixed(2) + '</strong>';
            cartHTML += '</div>';
            cartItemsContainer.innerHTML = cartHTML;
        }
    }

    // ============================================
    // Gallery Page - Cart Buttons
    // ============================================

    // Add to Cart buttons
    const addToCartBtns = document.querySelectorAll('.btn-add-cart');

    addToCartBtns.forEach(function(btn) {
        btn.addEventListener('click', function() {
            const productName = this.getAttribute('data-product');
            const productPrice = this.getAttribute('data-price');

            addToCart(productName, productPrice);
            alert('Item added to the cart.');
        });
    });

    // View Cart buttons (header and hero)
    const viewCartBtn = document.getElementById('viewCartBtn');
    const viewCartBtnHero = document.getElementById('viewCartBtnHero');
    const viewCartSidebar = document.getElementById('viewCartSidebar');
    const cartModal = document.getElementById('cartModal');
    const closeCartBtn = document.getElementById('closeCartBtn');

    function openCartModal(e) {
        if (e) e.preventDefault();
        displayCart();
        if (cartModal) {
            cartModal.classList.add('active');
        }
    }

    function closeCartModal() {
        if (cartModal) {
            cartModal.classList.remove('active');
        }
    }

    if (viewCartBtn) {
        viewCartBtn.addEventListener('click', openCartModal);
    }

    if (viewCartBtnHero) {
        viewCartBtnHero.addEventListener('click', openCartModal);
    }

    if (viewCartSidebar) {
        viewCartSidebar.addEventListener('click', openCartModal);
    }

    if (closeCartBtn) {
        closeCartBtn.addEventListener('click', closeCartModal);
    }

    // Close modal when clicking outside
    if (cartModal) {
        cartModal.addEventListener('click', function(e) {
            if (e.target === cartModal) {
                closeCartModal();
            }
        });
    }

    // Clear Cart function handler
    function handleClearCart() {
        clearCart();
        displayCart();
        alert('Cart cleared.');
    }

    // Process Order function handler
    function handleProcessOrder() {
        const cart = getCart();
        if (cart.length === 0) {
            alert('Your cart is empty. Please add items before processing.');
        } else {
            clearCart();
            displayCart();
            closeCartModal();
            alert('Thank you for your order.');
        }
    }

    // Clear Cart buttons (gallery page and modal)
    const clearCartBtn = document.getElementById('clearCartBtn');
    const modalClearCartBtn = document.getElementById('modalClearCartBtn');

    if (clearCartBtn) {
        clearCartBtn.addEventListener('click', handleClearCart);
    }

    if (modalClearCartBtn) {
        modalClearCartBtn.addEventListener('click', handleClearCart);
    }

    // Process Order buttons (gallery page and modal)
    const processOrderBtn = document.getElementById('processOrderBtn');
    const modalProcessOrderBtn = document.getElementById('modalProcessOrderBtn');

    if (processOrderBtn) {
        processOrderBtn.addEventListener('click', handleProcessOrder);
    }

    if (modalProcessOrderBtn) {
        modalProcessOrderBtn.addEventListener('click', handleProcessOrder);
    }

    // ============================================
    // About Page - Contact Form with localStorage
    // ============================================
    const contactForm = document.getElementById('contactForm');

    if (contactForm) {
        contactForm.addEventListener('submit', function(e) {
            e.preventDefault();

            // Get form values
            const email = document.getElementById('email').value;
            const message = document.getElementById('message').value;

            // Create order object
            const customOrder = {
                email: email,
                message: message,
                timestamp: new Date().toISOString()
            };

            // Get existing orders from localStorage or create new array
            const existingOrders = localStorage.getItem('bookHavenOrders');
            const orders = existingOrders ? JSON.parse(existingOrders) : [];

            // Add new order to array
            orders.push(customOrder);

            // Save to localStorage
            localStorage.setItem('bookHavenOrders', JSON.stringify(orders));

            alert('Thank you for your message.');
            contactForm.reset();
        });
    }

});
