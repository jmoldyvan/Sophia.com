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
    // Gallery Page - Cart Buttons
    // ============================================

    // Add to Cart buttons
    const addToCartBtns = document.querySelectorAll('.btn-add-cart');

    addToCartBtns.forEach(function(btn) {
        btn.addEventListener('click', function() {
            alert('Item added to the cart.');
        });
    });

    // Clear Cart button
    const clearCartBtn = document.getElementById('clearCartBtn');

    if (clearCartBtn) {
        clearCartBtn.addEventListener('click', function() {
            alert('Cart cleared.');
        });
    }

    // Process Order button
    const processOrderBtn = document.getElementById('processOrderBtn');

    if (processOrderBtn) {
        processOrderBtn.addEventListener('click', function() {
            alert('Thank you for your order.');
        });
    }

    // ============================================
    // About Page - Contact Form
    // ============================================
    const contactForm = document.getElementById('contactForm');

    if (contactForm) {
        contactForm.addEventListener('submit', function(e) {
            e.preventDefault();
            alert('Thank you for your message.');
            contactForm.reset();
        });
    }

});
