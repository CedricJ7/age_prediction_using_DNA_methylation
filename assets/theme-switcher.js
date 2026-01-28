/**
 * Theme Switcher - Light/Dark Mode Toggle
 */

// Initialize theme from localStorage or default to dark
const initTheme = () => {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);
};

// Toggle between light and dark themes
const toggleTheme = () => {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeIcon(newTheme);

    // Trigger animation
    document.body.style.transition = 'background-color 0.3s ease, color 0.3s ease';
};

// Update theme icon
const updateThemeIcon = (theme) => {
    const btn = document.getElementById('theme-toggle');
    if (btn) {
        btn.innerHTML = theme === 'dark' ? '☀️' : '🌙';
        btn.setAttribute('aria-label', theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode');
    }
};

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    initTheme();

    // Add click handler to theme toggle button
    const btn = document.getElementById('theme-toggle');
    if (btn) {
        btn.addEventListener('click', toggleTheme);
    }
});

// Re-initialize when Dash re-renders
const observer = new MutationObserver(() => {
    const btn = document.getElementById('theme-toggle');
    if (btn && !btn.hasAttribute('data-initialized')) {
        btn.setAttribute('data-initialized', 'true');
        btn.addEventListener('click', toggleTheme);
        const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
        updateThemeIcon(currentTheme);
    }
});

observer.observe(document.body, {
    childList: true,
    subtree: true
});
