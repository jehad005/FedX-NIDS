/* SecureFed-NIDS HQ Light-First Theme Controller */
(function () {
    const saved = localStorage.getItem('hq_theme');
    if (saved === 'dark') {
        document.documentElement.classList.add('dark');
    } else {
        document.documentElement.classList.remove('dark');
    }
})();

function syncHqThemeIcon() {
    const isDark = document.documentElement.classList.contains('dark');
    const icon = document.getElementById('hq-theme-icon');
    const btn = icon ? icon.closest('.theme-toggle-btn') : null;
    if (icon) {
        icon.setAttribute('icon-name', isDark ? 'sun' : 'moon');
    }
    if (btn) {
        btn.setAttribute('title', isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode');
        btn.setAttribute('aria-label', isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode');
    }
}

function toggleHqTheme() {
    const html = document.documentElement;
    const icon = document.getElementById('hq-theme-icon');
    const btn = icon ? icon.closest('.theme-toggle-btn') : null;

    if (btn) {
        btn.classList.remove('is-spinning');
        void btn.offsetWidth;
        btn.classList.add('is-spinning');
    }

    if (html.classList.contains('dark')) {
        html.classList.remove('dark');
        localStorage.setItem('hq_theme', 'light');
    } else {
        html.classList.add('dark');
        localStorage.setItem('hq_theme', 'dark');
    }
    syncHqThemeIcon();
}

document.addEventListener('DOMContentLoaded', syncHqThemeIcon);
