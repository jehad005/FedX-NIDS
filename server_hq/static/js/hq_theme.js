/* SecureFed-NIDS HQ Dark / Light Theme Controller */
(function () {
    const saved = localStorage.getItem('theme');
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    if (saved === 'dark' || (!saved && prefersDark)) {
        document.documentElement.classList.add('dark');
    } else {
        document.documentElement.classList.remove('dark');
    }
})();

function syncHqThemeIcon() {
    const isDark = document.documentElement.classList.contains('dark');
    const icon = document.getElementById('hq-theme-icon');
    if (icon) {
        icon.setAttribute('icon-name', isDark ? 'sun' : 'moon');
    }
}

function toggleHqTheme() {
    const html = document.documentElement;
    if (html.classList.contains('dark')) {
        html.classList.remove('dark');
        localStorage.setItem('theme', 'light');
    } else {
        html.classList.add('dark');
        localStorage.setItem('theme', 'dark');
    }
    syncHqThemeIcon();
}

document.addEventListener('DOMContentLoaded', syncHqThemeIcon);
