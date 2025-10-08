document.addEventListener('DOMContentLoaded', () => {
    
    // Get references to DOM elements
    const mainContainer = document.getElementById('main-container');
    const loadingIndicator = document.getElementById('loading-indicator');
    const headerContent = document.getElementById('header-content');
    const linksContainer = document.getElementById('links-container');
    const widgetsContainer = document.getElementById('widgets-container');
    const stickerContainer = document.getElementById('sticker-container');

    // Main function to fetch data and build the page
    async function initializePage() {
        initParticles(); 

        try {
            const response = await fetch('data.json');
            if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
            
            const data = await response.json();
            
            // Build all page components from the data
            buildHeader(data.name, data.intro);
            buildLinks(data.links);
            buildWidgets(data.widgets);
            createStickers(data.draggableStickers);

            // Show content and hide loader
            mainContainer.classList.add('loaded');
            loadingIndicator.style.display = 'none';

        } catch (error) {
            console.error("Could not fetch or process data:", error);
            loadingIndicator.style.display = 'none';
            headerContent.innerHTML = '<h1>Oops!</h1><p>Could not load page content. Please try again later.</p>';
            mainContainer.classList.add('loaded');

        }
    }

    function initParticles() {
        tsParticles.load("tsparticles", {
            fpsLimit: 60,
            interactivity: {
                events: {
                    onHover: { enable: true, mode: "repulse" },
                    resize: true,
                },
                modes: {
                    repulse: { distance: 100, duration: 0.4 },
                },
            },
            particles: {
                color: { value: "#ffffff" },
                links: { color: "#ffffff", distance: 150, enable: true, opacity: 0.1, width: 1 },
                move: {
                    direction: "none",
                    enable: true,
                    outModes: { default: "bounce" },
                    random: false,
                    speed: 1,
                    straight: false,
                },
                number: { density: { enable: true, area: 800 }, value: 80 },
                opacity: { value: 0.2 },
                shape: { type: "circle" },
                size: { value: { min: 1, max: 5 } },
            },
            detectRetina: true,
        });
    }

    function buildHeader(name, intro) {
        headerContent.innerHTML = `<h1>${name}</h1><p>${intro}</p>`;
    }

    function buildLinks(links) {
        links.forEach(link => {
            const linkElement = document.createElement('a');
            linkElement.href = link.url;
            linkElement.target = '_blank';
            linkElement.classList.add('link-note');
            linkElement.innerHTML = `<h3>${link.title}</h3><p>${link.description}</p>`;
            linksContainer.appendChild(linkElement);
        });
    }
    
    function buildWidgets(widgets) {
        if (widgets.spotify && widgets.spotify.enabled) {
            const spotifyBox = document.createElement('div');
            spotifyBox.classList.add('widget-box');
            let content = `<h2>${widgets.spotify.title}</h2>`;
            if (widgets.spotify.iframeSrc && !widgets.spotify.iframeSrc.includes('YOUR_SPOTIFY_EMBED_URL_HERE')) {
                content += `<iframe src="${widgets.spotify.iframeSrc}" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"></iframe>`;
            } else {
                content += `<p>(Paste your Spotify embed URL in data.json)</p>`;
            }
            spotifyBox.innerHTML = content;
            widgetsContainer.appendChild(spotifyBox);
        }
        
        if (widgets.guestbook && widgets.guestbook.enabled) {
            const guestbookBox = document.createElement('div');
            guestbookBox.classList.add('widget-box');
            guestbookBox.innerHTML = `<h2>${widgets.guestbook.title}</h2><p>(Your Firebase guestbook goes here)</p>`;
            widgetsContainer.appendChild(guestbookBox);
        }
    }

    function createStickers(stickers) {
        stickers.forEach(stickerSrc => {
            const sticker = document.createElement('img');
            sticker.src = stickerSrc;
            sticker.classList.add('draggable-sticker');
            const padding = 100;
            sticker.style.top = `${Math.random() * (window.innerHeight - padding)}px`;
            sticker.style.left = `${Math.random() * (window.innerWidth - padding)}px`;
            stickerContainer.appendChild(sticker);
            makeDraggable(sticker);
        });
    }

    function makeDraggable(element) {
        let isDragging = false, offsetX, offsetY;
        element.addEventListener('mousedown', (e) => {
            isDragging = true;
            offsetX = e.clientX - element.getBoundingClientRect().left;
            offsetY = e.clientY - element.getBoundingClientRect().top;
            element.style.zIndex = 1000;
        });
        document.addEventListener('mousemove', (e) => {
            if (isDragging) {
                element.style.left = `${e.clientX - offsetX}px`;
                element.style.top = `${e.clientY - offsetY}px`;
            }
        });
        document.addEventListener('mouseup', () => {
            isDragging = false;
            element.style.zIndex = 5;
        });
    }
    
    // Start the process
    initializePage();
});